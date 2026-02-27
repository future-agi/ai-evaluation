/**
 * Content analyzer for AutoEval.
 * Detects input type and characteristics to select appropriate metrics.
 *
 * @module local/autoeval/analyzer
 */

import {
    AutoEvalInput,
    ContentCharacteristics,
    InputType,
    CODE_LANGUAGE_PATTERNS
} from './types';

/**
 * Analyze input content to determine characteristics.
 *
 * @param input - The input to analyze
 * @returns Detected content characteristics
 */
export function analyzeContent(input: AutoEvalInput): ContentCharacteristics {
    const { query, response, context, reference, code } = input;

    // Detect JSON
    const hasJson = detectJson(response);

    // Detect code - check explicit code field first, then response
    let codeDetection: { hasCode: boolean; language?: string };
    if (code && code.trim().length > 0) {
        // Explicit code field always means we have code
        codeDetection = detectCode(code);
        codeDetection.hasCode = true; // Force true when explicit code is provided
    } else {
        codeDetection = detectCode(response);
    }
    const hasCode = codeDetection.hasCode;
    const codeLanguage = codeDetection.language;

    // Check for context
    const hasContext = !!(context && (
        (Array.isArray(context) && context.length > 0) ||
        (typeof context === 'string' && context.trim().length > 0)
    ));

    // Check for reference
    const hasReference = !!(reference && reference.trim().length > 0);

    // Check if Q&A format
    const isQA = !!(query && query.trim().length > 0);

    // Determine length category
    const lengthCategory = categorizeLength(response);

    // Detect domains
    const domains = detectDomains(response, query);

    // Determine input type
    const inputType = determineInputType({
        hasJson,
        hasCode,
        hasContext,
        hasReference,
        isQA,
        code: !!code
    });

    return {
        inputType,
        hasJson,
        hasCode,
        codeLanguage,
        hasContext,
        hasReference,
        isQA,
        lengthCategory,
        domains
    };
}

/**
 * Detect if text contains valid JSON
 */
function detectJson(text: string): boolean {
    // Check for JSON-like patterns
    const jsonPatterns = [
        /^\s*\{[\s\S]*\}\s*$/,    // Object
        /^\s*\[[\s\S]*\]\s*$/,    // Array
    ];

    if (jsonPatterns.some(p => p.test(text))) {
        try {
            JSON.parse(text);
            return true;
        } catch {
            // Not valid JSON
        }
    }

    // Check for embedded JSON
    const embeddedJson = /```json\s*([\s\S]*?)```/.exec(text);
    if (embeddedJson) {
        try {
            JSON.parse(embeddedJson[1]);
            return true;
        } catch {
            // Not valid JSON
        }
    }

    return false;
}

/**
 * Detect if text contains code and identify the language
 */
function detectCode(text: string): { hasCode: boolean; language?: string } {
    // Check for code blocks
    const codeBlockMatch = /```(\w+)?\s*([\s\S]*?)```/.exec(text);
    if (codeBlockMatch) {
        const specifiedLang = codeBlockMatch[1]?.toLowerCase();
        if (specifiedLang && specifiedLang !== 'text' && specifiedLang !== 'plaintext') {
            return { hasCode: true, language: specifiedLang };
        }
    }

    // Try to detect language from patterns
    let bestMatch: { language: string; score: number } | null = null;

    for (const [language, patterns] of Object.entries(CODE_LANGUAGE_PATTERNS)) {
        let matchCount = 0;
        for (const pattern of patterns) {
            if (pattern.test(text)) {
                matchCount++;
            }
        }

        const score = matchCount / patterns.length;
        if (score > 0.2 && (!bestMatch || score > bestMatch.score)) {
            bestMatch = { language, score };
        }
    }

    if (bestMatch) {
        return { hasCode: true, language: bestMatch.language };
    }

    // Check for generic code indicators
    const genericCodePatterns = [
        /\bfunction\b.*\(/,
        /\bclass\b.*\{/,
        /\bif\s*\(.*\)\s*\{/,
        /\bfor\s*\(.*\)\s*\{/,
        /\bwhile\s*\(.*\)\s*\{/,
        /\breturn\b.*;/,
        /\bimport\b.*\bfrom\b/,
        /\bconst\s+\w+\s*=/,        // JavaScript/TypeScript const
        /\blet\s+\w+\s*=/,          // JavaScript/TypeScript let
        /\bvar\s+\w+\s*=/,          // JavaScript var
        /=>\s*\{/,                   // Arrow function
        /;\s*$/m,                    // Statement ending with semicolon
    ];

    const genericMatches = genericCodePatterns.filter(p => p.test(text)).length;
    if (genericMatches >= 1) {
        // Even a single strong indicator suggests code
        return { hasCode: true, language: 'javascript' };
    }

    return { hasCode: false };
}

/**
 * Categorize response length
 */
function categorizeLength(text: string): 'short' | 'medium' | 'long' {
    const wordCount = text.split(/\s+/).length;

    if (wordCount < 50) return 'short';
    if (wordCount < 200) return 'medium';
    return 'long';
}

/**
 * Detect content domains
 */
function detectDomains(response: string, query?: string): string[] {
    const domains: string[] = [];
    const combined = `${query || ''} ${response}`.toLowerCase();

    const domainPatterns: Record<string, RegExp[]> = {
        'technical': [
            /\bapi\b/, /\bfunction\b/, /\bclass\b/, /\bdatabase\b/,
            /\bserver\b/, /\bclient\b/, /\bprotocol\b/
        ],
        'medical': [
            /\bdiagnosis\b/, /\bsymptom\b/, /\btreatment\b/, /\bpatient\b/,
            /\bmedicine\b/, /\bdisease\b/, /\bdoctor\b/
        ],
        'legal': [
            /\bcontract\b/, /\blaw\b/, /\blegal\b/, /\bcourt\b/,
            /\bliability\b/, /\battorney\b/, /\bstatute\b/
        ],
        'financial': [
            /\binvestment\b/, /\bstock\b/, /\bmarket\b/, /\bfinance\b/,
            /\bbanking\b/, /\bcurrency\b/, /\bportfolio\b/
        ],
        'scientific': [
            /\bresearch\b/, /\bexperiment\b/, /\bhypothesis\b/, /\bdata\b/,
            /\banalysis\b/, /\bscientific\b/, /\bstudy\b/
        ],
        'educational': [
            /\blearn\b/, /\bteach\b/, /\beducation\b/, /\bstudent\b/,
            /\bcourse\b/, /\blesson\b/, /\bexplain\b/
        ],
    };

    for (const [domain, patterns] of Object.entries(domainPatterns)) {
        const matchCount = patterns.filter(p => p.test(combined)).length;
        if (matchCount >= 2) {
            domains.push(domain);
        }
    }

    return domains;
}

/**
 * Determine the input type based on characteristics
 */
function determineInputType(characteristics: {
    hasJson: boolean;
    hasCode: boolean;
    hasContext: boolean;
    hasReference: boolean;
    isQA: boolean;
    code: boolean;
}): InputType {
    const { hasJson, hasCode, hasContext, hasReference, isQA, code } = characteristics;

    // Explicit code input
    if (code) {
        return 'code';
    }

    // JSON response
    if (hasJson && !hasCode) {
        return 'json';
    }

    // Code response
    if (hasCode) {
        return 'code';
    }

    // RAG response (has context)
    if (hasContext) {
        return 'rag';
    }

    // Q&A with reference
    if (isQA && hasReference) {
        return 'qa';
    }

    // Q&A without reference
    if (isQA) {
        return 'qa';
    }

    // Plain text
    return 'text';
}

/**
 * Get a human-readable description of the detected characteristics
 */
export function describeCharacteristics(chars: ContentCharacteristics): string {
    const parts: string[] = [];

    parts.push(`Input type: ${chars.inputType}`);

    if (chars.hasCode && chars.codeLanguage) {
        parts.push(`Code language: ${chars.codeLanguage}`);
    }

    if (chars.hasContext) {
        parts.push('Has context for RAG evaluation');
    }

    if (chars.hasReference) {
        parts.push('Has reference answer for comparison');
    }

    if (chars.domains.length > 0) {
        parts.push(`Domains: ${chars.domains.join(', ')}`);
    }

    parts.push(`Length: ${chars.lengthCategory}`);

    return parts.join('; ');
}
