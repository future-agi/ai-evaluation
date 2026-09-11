/**
 * Base class for LLM providers with shared functionality.
 * @module local/llm/base
 */

import {
    BaseLLM,
    BaseLLMConfig,
    ChatMessage,
    GenerateOptions,
    ChatOptions,
    JudgeResult,
    JudgeInput
} from './types';

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG: Required<BaseLLMConfig> = {
    temperature: 0.0,
    maxTokens: 1024,
    timeout: 120
};

/**
 * Abstract base class for LLM providers.
 * Provides common functionality like response parsing.
 */
export abstract class AbstractLLM implements BaseLLM {
    abstract readonly provider: string;
    abstract readonly model: string;

    protected config: Required<BaseLLMConfig>;

    constructor(config: BaseLLMConfig = {}) {
        this.config = {
            ...DEFAULT_CONFIG,
            ...config
        };
    }

    abstract isAvailable(): Promise<boolean>;
    abstract generate(prompt: string, options?: GenerateOptions): Promise<string>;
    abstract chat(messages: ChatMessage[], options?: ChatOptions): Promise<string>;

    /**
     * Use LLM as a judge to evaluate a response
     */
    async judge(
        query: string,
        response: string,
        criteria: string,
        context?: string
    ): Promise<JudgeResult> {
        const systemPrompt = `You are an expert evaluator. Your task is to evaluate the quality of AI-generated responses based on specific criteria.

You MUST respond with a JSON object containing:
- "score": A number between 0 and 1 (0 = completely fails, 1 = perfect)
- "passed": A boolean (true if score >= 0.5)
- "reason": A brief explanation of your evaluation

IMPORTANT: Your response must be valid JSON only, with no additional text.`;

        let userPrompt = `## Evaluation Criteria
${criteria}

## Query
${query}

## Response to Evaluate
${response}`;

        if (context) {
            userPrompt += `\n\n## Context\n${context}`;
        }

        userPrompt += `\n\n## Your Evaluation (respond with JSON only)`;

        const rawResponse = await this.chat([
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
        ]);

        return this.parseJudgeResponse(rawResponse);
    }

    /**
     * Batch evaluate multiple queries
     */
    async batchJudge(evaluations: JudgeInput[]): Promise<JudgeResult[]> {
        const results: JudgeResult[] = [];

        for (const evalItem of evaluations) {
            try {
                const result = await this.judge(
                    evalItem.query,
                    evalItem.response,
                    evalItem.criteria,
                    evalItem.context
                );
                results.push(result);
            } catch (error) {
                results.push({
                    score: 0,
                    passed: false,
                    reason: `Evaluation failed: ${(error as Error).message}`,
                    rawResponse: undefined
                });
            }
        }

        return results;
    }

    /**
     * Parse judge response into structured result
     */
    protected parseJudgeResponse(response: string): JudgeResult {
        const trimmed = response.trim();

        // Try direct JSON parse
        try {
            const parsed = JSON.parse(trimmed);
            return this.validateJudgeResult(parsed, response);
        } catch {
            // Continue with other parsing strategies
        }

        // Try extracting JSON from code block
        const codeBlockMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (codeBlockMatch) {
            try {
                const parsed = JSON.parse(codeBlockMatch[1]);
                return this.validateJudgeResult(parsed, response);
            } catch {
                // Continue with other strategies
            }
        }

        // Try extracting embedded JSON
        const jsonMatch = trimmed.match(/\{[\s\S]*"score"[\s\S]*\}/);
        if (jsonMatch) {
            try {
                const parsed = JSON.parse(jsonMatch[0]);
                return this.validateJudgeResult(parsed, response);
            } catch {
                // Continue with regex extraction
            }
        }

        // Fallback: extract score using regex
        const scoreMatch = trimmed.match(/(?:score|rating)[:\s]*(\d+\.?\d*)/i);
        if (scoreMatch) {
            const score = this.normalizeScore(parseFloat(scoreMatch[1]));
            return {
                score,
                passed: score >= 0.5,
                reason: 'Score extracted from unstructured response',
                rawResponse: response
            };
        }

        // Default fallback
        return {
            score: 0.5,
            passed: true,
            reason: 'Could not parse judge response, using default score',
            rawResponse: response
        };
    }

    /**
     * Validate and normalize judge result
     */
    protected validateJudgeResult(parsed: any, rawResponse: string): JudgeResult {
        let score = typeof parsed.score === 'number' ? parsed.score : 0.5;
        score = this.normalizeScore(score);

        return {
            score,
            passed: parsed.passed ?? score >= 0.5,
            reason: parsed.reason || 'No reason provided',
            rawResponse
        };
    }

    /**
     * Normalize score to 0-1 range
     */
    protected normalizeScore(score: number): number {
        // Handle scores on different scales
        if (score > 10) {
            score = score / 100; // 0-100 scale
        } else if (score > 1) {
            score = score / 10; // 0-10 scale
        }
        // Clamp to [0, 1]
        return Math.max(0, Math.min(1, score));
    }
}
