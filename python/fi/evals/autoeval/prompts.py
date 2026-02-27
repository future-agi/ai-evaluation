"""LLM prompt templates for AutoEval."""

ANALYSIS_SYSTEM_PROMPT = """You are an AI evaluation expert. Your task is to analyze application descriptions and recommend appropriate evaluations and safety scanners.

You must respond in valid JSON format with the following structure:
{
    "category": "<category>",
    "risk_level": "<risk_level>",
    "domain_sensitivity": "<sensitivity>",
    "detected_features": ["<feature1>", "<feature2>"],
    "requirements": [
        {
            "category": "<requirement_category>",
            "importance": "<importance>",
            "reason": "<explanation>",
            "suggested_evals": ["<eval1>", "<eval2>"],
            "suggested_scanners": ["<scanner1>", "<scanner2>"]
        }
    ],
    "explanation": "<brief explanation of the analysis>"
}

## Valid Values

### Categories
- customer_support: Customer service chatbots, help desks
- rag_system: Retrieval-augmented generation, document Q&A
- code_assistant: Code generation, debugging, review
- content_moderation: Content filtering, safety systems
- agent_workflow: Autonomous agents with tool use
- chatbot: General conversational AI
- summarization: Text summarization
- translation: Language translation
- creative_writing: Content generation, copywriting
- data_extraction: Information extraction, parsing
- search: Search systems
- question_answering: Q&A systems
- unknown: Cannot determine

### Risk Levels
- low: Internal tools, development, testing (threshold: 0.6)
- medium: General public-facing applications (threshold: 0.7)
- high: Healthcare, finance, legal domains (threshold: 0.8)
- critical: Safety-critical systems (threshold: 0.9)

### Domain Sensitivity
- general: No special sensitivity
- pii_sensitive: Handles personal information
- financial: Banking, payments, investments
- healthcare: Medical, patient data, HIPAA
- legal: Legal documents, contracts
- children: Content for minors, COPPA
- government: Government/public sector

### Importance
- required: Must have for the application to be safe/functional
- recommended: Strongly advised but not mandatory
- optional: Nice to have

## Available Evaluations

### Semantic Evaluations
- CoherenceEval: Checks response coherence and logical flow

### Agentic Evaluations
- ActionSafetyEval: Detects unsafe agent actions
- ReasoningQualityEval: Evaluates reasoning quality

### Multimodal Evaluations
- ImageTextConsistencyEval: Checks image-text alignment
- CaptionQualityEval: Evaluates caption quality
- VisualQAEval: Evaluates visual Q&A responses
- ImageSafetyEval: Checks image safety
- CrossModalConsistencyEval: Cross-modal consistency

## Available Scanners

### Security Scanners
- jailbreak: Detects jailbreak and prompt manipulation attempts
- code_injection: Detects SQL, shell, path traversal injection
- secrets: Detects leaked API keys, passwords, credentials
- prompt_injection: Detects prompt injection attacks

### Safety Scanners
- toxicity: Detects toxic, harmful content
- bias: Detects biased content
- pii: Detects personally identifiable information

### Content Scanners
- malicious_url: Detects phishing and suspicious URLs
- invisible_chars: Detects Unicode manipulation
- language: Validates language requirements
- topic_restriction: Enforces topic boundaries

## Guidelines

1. Match the app type to the most specific category
2. Consider the domain - healthcare, finance, legal are high-risk
3. Recommend evaluations based on the app's primary function
4. Recommend scanners based on safety requirements
5. Higher risk = higher thresholds and more scanners
6. Always include basic safety scanners for public-facing apps
7. For RAG systems, always include faithfulness evaluations
8. For agents, always include tool use and safety evaluations"""

ANALYSIS_USER_PROMPT = """Analyze this application description and recommend evaluations and scanners:

{description}

Consider:
1. What type of application is this?
2. What are the risk factors?
3. What domain sensitivity applies?
4. What evaluations would ensure quality?
5. What scanners would ensure safety?

Respond with valid JSON only, no additional text."""


CLARIFICATION_QUESTIONS_PROMPT = """Based on the application description, generate clarifying questions to better configure the evaluation pipeline.

Application Description:
{description}

Current Analysis:
- Category: {category}
- Risk Level: {risk_level}
- Domain Sensitivity: {domain_sensitivity}
- Confidence: {confidence}

Generate 1-3 clarifying questions that would help improve the evaluation configuration. Focus on:
1. Deployment environment (internal vs production)
2. Data sensitivity (PII, financial, health data)
3. User base (general public, employees, children)
4. Special requirements (compliance, real-time)

Respond in JSON format:
{
    "questions": [
        {
            "question": "<question text>",
            "options": ["<option1>", "<option2>", "<option3>"],
            "impact": "<what this answer affects>"
        }
    ]
}"""
