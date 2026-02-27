"""Pre-built templates for AutoEval.

Provides ready-to-use evaluation configurations for common application types.
"""

from typing import Dict, Optional
from .config import AutoEvalConfig, EvalConfig, ScannerConfig


# Template definitions
TEMPLATES: Dict[str, AutoEvalConfig] = {
    "customer_support": AutoEvalConfig(
        name="customer_support",
        description="Evaluation pipeline for customer support chatbots and help desks",
        app_category="customer_support",
        risk_level="medium",
        domain_sensitivity="general",
        evaluations=[
            EvalConfig(
                name="answer_relevancy",
                threshold=0.7,
                weight=1.0,
            ),
        ],
        scanners=[
            ScannerConfig(
                name="JailbreakScanner",
                threshold=0.7,
                action="block",
            ),
            ScannerConfig(
                name="ToxicityScanner",
                threshold=0.7,
                action="block",
            ),
            ScannerConfig(
                name="PIIScanner",
                threshold=0.7,
                action="flag",
            ),
        ],
        global_pass_rate=0.8,
    ),
    "rag_system": AutoEvalConfig(
        name="rag_system",
        description="Evaluation pipeline for RAG-based document Q&A systems",
        app_category="rag_system",
        risk_level="medium",
        domain_sensitivity="general",
        evaluations=[
            EvalConfig(
                name="faithfulness",
                threshold=0.7,
                weight=1.5,
            ),
            EvalConfig(
                name="groundedness",
                threshold=0.7,
                weight=1.5,
            ),
            EvalConfig(
                name="answer_relevancy",
                threshold=0.7,
                weight=1.0,
            ),
        ],
        scanners=[
            ScannerConfig(
                name="JailbreakScanner",
                threshold=0.7,
                action="block",
            ),
        ],
        global_pass_rate=0.8,
    ),
    "code_assistant": AutoEvalConfig(
        name="code_assistant",
        description="Evaluation pipeline for code generation and development assistants",
        app_category="code_assistant",
        risk_level="medium",
        domain_sensitivity="general",
        evaluations=[
            EvalConfig(
                name="answer_relevancy",
                threshold=0.7,
                weight=1.0,
            ),
        ],
        scanners=[
            ScannerConfig(
                name="CodeInjectionScanner",
                threshold=0.8,
                action="block",
            ),
            ScannerConfig(
                name="SecretsScanner",
                threshold=0.9,
                action="block",
            ),
            ScannerConfig(
                name="JailbreakScanner",
                threshold=0.7,
                action="block",
            ),
        ],
        global_pass_rate=0.8,
    ),
    "content_moderation": AutoEvalConfig(
        name="content_moderation",
        description="Evaluation pipeline for content moderation and safety systems",
        app_category="content_moderation",
        risk_level="high",
        domain_sensitivity="general",
        evaluations=[],
        scanners=[
            ScannerConfig(
                name="ToxicityScanner",
                threshold=0.8,
                action="block",
            ),
            ScannerConfig(
                name="BiasScanner",
                threshold=0.8,
                action="flag",
            ),
            ScannerConfig(
                name="InvisibleCharScanner",
                threshold=0.9,
                action="block",
            ),
            ScannerConfig(
                name="MaliciousURLScanner",
                threshold=0.9,
                action="block",
            ),
        ],
        global_pass_rate=0.9,
    ),
    "agent_workflow": AutoEvalConfig(
        name="agent_workflow",
        description="Evaluation pipeline for autonomous agents with tool use",
        app_category="agent_workflow",
        risk_level="high",
        domain_sensitivity="general",
        evaluations=[
            EvalConfig(
                name="action_safety",
                threshold=0.85,
                weight=2.0,
            ),
            EvalConfig(
                name="reasoning_quality",
                threshold=0.7,
                weight=1.0,
            ),
        ],
        scanners=[
            ScannerConfig(
                name="JailbreakScanner",
                threshold=0.8,
                action="block",
            ),
            ScannerConfig(
                name="CodeInjectionScanner",
                threshold=0.85,
                action="block",
            ),
        ],
        global_pass_rate=0.85,
    ),
}

# Healthcare-specific template (high risk)
TEMPLATES["healthcare"] = AutoEvalConfig(
    name="healthcare",
    description="Evaluation pipeline for healthcare applications (HIPAA-compliant)",
    app_category="question_answering",
    risk_level="high",
    domain_sensitivity="healthcare",
    evaluations=[
        EvalConfig(
            name="faithfulness",
            threshold=0.85,
            weight=2.0,
        ),
        EvalConfig(
            name="answer_relevancy",
            threshold=0.8,
            weight=1.0,
        ),
    ],
    scanners=[
        ScannerConfig(
            name="PIIScanner",
            threshold=0.9,
            action="redact",
        ),
        ScannerConfig(
            name="SecretsScanner",
            threshold=0.95,
            action="block",
        ),
        ScannerConfig(
            name="ToxicityScanner",
            threshold=0.9,
            action="block",
        ),
        ScannerConfig(
            name="JailbreakScanner",
            threshold=0.85,
            action="block",
        ),
    ],
    global_pass_rate=0.9,
)

# Financial services template (high risk)
TEMPLATES["financial"] = AutoEvalConfig(
    name="financial",
    description="Evaluation pipeline for financial services applications",
    app_category="customer_support",
    risk_level="high",
    domain_sensitivity="financial",
    evaluations=[
        EvalConfig(
            name="factual_consistency",
            threshold=0.85,
            weight=2.0,
        ),
        EvalConfig(
            name="answer_relevancy",
            threshold=0.8,
            weight=1.0,
        ),
    ],
    scanners=[
        ScannerConfig(
            name="PIIScanner",
            threshold=0.9,
            action="redact",
        ),
        ScannerConfig(
            name="SecretsScanner",
            threshold=0.95,
            action="block",
        ),
        ScannerConfig(
            name="JailbreakScanner",
            threshold=0.85,
            action="block",
        ),
    ],
    global_pass_rate=0.9,
)


def get_template(name: str) -> Optional[AutoEvalConfig]:
    """
    Get a pre-built template by name.

    Args:
        name: Template name (e.g., "customer_support", "rag_system")

    Returns:
        AutoEvalConfig copy or None if not found

    Example:
        config = get_template("rag_system")
        if config:
            pipeline = AutoEvalPipeline(config)
    """
    template = TEMPLATES.get(name)
    if template:
        return template.copy()
    return None


def list_templates() -> Dict[str, str]:
    """
    List all available templates with descriptions.

    Returns:
        Dictionary of template names to descriptions

    Example:
        for name, desc in list_templates().items():
            print(f"{name}: {desc}")
    """
    return {name: config.description for name, config in TEMPLATES.items()}


def get_template_names() -> list:
    """
    Get list of all template names.

    Returns:
        List of template names
    """
    return list(TEMPLATES.keys())
