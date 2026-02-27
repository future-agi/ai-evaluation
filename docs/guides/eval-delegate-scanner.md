# Eval Delegate Scanner

> **Language Support:** Python ✅ | TypeScript 📋 | Go 📋 | Java 📋

The `EvalDelegateScanner` bridges the guardrails scanner system with the evaluation framework, allowing you to use battle-tested LLM-based evaluations as fast safety scanners.

## Overview

Instead of reimplementing detection logic, the EvalDelegateScanner delegates to existing evaluation templates:

| Category | Eval Template | Description |
|----------|---------------|-------------|
| `pii` | Template 14 | Detects personally identifiable information |
| `toxicity` | Template 15 | Detects toxic or harmful content |
| `prompt_injection` | Template 18 | Detects prompt injection attempts |
| `bias` | Template 69 | Detects various forms of bias |
| `racial_bias` | Template 77 | Detects racial bias |
| `gender_bias` | Template 78 | Detects gender bias |
| `age_bias` | Template 79 | Detects age bias |
| `content_safety` | Template 93 | Detects content safety violations |
| `nsfw` | Template 20 | Detects not-safe-for-work content |
| `sexist` | Template 17 | Detects sexist content |

## Quick Start

### Basic Usage

```python
from fi.evals.guardrails.scanners import EvalDelegateScanner, EvalCategory

# Create a scanner for toxicity detection
scanner = EvalDelegateScanner(categories=[EvalCategory.TOXICITY])
result = scanner.scan("Your content here")

if not result.passed:
    print(f"Detected: {result.reason}")
```

### Factory Methods

Convenience factory methods create pre-configured scanners:

```python
from fi.evals.guardrails.scanners import EvalDelegateScanner

# Single-purpose scanners
toxicity_scanner = EvalDelegateScanner.for_toxicity()
pii_scanner = EvalDelegateScanner.for_pii()
prompt_injection_scanner = EvalDelegateScanner.for_prompt_injection()

# Multi-category scanners
bias_scanner = EvalDelegateScanner.for_bias(include_specific=True)
safety_scanner = EvalDelegateScanner.for_safety()
moderation_scanner = EvalDelegateScanner.for_content_moderation()
```

### Convenience Aliases

For even simpler usage, convenience aliases are available:

```python
from fi.evals.guardrails.scanners import (
    PIIScanner,
    ToxicityScanner,
    PromptInjectionScanner,
    BiasScanner,
    SafetyScanner,
    ContentModerationScanner,
)

# Create scanners with default settings
pii = PIIScanner()
toxicity = ToxicityScanner()
safety = SafetyScanner()
```

## Configuration

### Custom Thresholds

Set custom detection thresholds per category:

```python
scanner = EvalDelegateScanner(
    categories=[EvalCategory.TOXICITY, EvalCategory.PII],
    thresholds={
        EvalCategory.TOXICITY: 0.7,  # Higher threshold = less sensitive
        EvalCategory.PII: 0.3,       # Lower threshold = more sensitive
    }
)
```

### Aggregation Modes

Control how multiple category results are combined:

```python
# "any" mode (default): Fail if ANY category fails
scanner = EvalDelegateScanner(
    categories=[EvalCategory.TOXICITY, EvalCategory.PII],
    aggregation="any"
)

# "all" mode: Fail only if ALL categories fail
scanner = EvalDelegateScanner(
    categories=[EvalCategory.TOXICITY, EvalCategory.PII],
    aggregation="all"
)
```

### Execution Options

```python
scanner = EvalDelegateScanner(
    categories=[EvalCategory.TOXICITY],
    prefer_local=True,       # Use local evaluator if available (faster)
    api_key="your-api-key",  # For cloud evaluation
    timeout=30,              # Timeout in seconds
    parallel=True,           # Run categories in parallel
)
```

## Integration with Pipeline

Use EvalDelegateScanner alongside other scanners:

```python
from fi.evals.guardrails.scanners import (
    ScannerPipeline,
    JailbreakScanner,
    CodeInjectionScanner,
    EvalDelegateScanner,
    EvalCategory,
)

# Create a comprehensive pipeline
pipeline = ScannerPipeline([
    # Fast pattern-based scanners first
    JailbreakScanner(),
    CodeInjectionScanner(),

    # LLM-based evaluation scanners
    EvalDelegateScanner(
        categories=[EvalCategory.TOXICITY, EvalCategory.PII],
    ),
])

# Scan content
result = pipeline.scan("User input here")
if not result.passed:
    print(f"Blocked by: {result.blocked_by}")
```

## Real-World Examples

### 1. Content Moderation API

Build a content moderation service for user-generated content:

```python
from fi.evals.guardrails.scanners import ContentModerationScanner
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ModerationAction(Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"

@dataclass
class ModerationResult:
    action: ModerationAction
    score: float
    flagged_categories: List[str]
    details: dict

class ContentModerator:
    """Production-ready content moderation service."""

    def __init__(self, flag_threshold: float = 0.3, block_threshold: float = 0.7):
        self.scanner = ContentModerationScanner()
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold

    def moderate(self, content: str, user_id: Optional[str] = None) -> ModerationResult:
        result = self.scanner.scan(content)

        # Determine action based on score
        if result.score >= self.block_threshold:
            action = ModerationAction.BLOCK
        elif result.score >= self.flag_threshold:
            action = ModerationAction.FLAG
        else:
            action = ModerationAction.ALLOW

        return ModerationResult(
            action=action,
            score=result.score,
            flagged_categories=result.metadata.get("categories_failed", []),
            details=result.metadata.get("category_results", {}),
        )

# Usage
moderator = ContentModerator()
result = moderator.moderate("User comment here")

if result.action == ModerationAction.BLOCK:
    print("Content blocked - too harmful")
elif result.action == ModerationAction.FLAG:
    print("Content flagged for review")
else:
    print("Content allowed")
```

### 2. Chatbot Safety Layer

Protect your chatbot from harmful inputs and outputs:

```python
from fi.evals.guardrails.scanners import (
    ScannerPipeline,
    JailbreakScanner,
    EvalDelegateScanner,
    EvalCategory,
)
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SafetyCheckResult:
    safe: bool
    reason: Optional[str]
    blocked_by: Optional[str]

class ChatbotSafetyLayer:
    """Safety layer for LLM chatbot applications."""

    def __init__(self):
        # Input scanner: check for jailbreaks and prompt injection
        self.input_scanner = ScannerPipeline([
            JailbreakScanner(),
            EvalDelegateScanner(
                categories=[EvalCategory.PROMPT_INJECTION],
                threshold=0.5,
            ),
        ])

        # Output scanner: check for harmful content
        self.output_scanner = EvalDelegateScanner(
            categories=[
                EvalCategory.TOXICITY,
                EvalCategory.PII,
                EvalCategory.CONTENT_SAFETY,
            ],
            threshold=0.6,
        )

    def check_input(self, user_message: str) -> SafetyCheckResult:
        """Check user input before sending to LLM."""
        result = self.input_scanner.scan(user_message)

        return SafetyCheckResult(
            safe=result.passed,
            reason=result.reason if not result.passed else None,
            blocked_by=result.blocked_by[0] if hasattr(result, 'blocked_by') and result.blocked_by else None,
        )

    def check_output(self, llm_response: str) -> SafetyCheckResult:
        """Check LLM output before sending to user."""
        result = self.output_scanner.scan(llm_response)

        return SafetyCheckResult(
            safe=result.passed,
            reason=result.reason if not result.passed else None,
            blocked_by=None,
        )

    def safe_chat(self, user_message: str, llm_func) -> Tuple[bool, str]:
        """Complete safe chat flow."""
        # Check input
        input_check = self.check_input(user_message)
        if not input_check.safe:
            return False, f"Input blocked: {input_check.reason}"

        # Get LLM response
        llm_response = llm_func(user_message)

        # Check output
        output_check = self.check_output(llm_response)
        if not output_check.safe:
            return False, "I apologize, but I cannot provide that response."

        return True, llm_response

# Usage
safety = ChatbotSafetyLayer()

def mock_llm(msg):
    return "Here's a helpful response..."

success, response = safety.safe_chat("How do I hack a computer?", mock_llm)
print(f"Success: {success}, Response: {response}")
```

### 3. PII Detection for Data Pipeline

Scan data pipelines for personally identifiable information:

```python
from fi.evals.guardrails.scanners import PIIScanner
from typing import List, Dict, Any
import json

class DataPipelinePIIScanner:
    """Scan data records for PII before storage or processing."""

    def __init__(self, sensitivity: str = "medium"):
        thresholds = {"low": 0.7, "medium": 0.5, "high": 0.3}
        self.scanner = PIIScanner(threshold=thresholds.get(sensitivity, 0.5))

    def scan_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Scan a single record for PII."""
        # Convert record to text for scanning
        text = json.dumps(record, default=str)
        result = self.scanner.scan(text)

        return {
            "has_pii": not result.passed,
            "confidence": result.score,
            "record_id": record.get("id"),
        }

    def scan_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scan a batch of records."""
        results = []
        pii_count = 0

        for record in records:
            scan_result = self.scan_record(record)
            results.append(scan_result)
            if scan_result["has_pii"]:
                pii_count += 1

        return {
            "total_records": len(records),
            "pii_detected": pii_count,
            "clean_records": len(records) - pii_count,
            "details": results,
        }

# Usage
scanner = DataPipelinePIIScanner(sensitivity="high")

records = [
    {"id": 1, "text": "Contact john@email.com for details"},
    {"id": 2, "text": "The weather is nice today"},
    {"id": 3, "text": "Call me at 555-123-4567"},
]

report = scanner.scan_batch(records)
print(f"Found PII in {report['pii_detected']} of {report['total_records']} records")
```

### 4. HR/Hiring Bias Detection

Detect bias in job postings or hiring-related content:

```python
from fi.evals.guardrails.scanners import BiasScanner, EvalCategory
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BiasReport:
    has_bias: bool
    bias_types: List[str]
    severity: str  # low, medium, high
    recommendations: List[str]

class HRBiasDetector:
    """Detect bias in HR/hiring content."""

    def __init__(self):
        self.scanner = BiasScanner(
            include_specific=True,
            threshold=0.4,
        )

    def analyze_job_posting(self, job_description: str) -> BiasReport:
        """Analyze a job posting for bias."""
        result = self.scanner.scan(job_description)

        bias_types = []
        recommendations = []

        category_results = result.metadata.get("category_results", {})

        if category_results.get("gender_bias", {}).get("passed") is False:
            bias_types.append("gender")
            recommendations.append("Review language for gender-neutral alternatives")

        if category_results.get("age_bias", {}).get("passed") is False:
            bias_types.append("age")
            recommendations.append("Remove age-related requirements or preferences")

        if category_results.get("racial_bias", {}).get("passed") is False:
            bias_types.append("racial")
            recommendations.append("Review for cultural bias or exclusionary language")

        # Determine severity
        if result.score >= 0.7:
            severity = "high"
        elif result.score >= 0.4:
            severity = "medium"
        else:
            severity = "low"

        return BiasReport(
            has_bias=not result.passed,
            bias_types=bias_types,
            severity=severity,
            recommendations=recommendations,
        )

# Usage
detector = HRBiasDetector()

job_posting = """
We're looking for a young, energetic developer to join our team.
Must be a native English speaker. Recent graduates preferred.
"""

report = detector.analyze_job_posting(job_posting)
if report.has_bias:
    print(f"Bias detected ({report.severity}): {report.bias_types}")
    for rec in report.recommendations:
        print(f"  - {rec}")
```

### 5. Customer Support Safety

Screen customer support conversations:

```python
from fi.evals.guardrails.scanners import (
    EvalDelegateScanner,
    EvalCategory,
    ScannerPipeline,
    SecretsScanner,
)
from enum import Enum
from typing import Optional

class EscalationLevel(Enum):
    NONE = "none"
    SUPERVISOR = "supervisor"
    SECURITY = "security"

class CustomerSupportSafety:
    """Safety system for customer support interactions."""

    def __init__(self):
        self.scanner = ScannerPipeline([
            # Check for leaked credentials
            SecretsScanner(),
            # Check for harmful content
            EvalDelegateScanner(
                categories=[
                    EvalCategory.TOXICITY,
                    EvalCategory.PII,
                ],
            ),
        ])

    def process_message(self, message: str, is_customer: bool) -> dict:
        """Process a support message."""
        result = self.scanner.scan(message)

        escalation = EscalationLevel.NONE
        actions = []

        if not result.passed:
            failed = result.metadata.get("categories_failed", [])

            if "toxicity" in str(failed).lower():
                escalation = EscalationLevel.SUPERVISOR
                actions.append("Route to supervisor for de-escalation")

            if "pii" in str(failed).lower() and is_customer:
                actions.append("Remind customer not to share sensitive info")

            if "secrets" in str(result.blocked_by or []):
                escalation = EscalationLevel.SECURITY
                actions.append("Flag for security review - credentials detected")

        return {
            "safe": result.passed,
            "escalation": escalation.value,
            "actions": actions,
            "score": result.score,
        }

# Usage
safety = CustomerSupportSafety()

customer_msg = "Here's my password: MySecret123! Can you check my account?"
result = safety.process_message(customer_msg, is_customer=True)

if result["escalation"] != "none":
    print(f"Escalation needed: {result['escalation']}")
for action in result["actions"]:
    print(f"  Action: {action}")
```

### 6. RAG Input/Output Guardrails

Complete guardrails for RAG applications:

```python
from fi.evals.guardrails.scanners import (
    ScannerPipeline,
    JailbreakScanner,
    EvalDelegateScanner,
    EvalCategory,
    CodeInjectionScanner,
)
from typing import List, Optional, Callable
from dataclasses import dataclass

@dataclass
class RAGResponse:
    success: bool
    answer: Optional[str]
    sources: List[str]
    blocked_reason: Optional[str]

class RAGGuardrails:
    """Complete guardrails for RAG systems."""

    def __init__(self):
        # Input guardrails
        self.input_guards = ScannerPipeline([
            JailbreakScanner(),
            CodeInjectionScanner(),
            EvalDelegateScanner(
                categories=[EvalCategory.PROMPT_INJECTION],
            ),
        ])

        # Output guardrails
        self.output_guards = EvalDelegateScanner(
            categories=[
                EvalCategory.TOXICITY,
                EvalCategory.PII,
                EvalCategory.CONTENT_SAFETY,
            ],
            aggregation="any",
        )

    def query(
        self,
        question: str,
        retriever: Callable[[str], List[str]],
        generator: Callable[[str, List[str]], str],
    ) -> RAGResponse:
        """Execute a guarded RAG query."""

        # 1. Check input
        input_result = self.input_guards.scan(question)
        if not input_result.passed:
            return RAGResponse(
                success=False,
                answer=None,
                sources=[],
                blocked_reason=f"Input blocked: {input_result.reason}",
            )

        # 2. Retrieve documents
        documents = retriever(question)

        # 3. Generate answer
        answer = generator(question, documents)

        # 4. Check output
        output_result = self.output_guards.scan(answer)
        if not output_result.passed:
            return RAGResponse(
                success=False,
                answer=None,
                sources=documents,
                blocked_reason=f"Output blocked: {output_result.reason}",
            )

        return RAGResponse(
            success=True,
            answer=answer,
            sources=documents,
            blocked_reason=None,
        )

# Usage
rag = RAGGuardrails()

def mock_retriever(q):
    return ["Document 1...", "Document 2..."]

def mock_generator(q, docs):
    return "Based on the documents, here's the answer..."

response = rag.query(
    "What is machine learning?",
    retriever=mock_retriever,
    generator=mock_generator,
)

if response.success:
    print(f"Answer: {response.answer}")
else:
    print(f"Blocked: {response.blocked_reason}")
```

### 7. Multi-Tenant SaaS Safety

Different safety levels for different tenant tiers:

```python
from fi.evals.guardrails.scanners import EvalDelegateScanner, EvalCategory
from typing import Dict
from enum import Enum

class TenantTier(Enum):
    FREE = "free"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"

class MultiTenantSafety:
    """Configurable safety for multi-tenant SaaS."""

    TIER_CONFIGS = {
        TenantTier.FREE: {
            "categories": [EvalCategory.TOXICITY, EvalCategory.CONTENT_SAFETY],
            "threshold": 0.3,  # Strict
        },
        TenantTier.STANDARD: {
            "categories": [
                EvalCategory.TOXICITY,
                EvalCategory.PII,
                EvalCategory.CONTENT_SAFETY,
            ],
            "threshold": 0.5,  # Moderate
        },
        TenantTier.ENTERPRISE: {
            "categories": [
                EvalCategory.TOXICITY,
                EvalCategory.PII,
                EvalCategory.BIAS,
                EvalCategory.CONTENT_SAFETY,
                EvalCategory.PROMPT_INJECTION,
            ],
            "threshold": 0.6,  # Configurable
        },
    }

    def __init__(self):
        self.scanners: Dict[TenantTier, EvalDelegateScanner] = {}
        for tier, config in self.TIER_CONFIGS.items():
            self.scanners[tier] = EvalDelegateScanner(
                categories=config["categories"],
                thresholds={cat: config["threshold"] for cat in config["categories"]},
            )

    def scan_for_tenant(self, content: str, tier: TenantTier) -> dict:
        """Scan content based on tenant tier."""
        scanner = self.scanners[tier]
        result = scanner.scan(content)

        return {
            "passed": result.passed,
            "tier": tier.value,
            "categories_checked": len(scanner.categories),
            "issues": result.metadata.get("categories_failed", []),
        }

# Usage
safety = MultiTenantSafety()

content = "Some user content..."
result = safety.scan_for_tenant(content, TenantTier.ENTERPRISE)
print(f"Tier {result['tier']}: {'Passed' if result['passed'] else 'Blocked'}")
```

## Understanding Results

The `ScanResult` includes detailed metadata:

```python
result = scanner.scan("Test content")

# Basic result
print(result.passed)           # True/False
print(result.score)            # Aggregated score (0.0-1.0)
print(result.reason)           # Human-readable reason
print(result.latency_ms)       # Processing time

# Detailed metadata
print(result.metadata["categories_checked"])   # List of checked categories
print(result.metadata["categories_failed"])    # List of failed categories
print(result.metadata["category_results"])     # Per-category details

# Example category_results:
# {
#     "toxicity": {"passed": True, "score": 0.1, "source": "local"},
#     "pii": {"passed": False, "score": 0.8, "source": "cloud"},
# }
```

## Notes

- **Local vs Cloud**: The scanner prefers local evaluation (faster) but falls back to cloud API when needed
- **Threshold Interpretation**: Most categories use "inverted" scoring where high score = detected (bad)
- **Parallel Execution**: Multiple categories run in parallel by default for better performance
- **Graceful Degradation**: If no evaluator is available, the scanner passes content by default

## Dependencies

- Requires `fi.evals.local` for local evaluation
- Requires API key for cloud evaluation when local is unavailable
