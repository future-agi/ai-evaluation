# AutoEval - Automatic Evaluation Pipeline Builder

> **Language Support:** Python ✅ | TypeScript 🚧 | Go 📋 | Java 📋

AutoEval automatically designs evaluation pipelines from natural language application descriptions. It analyzes the app type, risk level, and domain sensitivity to recommend appropriate evaluations and scanners with properly calibrated thresholds.

## Quick Start

<!-- tabs:start -->
### **Python**

```python
from fi.evals.autoeval import AutoEvalPipeline

# Create from natural language description
pipeline = AutoEvalPipeline.from_description(
    "A customer support chatbot for a healthcare company that retrieves "
    "patient appointment information and answers questions about billing."
)

# Run evaluation
result = pipeline.evaluate({
    "query": "When is my next appointment?",
    "response": "Your next appointment is scheduled for Monday, January 15th at 2:00 PM with Dr. Smith.",
    "context": "Patient record shows appointment on 2024-01-15 14:00 with Dr. Smith, Cardiology.",
})

print(f"Passed: {result.passed}")
print(result.explain())
```

### **TypeScript**

```typescript
// Coming soon
import { AutoEvalPipeline } from '@anthropic/ai-evaluation';

const pipeline = await AutoEvalPipeline.fromDescription(
    "A customer support chatbot for a healthcare company..."
);

const result = await pipeline.evaluate({
    query: "When is my next appointment?",
    response: "Your next appointment is scheduled for Monday...",
    context: "Patient record shows...",
});
```
<!-- tabs:end -->

## Real-World Examples

### Example 1: Healthcare RAG Chatbot

A common use case is building a patient-facing healthcare chatbot that retrieves information from medical records.

```python
from fi.evals.autoeval import AutoEvalPipeline, EvalConfig, ScannerConfig

# Create pipeline from description
pipeline = AutoEvalPipeline.from_description(
    "A HIPAA-compliant patient portal chatbot for a hospital. "
    "Patients can ask about their appointments, test results, medications, "
    "and billing information. The system retrieves from electronic health records."
)

# The pipeline automatically detects:
# - Category: customer_support / rag_system
# - Risk Level: HIGH (healthcare domain)
# - Domain Sensitivity: healthcare
# - Adds: FactualConsistencyEval, PIIScanner with redact action, etc.

print(pipeline.explain())

# Customize if needed
pipeline.set_threshold("FactualConsistencyEval", 0.95)  # Stricter for medical info
pipeline.add(ScannerConfig("SecretsScanner", threshold=0.99, action="block"))

# Test with real-world scenario
result = pipeline.evaluate({
    "query": "What medications am I currently taking?",
    "response": "Based on your records, you are currently prescribed:\n"
                "1. Lisinopril 10mg - once daily for blood pressure\n"
                "2. Metformin 500mg - twice daily for diabetes",
    "context": "Patient medications: Lisinopril 10mg QD, Metformin 500mg BID. "
               "Last updated: 2024-01-10. Allergies: Penicillin.",
})

if not result.passed:
    print("Evaluation failed:")
    print(result.explain())
```

### Example 2: Financial Services Assistant

A banking chatbot that helps customers with account inquiries and transactions.

```python
from fi.evals.autoeval import AutoEvalPipeline

# Use the pre-built financial template
pipeline = AutoEvalPipeline.from_template("financial")

# Or create from description for more customization
pipeline = AutoEvalPipeline.from_description(
    "A mobile banking assistant that helps customers check balances, "
    "review recent transactions, transfer money between accounts, "
    "and answer questions about fees and interest rates."
)

# The pipeline detects financial domain and adds:
# - FactualConsistencyEval (high weight for accuracy)
# - PIIScanner, SecretsScanner (block/redact sensitive data)
# - JailbreakScanner (prevent prompt injection attacks)

# Test scenarios
test_cases = [
    {
        "query": "What's my checking account balance?",
        "response": "Your checking account ending in 4532 has a current balance of $2,847.63.",
        "context": "Account 4532: Balance $2847.63, Available $2847.63, Last transaction -$45.00 at Target",
    },
    {
        "query": "Transfer $500 to savings",
        "response": "I've initiated a transfer of $500 from your checking to savings account. "
                    "The transfer will be completed within 1 business day.",
        "context": "Transfer request: $500 from checking (4532) to savings (7891). Status: Pending.",
    },
]

for test in test_cases:
    result = pipeline.evaluate(test)
    print(f"Query: {test['query'][:50]}...")
    print(f"  Passed: {result.passed}, Latency: {result.total_latency_ms:.1f}ms")
```

### Example 3: Code Generation Assistant

An AI coding assistant for developers with security-focused evaluation.

```python
from fi.evals.autoeval import AutoEvalPipeline

pipeline = AutoEvalPipeline.from_description(
    "A VS Code extension that generates code snippets, explains code, "
    "reviews pull requests, and helps debug issues. Used by enterprise "
    "developers working on financial trading systems."
)

# Detects: code_assistant category, high risk (enterprise + financial)
# Adds: CodeInjectionScanner, SecretsScanner, JailbreakScanner

# Test code generation safety
result = pipeline.evaluate({
    "query": "Write a function to connect to our database",
    "response": '''def connect_db():
    import os
    conn_string = os.environ.get('DATABASE_URL')
    if not conn_string:
        raise ValueError("DATABASE_URL not set")
    return psycopg2.connect(conn_string)''',
})

# Check if the response accidentally includes secrets
result2 = pipeline.evaluate({
    "query": "Show me the database connection code",
    "response": '''# Database connection
DB_PASSWORD = "super_secret_123"  # This should be flagged!
conn = connect("host=db.example.com password=" + DB_PASSWORD)''',
})

print(f"Safe code: {result.passed}")
print(f"Code with secrets: {result2.passed}")  # Should fail
```

### Example 4: Content Moderation System

A content moderation pipeline for user-generated content.

```python
from fi.evals.autoeval import AutoEvalPipeline

# Use the content moderation template
pipeline = AutoEvalPipeline.from_template("content_moderation")

# Test various content types
test_contents = [
    "This product is great! Highly recommend.",  # Safe
    "Check out this link: bit.ly/free-iphone",   # Potentially malicious URL
    "I hate everyone who disagrees with me!!!",   # Potentially toxic
]

for content in test_contents:
    result = pipeline.evaluate(
        inputs={"response": content},
        scan_content=content,
    )
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {content[:50]}...")
    if not result.passed and result.scan_result:
        print(f"  Blocked by: {result.scan_result.blocked_by}")
```

### Example 5: Autonomous Agent Evaluation

Evaluating an AI agent that can browse the web and execute code.

```python
from fi.evals.autoeval import AutoEvalPipeline, EvalConfig

pipeline = AutoEvalPipeline.from_description(
    "An autonomous research agent that can search the web, read documents, "
    "write and execute Python code, and send emails. Used for market research."
)

# Detects agent_workflow category, adds:
# - ToolUseCorrectnessEval
# - TrajectoryEfficiencyEval
# - GoalCompletionEval
# - ActionSafetyEval (critical for autonomous agents)
# - CodeInjectionScanner, JailbreakScanner

# Increase safety thresholds for autonomous agent
pipeline.set_threshold("ActionSafetyEval", 0.95)

# Evaluate an agent trajectory
result = pipeline.evaluate({
    "goal": "Find the current stock price of Apple Inc.",
    "trajectory": [
        {"action": "web_search", "input": "AAPL stock price", "result": "Found Yahoo Finance"},
        {"action": "read_page", "input": "https://finance.yahoo.com/quote/AAPL", "result": "$185.23"},
    ],
    "response": "The current stock price of Apple Inc. (AAPL) is $185.23.",
    "tools_available": ["web_search", "read_page", "execute_code", "send_email"],
})

print(f"Agent evaluation: {result.passed}")
```

### Example 6: Children's Educational App

Extra-strict safety for content targeting minors.

```python
from fi.evals.autoeval import AutoEvalPipeline

pipeline = AutoEvalPipeline.from_description(
    "An educational chatbot for elementary school students (ages 6-12). "
    "Helps with homework, explains concepts in simple terms, and provides "
    "age-appropriate learning activities."
)

# Detects CHILDREN domain sensitivity
# Sets very strict thresholds (0.9+) for safety scanners
# Adds ToxicityScanner and BiasScanner with block action

# Verify strict settings
for scanner in pipeline.config.scanners:
    print(f"{scanner.name}: threshold={scanner.threshold}, action={scanner.action}")

# Test content safety
test_responses = [
    "Great job! 2 + 2 = 4. You're learning so fast!",  # Safe
    "That's a stupid question, figure it out yourself.",  # Toxic - should fail
]

for response in test_responses:
    result = pipeline.evaluate({"response": response})
    print(f"'{response[:40]}...' - {'SAFE' if result.passed else 'BLOCKED'}")
```

## Configuration Export & Version Control

Export your pipeline configuration for reproducibility and version control:

```python
from fi.evals.autoeval import AutoEvalPipeline

# Create and customize pipeline
pipeline = AutoEvalPipeline.from_description(
    "A customer support bot for an e-commerce platform."
)
pipeline.set_threshold("CoherenceEval", 0.8)
pipeline.add(ScannerConfig("PIIScanner", action="redact"))

# Export to YAML for version control
pipeline.export_yaml("eval_configs/customer_support_v1.yaml")

# Later, load the same configuration
pipeline_v1 = AutoEvalPipeline.from_yaml("eval_configs/customer_support_v1.yaml")
```

Example YAML output:

```yaml
version: "1.0.0"
name: autoeval_customer_support
description: A customer support bot for an e-commerce platform.
metadata:
  app_category: customer_support
  risk_level: medium
  domain_sensitivity: general
  generated_by: autoeval
execution:
  mode: non_blocking
  parallel_workers: 4
  timeout_seconds: 30
  fail_fast: false
thresholds:
  global_pass_rate: 0.8
evaluations:
  - name: CoherenceEval
    threshold: 0.8
    weight: 1.0
    enabled: true
  - name: SemanticSimilarityEval
    threshold: 0.7
    weight: 1.0
    enabled: true
scanners:
  - name: JailbreakScanner
    threshold: 0.7
    action: block
    enabled: true
  - name: ToxicityScanner
    threshold: 0.7
    action: block
    enabled: true
  - name: PIIScanner
    threshold: 0.7
    action: redact
    enabled: true
```

## Available Templates

| Template | Use Case | Key Evaluations | Key Scanners |
|----------|----------|-----------------|--------------|
| `customer_support` | Help desks, FAQs | Coherence, SemanticSimilarity | Jailbreak, Toxicity, PII |
| `rag_system` | Document Q&A | FactualConsistency, Entailment | Jailbreak |
| `code_assistant` | Code generation | Coherence | CodeInjection, Secrets, Jailbreak |
| `content_moderation` | UGC filtering | - | Toxicity, Bias, MaliciousURL |
| `agent_workflow` | Autonomous agents | ToolUse, GoalCompletion, ActionSafety | Jailbreak, CodeInjection |
| `healthcare` | HIPAA compliance | FactualConsistency | PII (redact), Secrets, Toxicity |
| `financial` | Banking, payments | FactualConsistency | PII (redact), Secrets, Jailbreak |

## API Reference

### AutoEvalPipeline

```python
class AutoEvalPipeline:
    @classmethod
    def from_description(cls, description: str, llm_provider=None) -> "AutoEvalPipeline":
        """Create from natural language description."""

    @classmethod
    def from_template(cls, template_name: str) -> "AutoEvalPipeline":
        """Create from pre-built template."""

    @classmethod
    def from_yaml(cls, path: str) -> "AutoEvalPipeline":
        """Load from YAML file."""

    def evaluate(self, inputs: Dict[str, Any], scan_content: str = None) -> AutoEvalResult:
        """Run the evaluation pipeline."""

    def add(self, item: Union[EvalConfig, ScannerConfig]) -> "AutoEvalPipeline":
        """Add an evaluation or scanner."""

    def remove(self, name: str) -> "AutoEvalPipeline":
        """Remove by name."""

    def set_threshold(self, name: str, threshold: float) -> "AutoEvalPipeline":
        """Set threshold for an eval or scanner."""

    def enable(self, name: str) -> "AutoEvalPipeline":
        """Enable an eval or scanner."""

    def disable(self, name: str) -> "AutoEvalPipeline":
        """Disable an eval or scanner."""

    def export_yaml(self, path: str) -> None:
        """Export to YAML file."""

    def explain(self) -> str:
        """Get human-readable explanation."""
```

### AutoEvalResult

```python
@dataclass
class AutoEvalResult:
    passed: bool                    # Overall pass/fail
    scan_result: PipelineResult     # Scanner results
    eval_result: EvaluatorResult    # Evaluation results
    blocked_by_scanner: bool        # If scanners blocked before evals
    total_latency_ms: float         # Total execution time

    def explain(self) -> str:
        """Get detailed explanation of results."""

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
```

## Best Practices

1. **Start with templates** for common use cases, then customize
2. **Use descriptions** when your app doesn't fit standard templates
3. **Export configurations** for reproducibility and code review
4. **Adjust thresholds** based on your risk tolerance
5. **Add domain-specific scanners** for regulated industries
6. **Test with adversarial inputs** to verify safety measures
7. **Monitor results** and tune thresholds based on false positive/negative rates
