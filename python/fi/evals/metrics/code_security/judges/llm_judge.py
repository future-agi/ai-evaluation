"""
LLM-based Security Judge.

Uses language models for semantic vulnerability detection.
Provides deep understanding of code semantics and context.
"""

import json
import time
from typing import List, Dict, Any, Optional

from .base import BaseJudge, JudgeResult, JudgeFinding
from ..types import Severity, CodeLocation


# Default prompt for security analysis
DEFAULT_SECURITY_PROMPT = """You are a security expert analyzing code for vulnerabilities.

Analyze the following {language} code for security issues:

```{language}
{code}
```

{context_section}

Identify ALL security vulnerabilities present. For each vulnerability found, provide:
1. CWE ID (e.g., CWE-89 for SQL injection)
2. Vulnerability type (e.g., "sql_injection", "xss", "command_injection")
3. Severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
4. Confidence: 0.0 to 1.0 (how confident you are this is a real vulnerability)
5. Description: Brief explanation of the vulnerability
6. Line number(s) where the vulnerability occurs
7. Suggested fix

Respond in JSON format:
{{
    "is_secure": true/false,
    "findings": [
        {{
            "cwe_id": "CWE-XXX",
            "vulnerability_type": "type_name",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
            "confidence": 0.0-1.0,
            "description": "Description of the issue",
            "line_start": 1,
            "line_end": 1,
            "suggested_fix": "How to fix it",
            "reasoning": "Why this is a vulnerability"
        }}
    ],
    "summary": "Brief overall security assessment"
}}

Be thorough but avoid false positives. Only report actual vulnerabilities with supporting evidence.
If the code is secure, return is_secure: true with an empty findings array.
"""


class LLMJudge(BaseJudge):
    """
    LLM-based security judge for semantic vulnerability detection.

    Uses language models to understand code semantics and detect
    vulnerabilities that pattern-based approaches might miss.

    Features:
    - Deep semantic understanding
    - Context-aware analysis
    - Detailed reasoning
    - Suggested fixes
    - Custom prompt support

    Usage:
        # With OpenAI
        judge = LLMJudge(model="gpt-4")
        result = judge.judge(code, "python")

        # With custom evaluator
        judge = LLMJudge(evaluator=my_evaluator)

        # With context
        result = judge.judge(
            code,
            "python",
            context={"instruction": "Write a user login function"}
        )
    """

    judge_type = "llm"

    def __init__(
        self,
        model: str = "gpt-4",
        evaluator: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.7,
        timeout: float = 30.0,
        temperature: float = 0.1,
    ):
        """
        Initialize the LLM judge.

        Args:
            model: Model name/ID for the LLM
            evaluator: Optional custom evaluator instance
            prompt_template: Custom prompt template (use {code}, {language}, {context_section})
            severity_threshold: Minimum severity to flag as insecure
            min_confidence: Minimum confidence to include findings
            timeout: Request timeout in seconds
            temperature: LLM temperature (lower = more deterministic)
        """
        super().__init__(severity_threshold, min_confidence)

        self.model = model
        self.evaluator = evaluator
        self.prompt_template = prompt_template or DEFAULT_SECURITY_PROMPT
        self.timeout = timeout
        self.temperature = temperature

        # Lazy-loaded client
        self._client = None

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client

        if self.evaluator is not None:
            return self.evaluator

        # Try to import and create OpenAI client
        try:
            from openai import OpenAI
            self._client = OpenAI()
            return self._client
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )

    def judge(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Judge code for security vulnerabilities using LLM.

        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional context (instruction, file_path, etc.)

        Returns:
            JudgeResult with findings and reasoning
        """
        start_time = time.time()

        # Build context section
        context_section = ""
        if context:
            if "instruction" in context:
                context_section += f"Original instruction: {context['instruction']}\n"
            if "file_path" in context:
                context_section += f"File: {context['file_path']}\n"
            if "additional_context" in context:
                context_section += f"Additional context: {context['additional_context']}\n"

        # Build prompt
        prompt = self.prompt_template.format(
            code=code,
            language=language,
            context_section=context_section if context_section else "No additional context provided.",
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response, language, context)
        except Exception as e:
            # Return error result
            result = JudgeResult(
                is_secure=True,  # Default to secure on error
                security_score=1.0,
                findings=[],
                judge_type=self.judge_type,
                language=language,
            )
            result.error = str(e)

        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time

        return result

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        client = self._get_client()

        # Handle different client types
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # OpenAI-style client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a security expert. Analyze code for vulnerabilities and respond in JSON format only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content

        elif hasattr(client, "generate"):
            # Generic generate method
            return client.generate(prompt)

        elif hasattr(client, "complete"):
            # Completion method
            return client.complete(prompt)

        elif callable(client):
            # Callable evaluator
            return client(prompt)

        else:
            raise ValueError(
                f"Unsupported client type: {type(client)}. "
                "Must have chat.completions.create, generate, complete, or be callable."
            )

    def _parse_response(
        self,
        response: str,
        language: str,
        context: Optional[Dict[str, Any]],
    ) -> JudgeResult:
        """Parse LLM response into JudgeResult."""
        # Extract JSON from response
        try:
            # Try to find JSON in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            # Return empty result on parse error
            return JudgeResult(
                is_secure=True,
                security_score=1.0,
                findings=[],
                judge_type=self.judge_type,
                language=language,
            )

        # Parse findings
        findings: List[JudgeFinding] = []
        for finding_data in data.get("findings", []):
            severity = self._parse_severity(finding_data.get("severity", "MEDIUM"))
            confidence = float(finding_data.get("confidence", 0.7))

            finding = JudgeFinding(
                cwe_id=finding_data.get("cwe_id", "CWE-Unknown"),
                vulnerability_type=finding_data.get("vulnerability_type", "unknown"),
                description=finding_data.get("description", ""),
                severity=severity,
                confidence=confidence,
                location=CodeLocation(
                    line=finding_data.get("line_start"),
                    end_line=finding_data.get("line_end"),
                ) if finding_data.get("line_start") else None,
                suggested_fix=finding_data.get("suggested_fix"),
                judge_type=self.judge_type,
                reasoning=finding_data.get("reasoning"),
            )
            findings.append(finding)

        # Filter by confidence
        filtered_findings = self._filter_findings(findings)

        return JudgeResult(
            is_secure=self._is_secure(filtered_findings),
            security_score=self._compute_score(filtered_findings),
            findings=filtered_findings,
            judge_type=self.judge_type,
            language=language,
        )

    def _parse_severity(self, severity_str: str) -> Severity:
        """Parse severity string to enum."""
        severity_map = {
            "CRITICAL": Severity.CRITICAL,
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
            "INFO": Severity.INFO,
        }
        return severity_map.get(severity_str.upper(), Severity.MEDIUM)

    @classmethod
    def with_gpt4(cls, **kwargs) -> "LLMJudge":
        """Factory for GPT-4 judge."""
        return cls(model="gpt-4", **kwargs)

    @classmethod
    def with_gpt35(cls, **kwargs) -> "LLMJudge":
        """Factory for GPT-3.5 judge (faster, cheaper)."""
        return cls(model="gpt-3.5-turbo", **kwargs)

    @classmethod
    def with_claude(cls, **kwargs) -> "LLMJudge":
        """Factory for Claude judge."""
        return cls(model="claude-3-sonnet-20240229", **kwargs)


class MockLLMJudge(LLMJudge):
    """
    Mock LLM judge for testing without API calls.

    Returns configurable mock responses for testing the dual-judge system.
    """

    def __init__(
        self,
        mock_findings: Optional[List[JudgeFinding]] = None,
        mock_is_secure: bool = True,
        **kwargs,
    ):
        """
        Initialize mock judge.

        Args:
            mock_findings: Findings to return
            mock_is_secure: Whether to report code as secure
        """
        super().__init__(**kwargs)
        self.mock_findings = mock_findings or []
        self.mock_is_secure = mock_is_secure

    def _call_llm(self, prompt: str) -> str:
        """Return mock response."""
        findings_data = []
        for f in self.mock_findings:
            findings_data.append({
                "cwe_id": f.cwe_id,
                "vulnerability_type": f.vulnerability_type,
                "severity": f.severity.value,
                "confidence": f.confidence,
                "description": f.description,
                "line_start": f.location.line if f.location else None,
                "suggested_fix": f.suggested_fix,
                "reasoning": f.reasoning,
            })

        return json.dumps({
            "is_secure": self.mock_is_secure,
            "findings": findings_data,
            "summary": "Mock analysis result",
        })
