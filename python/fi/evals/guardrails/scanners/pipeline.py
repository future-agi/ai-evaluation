"""
Scanner Pipeline for Guardrails.

Orchestrates multiple scanners to run in sequence or parallel.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.scanners.base import BaseScanner, ScanResult, ScannerAction


@dataclass
class PipelineResult:
    """Aggregated result from scanner pipeline."""
    passed: bool
    results: List[ScanResult]
    total_latency_ms: float
    blocked_by: List[str] = field(default_factory=list)
    flagged_by: List[str] = field(default_factory=list)

    @property
    def all_matches(self) -> List[Dict[str, Any]]:
        """Get all matches from all scanners."""
        matches = []
        for result in self.results:
            for match in result.matches:
                matches.append({
                    "scanner": result.scanner_name,
                    "category": result.category,
                    "pattern": match.pattern_name,
                    "text": match.matched_text,
                    "confidence": match.confidence,
                })
        return matches


class ScannerPipeline:
    """
    Pipeline that runs multiple scanners on content.

    Supports both sequential and parallel execution modes.
    Fast scanners (<10ms) can run in parallel for better throughput.

    Usage:
        pipeline = ScannerPipeline([
            JailbreakScanner(),
            CodeInjectionScanner(),
            SecretsScanner(),
        ])
        result = pipeline.scan("user input")
        if not result.passed:
            print(f"Blocked by: {result.blocked_by}")
    """

    def __init__(
        self,
        scanners: Optional[List[BaseScanner]] = None,
        parallel: bool = True,
        max_workers: int = 10,
        fail_fast: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            scanners: List of scanner instances
            parallel: Run scanners in parallel (default: True)
            max_workers: Max parallel workers
            fail_fast: Stop on first failure in sequential mode
        """
        self.scanners = scanners or []
        self.parallel = parallel
        self.max_workers = max_workers
        self.fail_fast = fail_fast

    def add_scanner(self, scanner: BaseScanner) -> "ScannerPipeline":
        """Add a scanner to the pipeline."""
        self.scanners.append(scanner)
        return self

    def remove_scanner(self, name: str) -> "ScannerPipeline":
        """Remove a scanner by name."""
        self.scanners = [s for s in self.scanners if s.name != name]
        return self

    def scan(self, content: str, context: Optional[str] = None) -> PipelineResult:
        """
        Run all scanners on content.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            PipelineResult with aggregated results
        """
        start = time.perf_counter()

        # Filter enabled scanners
        active_scanners = [s for s in self.scanners if s.enabled]

        if not active_scanners:
            return PipelineResult(
                passed=True,
                results=[],
                total_latency_ms=0.0,
            )

        if self.parallel:
            results = self._scan_parallel(active_scanners, content, context)
        else:
            results = self._scan_sequential(active_scanners, content, context)

        total_latency = (time.perf_counter() - start) * 1000

        # Aggregate results
        blocked_by = []
        flagged_by = []
        passed = True

        for result in results:
            if not result.passed:
                if result.action == ScannerAction.BLOCK:
                    blocked_by.append(result.scanner_name)
                    passed = False
                elif result.action == ScannerAction.FLAG:
                    flagged_by.append(result.scanner_name)

        return PipelineResult(
            passed=passed,
            results=results,
            total_latency_ms=total_latency,
            blocked_by=blocked_by,
            flagged_by=flagged_by,
        )

    def _scan_parallel(
        self,
        scanners: List[BaseScanner],
        content: str,
        context: Optional[str],
    ) -> List[ScanResult]:
        """Run scanners in parallel using thread pool."""
        results = []

        with ThreadPoolExecutor(max_workers=min(len(scanners), self.max_workers)) as executor:
            future_to_scanner = {
                executor.submit(scanner.scan, content, context): scanner
                for scanner in scanners
            }

            for future in as_completed(future_to_scanner):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    scanner = future_to_scanner[future]
                    results.append(ScanResult(
                        passed=True,  # Fail open on scanner error
                        scanner_name=scanner.name,
                        category=scanner.category,
                        reason=f"Scanner error: {str(e)}",
                    ))

        return results

    def _scan_sequential(
        self,
        scanners: List[BaseScanner],
        content: str,
        context: Optional[str],
    ) -> List[ScanResult]:
        """Run scanners sequentially."""
        results = []

        for scanner in scanners:
            try:
                result = scanner.scan(content, context)
                results.append(result)

                # Fail fast if enabled and scanner blocked
                if self.fail_fast and not result.passed and result.action == ScannerAction.BLOCK:
                    break

            except Exception as e:
                results.append(ScanResult(
                    passed=True,  # Fail open on scanner error
                    scanner_name=scanner.name,
                    category=scanner.category,
                    reason=f"Scanner error: {str(e)}",
                ))

        return results

    async def scan_async(self, content: str, context: Optional[str] = None) -> PipelineResult:
        """
        Run all scanners asynchronously.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            PipelineResult with aggregated results
        """
        start = time.perf_counter()

        active_scanners = [s for s in self.scanners if s.enabled]

        if not active_scanners:
            return PipelineResult(
                passed=True,
                results=[],
                total_latency_ms=0.0,
            )

        # Run all scanners concurrently
        tasks = [scanner.scan_async(content, context) for scanner in active_scanners]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scanner = active_scanners[i]
                processed_results.append(ScanResult(
                    passed=True,
                    scanner_name=scanner.name,
                    category=scanner.category,
                    reason=f"Scanner error: {str(result)}",
                ))
            else:
                processed_results.append(result)

        total_latency = (time.perf_counter() - start) * 1000

        # Aggregate
        blocked_by = []
        flagged_by = []
        passed = True

        for result in processed_results:
            if not result.passed:
                if result.action == ScannerAction.BLOCK:
                    blocked_by.append(result.scanner_name)
                    passed = False
                elif result.action == ScannerAction.FLAG:
                    flagged_by.append(result.scanner_name)

        return PipelineResult(
            passed=passed,
            results=processed_results,
            total_latency_ms=total_latency,
            blocked_by=blocked_by,
            flagged_by=flagged_by,
        )
