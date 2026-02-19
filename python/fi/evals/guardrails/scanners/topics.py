"""
Topic Restriction Scanner for Guardrails.

Restricts conversations to allowed topics and detects off-topic content.
Supports both keyword-based and semantic embedding-based detection.
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


# Topic descriptions for semantic embedding matching
TOPIC_DESCRIPTIONS: Dict[str, str] = {
    "politics": "Political discussions about elections, voting, government, political parties, politicians, legislation, and policy debates",
    "religion": "Religious discussions about faith, worship, God, spiritual beliefs, churches, mosques, temples, and religious texts",
    "violence": "Violent content including weapons, attacks, murder, assault, warfare, fighting, and physical harm",
    "drugs": "Drug-related content about narcotics, illegal substances, drug use, addiction, and trafficking",
    "adult_content": "Adult or sexual content including pornography, explicit material, and NSFW topics",
    "gambling": "Gambling discussions about casinos, betting, poker, lottery, and wagering",
    "medical_advice": "Medical advice about diagnoses, treatments, medications, symptoms, and health conditions",
    "financial_advice": "Financial advice about investments, stocks, trading, cryptocurrency, and portfolio management",
    "legal_advice": "Legal advice about lawsuits, attorneys, court cases, litigation, and legal proceedings",
    "customer_support": "Customer support topics like orders, shipping, refunds, account issues, and billing questions",
    "product_info": "Product information about features, specifications, pricing, availability, and warranties",
    "technical_support": "Technical support for software errors, bugs, installation, configuration, and troubleshooting",
    "general_knowledge": "General knowledge questions about facts, history, science, geography, and explanations",
}

# Predefined topic keywords for common restrictions
TOPIC_KEYWORDS: Dict[str, Set[str]] = {
    # Sensitive topics often restricted
    "politics": {
        "election", "vote", "democrat", "republican", "liberal", "conservative",
        "president", "congress", "senate", "parliament", "politician", "government",
        "left-wing", "right-wing", "campaign", "ballot", "trump", "biden", "party",
    },
    "religion": {
        "god", "jesus", "allah", "buddha", "church", "mosque", "temple", "prayer",
        "bible", "quran", "torah", "christian", "muslim", "jewish", "hindu",
        "atheist", "agnostic", "faith", "worship", "salvation", "sin", "heaven", "hell",
    },
    "violence": {
        "kill", "murder", "attack", "assault", "weapon", "gun", "bomb", "terrorist",
        "violence", "violent", "hurt", "harm", "blood", "death", "dead", "shoot",
        "stab", "fight", "war", "battle", "combat",
    },
    "drugs": {
        "cocaine", "heroin", "marijuana", "cannabis", "weed", "meth", "lsd", "mdma",
        "drug", "narcotic", "overdose", "addiction", "dealer", "cartel", "trafficking",
    },
    "adult_content": {
        "sex", "porn", "nude", "naked", "erotic", "explicit", "xxx", "nsfw",
        "fetish", "intimate", "sexual", "genitals",
    },
    "gambling": {
        "casino", "bet", "betting", "gamble", "gambling", "poker", "blackjack",
        "slot", "lottery", "wager", "odds", "bookie", "sportsbook",
    },
    "medical_advice": {
        "diagnosis", "treatment", "medication", "prescription", "dosage", "symptom",
        "disease", "illness", "cure", "therapy", "doctor", "patient",
    },
    "financial_advice": {
        "invest", "investment", "stock", "bond", "portfolio", "trading", "forex",
        "crypto", "bitcoin", "dividend", "retirement", "pension",
    },
    "legal_advice": {
        "lawsuit", "attorney", "lawyer", "court", "judge", "verdict", "settlement",
        "litigation", "defendant", "plaintiff", "legal", "illegal",
    },

    # Common allowed topics
    "customer_support": {
        "order", "shipping", "delivery", "refund", "return", "exchange", "tracking",
        "account", "password", "login", "subscription", "billing", "payment",
        "help", "support", "issue", "problem", "question",
    },
    "product_info": {
        "product", "feature", "specification", "price", "availability", "warranty",
        "size", "color", "model", "version", "compatible",
    },
    "technical_support": {
        "error", "bug", "crash", "install", "update", "download", "configure",
        "setup", "troubleshoot", "debug", "fix", "issue", "problem",
    },
    "general_knowledge": {
        "what", "how", "why", "when", "where", "who", "explain", "describe",
        "define", "meaning", "history", "science", "math", "geography",
    },
}


@register_scanner("topics")
class TopicRestrictionScanner(BaseScanner):
    """
    Scanner for topic restriction and off-topic detection.

    Supports two detection modes:
    - Keyword-based: Fast pattern matching (default)
    - Semantic: Embedding-based similarity matching

    Restricts conversations to allowed topics or blocks denied topics.

    Usage:
        # Keyword-based (fast, no dependencies)
        scanner = TopicRestrictionScanner(
            allowed_topics=["customer_support", "product_info"],
        )

        # Semantic embedding-based (requires sentence-transformers)
        scanner = TopicRestrictionScanner(
            denied_topics=["politics", "religion"],
            use_embeddings=True,
        )

        # Hybrid mode - combines both approaches
        scanner = TopicRestrictionScanner(
            allowed_topics=["customer_support"],
            use_embeddings=True,
            combine_scores=True,
        )

        # Custom topic descriptions for semantic matching
        scanner = TopicRestrictionScanner(
            custom_topic_descriptions={
                "insurance_claims": "Insurance claim processing, policy coverage, claim status",
            },
            allowed_topics=["insurance_claims"],
            use_embeddings=True,
        )

        result = scanner.scan("Who should I vote for in the election?")
        if not result.passed:
            print(f"Off-topic: {result.metadata.get('detected_topics')}")
    """

    name = "topics"
    category = "topic_restriction"
    description = "Restricts conversations to allowed topics"
    default_action = ScannerAction.FLAG

    # Default embedding model
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
        threshold: float = 0.5,
        allowed_topics: Optional[List[str]] = None,
        denied_topics: Optional[List[str]] = None,
        custom_topics: Optional[Dict[str, Set[str]]] = None,
        custom_topic_descriptions: Optional[Dict[str, str]] = None,
        min_keyword_matches: int = 2,
        case_sensitive: bool = False,
        use_embeddings: bool = False,
        embedding_model: Optional[str] = None,
        combine_scores: bool = True,
        embedding_weight: float = 0.6,
        keyword_weight: float = 0.4,
        semantic_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize topic restriction scanner.

        Args:
            action: Action on detection
            enabled: Whether scanner is enabled
            threshold: Confidence threshold (based on keyword match ratio)
            allowed_topics: List of allowed topic names (whitelist mode)
            denied_topics: List of denied topic names (blacklist mode)
            custom_topics: Custom topic definitions {topic_name: {keywords}}
            custom_topic_descriptions: Custom descriptions for semantic matching
            min_keyword_matches: Minimum keyword matches to detect a topic
            case_sensitive: Whether matching is case-sensitive
            use_embeddings: Enable semantic embedding-based detection
            embedding_model: Model name for embeddings (default: all-MiniLM-L6-v2)
            combine_scores: Combine keyword and embedding scores (hybrid mode)
            embedding_weight: Weight for embedding score in combined mode
            keyword_weight: Weight for keyword score in combined mode
            semantic_threshold: Similarity threshold for semantic matching
            device: Device for embedding model ('cpu', 'cuda', 'mps', or None)
        """
        super().__init__(action, enabled)
        self.threshold = threshold
        self.allowed_topics = set(allowed_topics) if allowed_topics else None
        self.denied_topics = set(denied_topics) if denied_topics else None
        self.min_keyword_matches = min_keyword_matches
        self.case_sensitive = case_sensitive

        # Embedding settings
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.combine_scores = combine_scores
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight
        self.semantic_threshold = semantic_threshold
        self.device = device

        # Build topic dictionaries
        self.topics = TOPIC_KEYWORDS.copy()
        if custom_topics:
            self.topics.update(custom_topics)

        self.topic_descriptions = TOPIC_DESCRIPTIONS.copy()
        if custom_topic_descriptions:
            self.topic_descriptions.update(custom_topic_descriptions)

        # Compile patterns for keyword matching
        self._topic_patterns: Dict[str, List[re.Pattern]] = {}
        for topic, keywords in self.topics.items():
            flags = 0 if case_sensitive else re.IGNORECASE
            patterns = [
                re.compile(r'\b' + re.escape(kw) + r'\b', flags)
                for kw in keywords
            ]
            self._topic_patterns[topic] = patterns

        # Lazy-loaded embedding components
        self._embedding_model: Optional[Any] = None
        self._topic_embeddings: Optional[Dict[str, Any]] = None
        self._embeddings_available = False
        self._embeddings_load_error: Optional[str] = None

        # Pre-load embeddings if requested
        if use_embeddings:
            self._load_embedding_model()

    def _load_embedding_model(self) -> bool:
        """
        Lazy load the embedding model and compute topic embeddings.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._embedding_model is not None:
            return self._embeddings_available

        try:
            from sentence_transformers import SentenceTransformer

            # Determine device
            device = self.device
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except ImportError:
                    device = "cpu"

            # Load embedding model
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device,
            )
            self._device = device

            # Pre-compute topic embeddings
            self._compute_topic_embeddings()

            self._embeddings_available = True
            return True

        except ImportError as e:
            self._embeddings_load_error = f"sentence-transformers not installed: {e}"
            self._embeddings_available = False
            return False
        except Exception as e:
            self._embeddings_load_error = f"Failed to load embedding model: {e}"
            self._embeddings_available = False
            return False

    def _compute_topic_embeddings(self) -> None:
        """Pre-compute embeddings for all topic descriptions."""
        if self._embedding_model is None:
            return

        self._topic_embeddings = {}

        # Determine which topics to compute embeddings for
        relevant_topics = set()
        if self.allowed_topics:
            relevant_topics.update(self.allowed_topics)
        if self.denied_topics:
            relevant_topics.update(self.denied_topics)
        if not relevant_topics:
            # Compute for all topics if no specific restriction
            relevant_topics = set(self.topic_descriptions.keys())

        for topic in relevant_topics:
            if topic in self.topic_descriptions:
                description = self.topic_descriptions[topic]
            else:
                # Fallback to keywords as description
                keywords = self.topics.get(topic, set())
                description = " ".join(keywords) if keywords else topic

            embedding = self._embedding_model.encode(
                description,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            self._topic_embeddings[topic] = embedding

    def _semantic_similarity(
        self, content: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute semantic similarity between content and topic descriptions.

        Args:
            content: Text to analyze

        Returns:
            Dict of {topic: {similarity, confidence}}
        """
        if not self._embeddings_available or self._embedding_model is None:
            return {}

        try:
            from sentence_transformers import util

            # Encode content
            content_embedding = self._embedding_model.encode(
                content,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            results = {}
            for topic, topic_embedding in self._topic_embeddings.items():
                # Compute cosine similarity
                similarity = util.cos_sim(content_embedding, topic_embedding).item()

                # Normalize to 0-1 range (cosine similarity can be negative)
                confidence = max(0.0, (similarity + 1.0) / 2.0)

                results[topic] = {
                    "similarity": similarity,
                    "confidence": confidence,
                    "method": "semantic",
                }

            return results

        except Exception as e:
            # Return empty on error
            return {}

    def _detect_topics_keywords(self, text: str) -> Dict[str, Dict]:
        """
        Detect topics in text based on keyword matching.

        Returns:
            Dict of {topic: {count, keywords, confidence, method}}
        """
        detected = {}

        for topic, patterns in self._topic_patterns.items():
            matched_keywords = []
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    matched_keywords.extend(matches)

            if len(matched_keywords) >= self.min_keyword_matches:
                # Calculate confidence based on keyword density
                total_keywords = len(self.topics[topic])
                unique_matches = len(set(kw.lower() for kw in matched_keywords))
                confidence = min(unique_matches / total_keywords * 2, 1.0)

                detected[topic] = {
                    "count": len(matched_keywords),
                    "unique_count": unique_matches,
                    "keywords": list(set(matched_keywords)),
                    "confidence": confidence,
                    "method": "keyword",
                }

        return detected

    def _detect_topics(self, text: str) -> Dict[str, Dict]:
        """
        Detect topics using configured method(s).

        Returns:
            Dict of {topic: {confidence, method, ...}}
        """
        # Get keyword-based detection
        keyword_results = self._detect_topics_keywords(text)

        # Get semantic detection if enabled
        semantic_results = {}
        if self.use_embeddings and self._embeddings_available:
            semantic_results = self._semantic_similarity(text)

        # Combine results
        if not self.use_embeddings or not self._embeddings_available:
            return keyword_results

        if not self.combine_scores:
            # Semantic-only mode (filter by threshold)
            return {
                topic: info
                for topic, info in semantic_results.items()
                if info["confidence"] >= self.semantic_threshold
            }

        # Hybrid mode: combine scores
        all_topics = set(keyword_results.keys()) | set(semantic_results.keys())
        combined = {}

        for topic in all_topics:
            kw_info = keyword_results.get(topic, {})
            sem_info = semantic_results.get(topic, {})

            kw_conf = kw_info.get("confidence", 0.0)
            sem_conf = sem_info.get("confidence", 0.0)

            # Weighted combination
            combined_confidence = (
                self.keyword_weight * kw_conf +
                self.embedding_weight * sem_conf
            )

            combined[topic] = {
                "confidence": combined_confidence,
                "keyword_confidence": kw_conf,
                "semantic_confidence": sem_conf,
                "method": "hybrid",
            }

            # Include keyword details if available
            if kw_info:
                combined[topic]["keywords"] = kw_info.get("keywords", [])
                combined[topic]["count"] = kw_info.get("count", 0)

            # Include semantic similarity if available
            if sem_info:
                combined[topic]["similarity"] = sem_info.get("similarity", 0.0)

        return combined

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content for topic violations.

        Uses keyword matching, semantic similarity, or both depending on config.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with topic detection details
        """
        start = time.perf_counter()
        matches = []
        issues = []
        metadata: Dict[str, Any] = {}

        # Record detection mode
        if self.use_embeddings and self._embeddings_available:
            metadata["detection_mode"] = "hybrid" if self.combine_scores else "semantic"
            metadata["embedding_model"] = self.embedding_model_name
        else:
            metadata["detection_mode"] = "keyword"
            if self.use_embeddings and not self._embeddings_available:
                metadata["embedding_error"] = self._embeddings_load_error

        # Detect topics
        detected_topics = self._detect_topics(content)

        # Also check context if provided
        if context:
            context_topics = self._detect_topics(context)
            for topic, info in context_topics.items():
                if topic in detected_topics:
                    # Merge: take max confidence
                    if info["confidence"] > detected_topics[topic]["confidence"]:
                        detected_topics[topic] = info
                else:
                    detected_topics[topic] = info

        # Check topic restrictions
        violation = False

        # Whitelist mode: only allowed topics are permitted
        if self.allowed_topics:
            for topic, info in detected_topics.items():
                if info["confidence"] >= self.threshold:
                    if topic not in self.allowed_topics:
                        matches.append(ScanMatch(
                            pattern_name="off_topic",
                            matched_text=f"Topic: {topic}",
                            start=0,
                            end=len(content),
                            confidence=info["confidence"],
                            metadata={
                                "topic": topic,
                                "keywords": info.get("keywords", []),
                                "method": info.get("method", "unknown"),
                            },
                        ))
                        issues.append(f"Off-topic: {topic}")
                        violation = True

            # Also flag if no allowed topic was detected
            allowed_detected = any(
                topic in self.allowed_topics and info["confidence"] >= self.threshold
                for topic, info in detected_topics.items()
            )
            if not allowed_detected and not violation and detected_topics:
                # Content doesn't match any allowed topic
                matches.append(ScanMatch(
                    pattern_name="no_allowed_topic",
                    matched_text="No allowed topic detected",
                    start=0,
                    end=len(content),
                    confidence=0.6,
                ))
                issues.append("No allowed topic detected")
                violation = True

        # Blacklist mode: denied topics are blocked
        if self.denied_topics:
            for topic, info in detected_topics.items():
                if topic in self.denied_topics and info["confidence"] >= self.threshold:
                    matches.append(ScanMatch(
                        pattern_name="denied_topic",
                        matched_text=f"Topic: {topic}",
                        start=0,
                        end=len(content),
                        confidence=info["confidence"],
                        metadata={
                            "topic": topic,
                            "keywords": info.get("keywords", []),
                            "method": info.get("method", "unknown"),
                        },
                    ))
                    issues.append(f"Denied topic: {topic}")
                    violation = True

        latency = (time.perf_counter() - start) * 1000

        # Determine result
        max_confidence = max([m.confidence for m in matches], default=0.0)

        metadata["detected_topics"] = {
            k: v for k, v in detected_topics.items()
            if v["confidence"] >= self.threshold
        }

        if violation:
            return self._create_result(
                passed=False,
                matches=matches,
                score=max_confidence,
                reason="; ".join(issues),
                latency_ms=latency,
                metadata=metadata,
            )

        return self._create_result(
            passed=True,
            matches=[],
            score=0.0,
            reason="Content is on-topic",
            latency_ms=latency,
            metadata=metadata,
        )

    @classmethod
    def with_embeddings(
        cls,
        allowed_topics: Optional[List[str]] = None,
        denied_topics: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> "TopicRestrictionScanner":
        """
        Factory method to create an embedding-enabled topic scanner.

        Args:
            allowed_topics: Allowed topic list
            denied_topics: Denied topic list
            embedding_model: Model to use (defaults to all-MiniLM-L6-v2)
            threshold: Detection threshold
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured TopicRestrictionScanner with embeddings enabled
        """
        return cls(
            allowed_topics=allowed_topics,
            denied_topics=denied_topics,
            use_embeddings=True,
            embedding_model=embedding_model,
            threshold=threshold,
            **kwargs,
        )

    @classmethod
    def semantic_only(
        cls,
        allowed_topics: Optional[List[str]] = None,
        denied_topics: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> "TopicRestrictionScanner":
        """
        Factory method to create a semantic-only topic scanner (no keywords).

        Args:
            allowed_topics: Allowed topic list
            denied_topics: Denied topic list
            embedding_model: Model to use
            threshold: Detection threshold
            **kwargs: Additional arguments

        Returns:
            Configured TopicRestrictionScanner using semantic-only detection
        """
        return cls(
            allowed_topics=allowed_topics,
            denied_topics=denied_topics,
            use_embeddings=True,
            combine_scores=False,
            embedding_model=embedding_model,
            threshold=threshold,
            **kwargs,
        )
