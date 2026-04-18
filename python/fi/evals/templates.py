"""
Template class shells for cloud evals.

These classes exist purely for IDE autocomplete and backward compatibility
with code that does ``from fi.evals import Toxicity``. Input schemas are
no longer hardcoded here — the cloud registry (``fi.evals.core.cloud_registry``)
fetches ``required_keys`` from the api and maps user inputs dynamically.

To call an eval by name without importing a class, just pass the string::

    evaluate("customer_agent_query_handling", conversation=[...], model="turing_flash")

Renamed templates have a module-level alias from the old name to the new
class (e.g. ``NoOpenAIReference = NoLLMReference``) so existing user code
keeps working seamlessly. Templates that were removed outright are gone —
an ``ImportError`` is a clearer signal than a silent runtime 400.
"""
from typing import Any, Dict, List, Optional

from fi.utils.errors import MissingRequiredConfigForEvalTemplate


# ---------------------------------------------------------------------------
# EvalTemplate base class
# ---------------------------------------------------------------------------

class EvalTemplate:
    """Lightweight marker for cloud eval templates.

    Input filtering and validation happen dynamically via the cloud
    registry — no hardcoded Input schema.
    """

    eval_id: str = ""
    eval_name: str = ""
    description: str = ""
    eval_tags: List[str] = []
    required_keys: List[str] = []
    output: str = ""
    eval_type_id: str = ""
    config_schema: Dict[str, Any] = {}
    criteria: str = ""
    choices: List[str] = []
    multi_choice: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def __repr__(self) -> str:
        return f"EvalTemplate(name={self.eval_name})"

    def validate_config(self, config: Dict[str, Any]) -> None:
        for key in self.config_schema:
            if key not in config:
                raise MissingRequiredConfigForEvalTemplate(key, self.eval_name)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ConversationCoherence(EvalTemplate):
    eval_name = "conversation_coherence"
    eval_id = "1"


class ConversationResolution(EvalTemplate):
    eval_name = "conversation_resolution"
    eval_id = "2"


# ---------------------------------------------------------------------------
# Content moderation & safety
# ---------------------------------------------------------------------------

class ContentModeration(EvalTemplate):
    eval_name = "content_moderation"
    eval_id = "4"


class PII(EvalTemplate):
    eval_name = "pii"
    eval_id = "14"


class Toxicity(EvalTemplate):
    eval_name = "toxicity"
    eval_id = "15"


class Sexist(EvalTemplate):
    eval_name = "sexist"
    eval_id = "17"


class PromptInjection(EvalTemplate):
    eval_name = "prompt_injection"
    eval_id = "18"


class DataPrivacyCompliance(EvalTemplate):
    eval_name = "data_privacy_compliance"
    eval_id = "22"


class NoRacialBias(EvalTemplate):
    eval_name = "no_racial_bias"
    eval_id = "77"


class NoGenderBias(EvalTemplate):
    eval_name = "no_gender_bias"
    eval_id = "78"


class NoAgeBias(EvalTemplate):
    eval_name = "no_age_bias"
    eval_id = "79"


class NoLLMReference(EvalTemplate):
    """Checks that the output doesn't reference the underlying LLM (e.g. 'I am GPT…').

    Replaced the old ``NoOpenAIReference`` template in the api revamp.
    """

    eval_name = "no_llm_reference"


# Alias: old class name → current eval. Import-level backward compatibility.
NoOpenAIReference = NoLLMReference


class NoApologies(EvalTemplate):
    eval_name = "no_apologies"
    eval_id = "81"


class ContentSafety(EvalTemplate):
    eval_name = "content_safety_violation"
    eval_id = "93"


class NoHarmfulTherapeuticGuidance(EvalTemplate):
    eval_name = "no_harmful_therapeutic_guidance"
    eval_id = "90"


class ClinicallyInappropriateTone(EvalTemplate):
    eval_name = "clinically_inappropriate_tone"
    eval_id = "91"


class IsHarmfulAdvice(EvalTemplate):
    eval_name = "is_harmful_advice"
    eval_id = "92"


# ---------------------------------------------------------------------------
# RAG / Grounding
# ---------------------------------------------------------------------------

class ContextAdherence(EvalTemplate):
    eval_name = "context_adherence"
    eval_id = "5"


class ContextRelevance(EvalTemplate):
    eval_name = "context_relevance"
    eval_id = "9"


class Completeness(EvalTemplate):
    eval_name = "completeness"
    eval_id = "10"


class ChunkAttribution(EvalTemplate):
    eval_name = "chunk_attribution"
    eval_id = "11"


class ChunkUtilization(EvalTemplate):
    eval_name = "chunk_utilization"
    eval_id = "12"


class Groundedness(EvalTemplate):
    eval_name = "groundedness"
    eval_id = "47"


class FactualAccuracy(EvalTemplate):
    eval_name = "factual_accuracy"
    eval_id = "66"


class DetectHallucination(EvalTemplate):
    """Detects hallucinated or unsupported claims in the output.

    Replaced the old ``DetectHallucinationMissingInfo`` template in the revamp.
    """

    eval_name = "detect_hallucination"


# Alias: old class name → current eval.
DetectHallucinationMissingInfo = DetectHallucination


class IsFactuallyConsistent(EvalTemplate):
    eval_name = "is_factually_consistent"
    eval_id = "95"


# ---------------------------------------------------------------------------
# Tone & quality
# ---------------------------------------------------------------------------

class Tone(EvalTemplate):
    eval_name = "tone"
    eval_id = "16"


class PromptAdherence(EvalTemplate):
    eval_name = "prompt_adherence"
    eval_id = "65"


class IsPolite(EvalTemplate):
    eval_name = "is_polite"
    eval_id = "82"


class IsConcise(EvalTemplate):
    eval_name = "is_concise"
    eval_id = "83"


class IsHelpful(EvalTemplate):
    eval_name = "is_helpful"
    eval_id = "84"


class IsInformalTone(EvalTemplate):
    eval_name = "is_informal_tone"
    eval_id = "97"


class AnswerRefusal(EvalTemplate):
    eval_name = "answer_refusal"
    eval_id = "88"


class TaskCompletion(EvalTemplate):
    eval_name = "task_completion"
    eval_id = "99"


# ---------------------------------------------------------------------------
# Format / structure
# ---------------------------------------------------------------------------

class IsJson(EvalTemplate):
    eval_name = "is_json"
    eval_id = "23"


class OneLine(EvalTemplate):
    eval_name = "one_line"
    eval_id = "38"


class ContainsValidLink(EvalTemplate):
    eval_name = "contains_valid_link"
    eval_id = "39"


class IsEmail(EvalTemplate):
    eval_name = "is_email"
    eval_id = "40"


class ContainsCode(EvalTemplate):
    """New in the api revamp."""

    eval_name = "contains_code"


# ---------------------------------------------------------------------------
# Comparison / matching
# ---------------------------------------------------------------------------

class Ranking(EvalTemplate):
    eval_name = "eval_ranking"
    eval_id = "61"


class SummaryQuality(EvalTemplate):
    eval_name = "summary_quality"
    eval_id = "64"


class IsGoodSummary(EvalTemplate):
    eval_name = "is_good_summary"
    eval_id = "94"


class FuzzyMatch(EvalTemplate):
    eval_name = "fuzzy_match"
    eval_id = "87"


class BleuScore(EvalTemplate):
    eval_name = "bleu_score"
    eval_id = "101"


class TranslationAccuracy(EvalTemplate):
    eval_name = "translation_accuracy"
    eval_id = "67"


class CulturalSensitivity(EvalTemplate):
    eval_name = "cultural_sensitivity"
    eval_id = "68"


class BiasDetection(EvalTemplate):
    eval_name = "bias_detection"
    eval_id = "69"


class GroundTruthMatch(EvalTemplate):
    """New in the api revamp."""

    eval_name = "ground_truth_match"


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------

class EvaluateFunctionCalling(EvalTemplate):
    eval_name = "evaluate_function_calling"
    eval_id = "98"


# Alias: old class name → current eval.
LLMFunctionCalling = EvaluateFunctionCalling


# ---------------------------------------------------------------------------
# Domain / agent / multimodal
# ---------------------------------------------------------------------------

class TextToSQL(EvalTemplate):
    """New in the api revamp."""

    eval_name = "text_to_sql"


class PromptInstructionAdherence(EvalTemplate):
    """New in the api revamp."""

    eval_name = "prompt_instruction_adherence"


class ProtectFlash(EvalTemplate):
    """New in the api revamp — lightweight prompt-injection check."""

    eval_name = "protect_flash"


class ImageInstructionAdherence(EvalTemplate):
    """New in the api revamp."""

    eval_name = "image_instruction_adherence"


class SyntheticImageEvaluator(EvalTemplate):
    """New in the api revamp."""

    eval_name = "synthetic_image_evaluator"


class OCREvaluation(EvalTemplate):
    """New in the api revamp."""

    eval_name = "ocr_evaluation"


class ASRAccuracy(EvalTemplate):
    """Speech-to-text accuracy eval.

    Replaced the old ``AudioTranscriptionEvaluator`` in the revamp.
    """

    eval_name = "ASR/STT_accuracy"


# Alias: old class name → current eval.
AudioTranscriptionEvaluator = ASRAccuracy


class TTSAccuracy(EvalTemplate):
    """New in the api revamp."""

    eval_name = "TTS_accuracy"


class AudioQualityEvaluator(EvalTemplate):
    eval_name = "audio_quality"
    eval_id = "75"


class CaptionHallucination(EvalTemplate):
    eval_name = "caption_hallucination"
    eval_id = "100"


class IsCompliant(EvalTemplate):
    eval_name = "is_compliant"
    eval_id = "96"


# ---------------------------------------------------------------------------
# Customer-agent family (new in revamp)
# ---------------------------------------------------------------------------

class CustomerAgentClarificationSeeking(EvalTemplate):
    eval_name = "customer_agent_clarification_seeking"


class CustomerAgentContextRetention(EvalTemplate):
    eval_name = "customer_agent_context_retention"


class CustomerAgentConversationQuality(EvalTemplate):
    eval_name = "customer_agent_conversation_quality"


class CustomerAgentHumanEscalation(EvalTemplate):
    eval_name = "customer_agent_human_escalation"


class CustomerAgentInterruptionHandling(EvalTemplate):
    eval_name = "customer_agent_interruption_handling"


class CustomerAgentLanguageHandling(EvalTemplate):
    eval_name = "customer_agent_language_handling"


class CustomerAgentLoopDetection(EvalTemplate):
    eval_name = "customer_agent_loop_detection"


class CustomerAgentObjectionHandling(EvalTemplate):
    eval_name = "customer_agent_objection_handling"


class CustomerAgentPromptConformance(EvalTemplate):
    eval_name = "customer_agent_prompt_conformance"


class CustomerAgentQueryHandling(EvalTemplate):
    eval_name = "customer_agent_query_handling"


class CustomerAgentTerminationHandling(EvalTemplate):
    eval_name = "customer_agent_termination_handling"
