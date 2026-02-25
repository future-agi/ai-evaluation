from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pydantic import BaseModel, Field

from fi.utils.errors import (
    MissingRequiredKey,
    MissingRequiredConfigForEvalTemplate,
)


# ---------------------------------------------------------------------------
# Typed Input base classes — for IDE autocomplete and validation
# ---------------------------------------------------------------------------

class OutputOnly(BaseModel):
    """Templates that only need the model output."""
    output: str

class OutputWithContext(BaseModel):
    """Templates that need output + context."""
    output: str
    context: str
    input: Optional[str] = None

class OutputWithInput(BaseModel):
    """Templates that need output + input/query."""
    output: str
    input: str

class ConversationMessages(BaseModel):
    """Templates that operate on a conversation history."""
    messages: list

class OutputWithExpected(BaseModel):
    """Templates that compare output against expected output."""
    output: str
    expected_output: str
    input: Optional[str] = None

class ImageInput(BaseModel):
    """Templates that evaluate images."""
    output: str
    image_url: Optional[str] = None
    input_image_url: Optional[str] = None
    output_image_url: Optional[str] = None

class AudioInput(BaseModel):
    """Templates that evaluate audio."""
    output: str
    input: Optional[str] = None


# ---------------------------------------------------------------------------
# EvalTemplate base class
# ---------------------------------------------------------------------------

class EvalTemplate:
    eval_id: str
    eval_name: str
    description: str
    eval_tags: List[str]
    required_keys: List[str]
    output: str
    eval_type_id: str
    config_schema: Dict[str, Any]
    criteria: str
    choices: List[str]
    multi_choice: bool

    # Typed input model — subclasses override this for IDE support
    Input = None

    def __init__(self, config: Optional[Dict[str, Any]] = {}) -> None:
        self.config = config

    def __repr__(self):
        return f"EvalTemplate(name={self.eval_name})"

    def validate_config(self, config: Dict[str, Any]):
        for key, value in self.config_schema.items():
            if key not in config:
                raise MissingRequiredConfigForEvalTemplate(key, self.name)
            else:
                if key == "model" and config[key] not in model_list:
                    raise ValueError(
                        "Model not supported, please choose from the list of supported models"
                    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class ConversationCoherence(EvalTemplate):
    eval_name = "conversation_coherence"
    eval_id = "1"
    Input = ConversationMessages


class ConversationResolution(EvalTemplate):
    eval_name = "conversation_resolution"
    eval_id = "2"
    Input = ConversationMessages


# ---------------------------------------------------------------------------
# Content moderation & safety
# ---------------------------------------------------------------------------

class ContentModeration(EvalTemplate):
    eval_name = "content_moderation"
    eval_id = "4"
    Input = OutputOnly


class PII(EvalTemplate):
    eval_name = "pii"
    eval_id = "14"
    Input = OutputOnly


class Toxicity(EvalTemplate):
    eval_name = "toxicity"
    eval_id = "15"
    Input = OutputOnly


class Sexist(EvalTemplate):
    eval_name = "sexist"
    eval_id = "17"
    Input = OutputOnly


class PromptInjection(EvalTemplate):
    eval_name = "prompt_injection"
    eval_id = "18"
    Input = OutputOnly


class SafeForWorkText(EvalTemplate):
    eval_name = "safe_for_work_text"
    eval_id = "20"
    Input = OutputOnly


class DataPrivacyCompliance(EvalTemplate):
    eval_name = "data_privacy_compliance"
    eval_id = "22"
    Input = OutputOnly


class NoRacialBias(EvalTemplate):
    eval_name = "no_racial_bias"
    eval_id = "77"
    Input = OutputOnly


class NoGenderBias(EvalTemplate):
    eval_name = "no_gender_bias"
    eval_id = "78"
    Input = OutputOnly


class NoAgeBias(EvalTemplate):
    eval_name = "no_age_bias"
    eval_id = "79"
    Input = OutputOnly


class NoOpenAIReference(EvalTemplate):
    eval_name = "no_openai_reference"
    eval_id = "80"
    Input = OutputOnly


class NoApologies(EvalTemplate):
    eval_name = "no_apologies"
    eval_id = "81"
    Input = OutputOnly


class ContentSafety(EvalTemplate):
    eval_name = "content_safety_violation"
    eval_id = "93"
    Input = OutputOnly


class NoHarmfulTherapeuticGuidance(EvalTemplate):
    eval_name = "no_harmful_therapeutic_guidance"
    eval_id = "90"
    Input = OutputOnly


class ClinicallyInappropriateTone(EvalTemplate):
    eval_name = "clinically_inappropriate_tone"
    eval_id = "91"
    Input = OutputOnly


class IsHarmfulAdvice(EvalTemplate):
    eval_name = "is_harmful_advice"
    eval_id = "92"
    Input = OutputOnly


# ---------------------------------------------------------------------------
# RAG & context
# ---------------------------------------------------------------------------

class ContextAdherence(EvalTemplate):
    eval_name = "context_adherence"
    eval_id = "5"
    Input = OutputWithContext


class ContextRelevance(EvalTemplate):
    eval_name = "context_relevance"
    eval_id = "9"
    Input = OutputWithContext


class Completeness(EvalTemplate):
    eval_name = "completeness"
    eval_id = "10"
    Input = OutputWithContext


class ChunkAttribution(EvalTemplate):
    eval_name = "chunk_attribution"
    eval_id = "11"
    Input = OutputWithContext


class ChunkUtilization(EvalTemplate):
    eval_name = "chunk_utilization"
    eval_id = "12"
    Input = OutputWithContext


class Groundedness(EvalTemplate):
    eval_name = "groundedness"
    eval_id = "47"
    Input = OutputWithContext


class FactualAccuracy(EvalTemplate):
    eval_name = "factual_accuracy"
    eval_id = "66"
    Input = OutputWithContext


class DetectHallucinationMissingInfo(EvalTemplate):
    eval_name = "detect_hallucination_missing_info"
    eval_id = "89"
    Input = OutputWithContext


class IsFactuallyConsistent(EvalTemplate):
    eval_name = "is_factually_consistent"
    eval_id = "95"
    Input = OutputWithContext


# ---------------------------------------------------------------------------
# Text quality
# ---------------------------------------------------------------------------

class Tone(EvalTemplate):
    eval_name = "tone"
    eval_id = "16"
    Input = OutputOnly


class NotGibberishText(EvalTemplate):
    eval_name = "not_gibberish_text"
    eval_id = "19"
    Input = OutputOnly


class PromptAdherence(EvalTemplate):
    eval_name = "prompt_adherence"
    eval_id = "65"
    Input = OutputWithInput


class IsPolite(EvalTemplate):
    eval_name = "is_polite"
    eval_id = "82"
    Input = OutputOnly


class IsConcise(EvalTemplate):
    eval_name = "is_concise"
    eval_id = "83"
    Input = OutputOnly


class IsHelpful(EvalTemplate):
    eval_name = "is_helpful"
    eval_id = "84"
    Input = OutputWithInput


class IsInformalTone(EvalTemplate):
    eval_name = "is_informal_tone"
    eval_id = "97"
    Input = OutputOnly


class AnswerRefusal(EvalTemplate):
    eval_name = "answer_refusal"
    eval_id = "88"
    Input = OutputWithInput


class TaskCompletion(EvalTemplate):
    eval_name = "task_completion"
    eval_id = "99"
    Input = OutputWithInput


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

class IsJson(EvalTemplate):
    eval_name = "is_json"
    eval_id = "23"
    Input = OutputOnly


class OneLine(EvalTemplate):
    eval_name = "one_line"
    eval_id = "38"
    Input = OutputOnly


class ContainsValidLink(EvalTemplate):
    eval_name = "contains_valid_link"
    eval_id = "39"
    Input = OutputOnly


class IsEmail(EvalTemplate):
    eval_name = "is_email"
    eval_id = "40"
    Input = OutputOnly


class NoValidLinks(EvalTemplate):
    eval_name = "no_valid_links"
    eval_id = "42"
    Input = OutputOnly


class IsCode(EvalTemplate):
    eval_name = "is_code"
    eval_id = "85"
    Input = OutputOnly


class IsCSV(EvalTemplate):
    eval_name = "is_csv"
    eval_id = "86"
    Input = OutputOnly


# ---------------------------------------------------------------------------
# Comparison / ranking
# ---------------------------------------------------------------------------

class Ranking(EvalTemplate):
    eval_name = "eval_ranking"
    eval_id = "61"
    Input = OutputWithExpected


class SummaryQuality(EvalTemplate):
    eval_name = "summary_quality"
    eval_id = "64"
    Input = OutputWithContext


class IsGoodSummary(EvalTemplate):
    eval_name = "is_good_summary"
    eval_id = "94"
    Input = OutputWithContext


class FuzzyMatch(EvalTemplate):
    eval_name = "fuzzy_match"
    eval_id = "87"
    Input = OutputWithExpected


class BleuScore(EvalTemplate):
    eval_name = "bleu_score"
    eval_id = "101"
    Input = OutputWithExpected


# ---------------------------------------------------------------------------
# Translation & cultural
# ---------------------------------------------------------------------------

class TranslationAccuracy(EvalTemplate):
    eval_name = "translation_accuracy"
    eval_id = "67"
    Input = OutputWithExpected


class CulturalSensitivity(EvalTemplate):
    eval_name = "cultural_sensitivity"
    eval_id = "68"
    Input = OutputOnly


class BiasDetection(EvalTemplate):
    eval_name = "bias_detection"
    eval_id = "69"
    Input = OutputOnly


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------

class LLMFunctionCalling(EvalTemplate):
    eval_name = "llm_function_calling"
    eval_id = "72"
    Input = OutputWithExpected


class EvaluateFunctionCalling(EvalTemplate):
    eval_name = "evaluate_function_calling"
    eval_id = "98"
    Input = OutputWithExpected


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioTranscriptionEvaluator(EvalTemplate):
    eval_name = "audio_transcription"
    eval_id = "73"
    Input = AudioInput


class AudioQualityEvaluator(EvalTemplate):
    eval_name = "audio_quality"
    eval_id = "75"
    Input = AudioInput


# ---------------------------------------------------------------------------
# Hallucination & image
# ---------------------------------------------------------------------------

class CaptionHallucination(EvalTemplate):
    eval_name = "caption_hallucination"
    eval_id = "100"
    Input = ImageInput


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

class IsCompliant(EvalTemplate):
    eval_name = "is_compliant"
    eval_id = "96"
    Input = OutputWithInput
