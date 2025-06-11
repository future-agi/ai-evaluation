from typing import Any, Dict, List, Optional

from fi.utils.errors import MissingRequiredKey, MissingRequiredConfigForEvalTemplate


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

    def __init__(self, config: Optional[Dict[str, Any]] = {}) -> None:
        self.config = config

    def __repr__(self):
        """
        Get the string representation of the evaluation template
        """
        return f"EvalTemplate(name={self.eval_name})"
    
    def validate_config(self, config: Dict[str, Any]):
        """
        Validate the config for the evaluation template
        """
        for key, value in self.config_schema.items():
            if key not in config:
                raise MissingRequiredConfigForEvalTemplate(key, self.name)
            else:
                if key == "model" and config[key] not in model_list:
                    raise ValueError(
                        "Model not supported, please choose from the list of supported models"
                    )



    # def validate_input(self, inputs: List[LLMTestCase]):
    #     """
    #     Validate the input against the evaluation template requirements

    #     Args:
    #         inputs: [
    #             LLMTestCase(QUERY='Who is Prime Minister of India?', RESPONSE='Narendra Modi')
    #         ]

    #     Returns:
    #         bool: True if the input is valid, False otherwise
    #     """

    #     for key in self.required_keys:
    #         for test_case in inputs:
    #             if getattr(test_case, key) is None:
    #                 raise MissingRequiredKey("test case input", key)

    #     return True


class ConversationCoherence(EvalTemplate):
    eval_name = "conversation_coherence"
    eval_id = "1"


class ConversationResolution(EvalTemplate):
    eval_name = "conversation_resolution"
    eval_id = "2"

class ContentModeration(EvalTemplate):
    eval_name = "content_moderation"
    eval_id = "4"

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

class PII(EvalTemplate):
    eval_name = "pii"
    eval_id = "14"


class Toxicity(EvalTemplate):
    eval_name = "toxicity"
    eval_id = "15"

class Tone(EvalTemplate):
    eval_name = "tone"
    eval_id = "16"

class Sexist(EvalTemplate):
    eval_name = "sexist"
    eval_id = "17"

class PromptInjection(EvalTemplate):
    eval_name = "prompt_injection"
    eval_id = "18"

class NotGibberishText(EvalTemplate):
    eval_name = "not_gibberish_text"
    eval_id = "19"

class SafeForWorkText(EvalTemplate):
    eval_name = "safe_for_work_text"
    eval_id = "20"

class PromptAdherence(EvalTemplate):
    eval_name = "prompt_adherence"
    eval_id = "65"

class DataPrivacyCompliance(EvalTemplate):
    eval_name = "data_privacy_compliance"
    eval_id = "22"


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

class NoValidLinks(EvalTemplate):
    eval_name = "no_valid_links"
    eval_id = "42"

class Groundedness(EvalTemplate):
    eval_name = "groundedness"
    eval_id = "47"


class Ranking(EvalTemplate):
    eval_name = "eval_ranking"
    eval_id = "61"



class SummaryQuality(EvalTemplate):
    eval_name = "summary_quality"
    eval_id = "64"


class FactualAccuracy(EvalTemplate):
    eval_name = "factual_accuracy"
    eval_id = "66"

class TranslationAccuracy(EvalTemplate):
    eval_name = "translation_accuracy"
    eval_id = "67"

class CulturalSensitivity(EvalTemplate):
    eval_name = "cultural_sensitivity"
    eval_id = "68"

class BiasDetection(EvalTemplate):
    eval_name = "bias_detection"
    eval_id = "69"

class LLMFunctionCalling(EvalTemplate):
    eval_name = "llm_function_calling"
    eval_id = "72"

class AudioTranscriptionEvaluator(EvalTemplate):
    eval_name = "audio_transcription"
    eval_id = "73"


class AudioQualityEvaluator(EvalTemplate):
    eval_name = "audio_quality"
    eval_id = "75"

class NoRacialBias(EvalTemplate):
    eval_name = "no_racial_bias"
    eval_id = "77"

class NoGenderBias(EvalTemplate):
    eval_name = "no_gender_bias"
    eval_id = "78"


class NoAgeBias(EvalTemplate):
    eval_name = "no_age_bias"
    eval_id = "79"

class NoOpenAIReference(EvalTemplate):
    eval_name = "no_openai_reference"
    eval_id = "80"

class NoApologies(EvalTemplate):
    eval_name = "no_apologies"
    eval_id = "81"

class IsPolite(EvalTemplate):
    eval_name = "is_polite"
    eval_id = "82"

class IsConcise(EvalTemplate):
    eval_name = "is_concise"
    eval_id = "83"

class IsHelpful(EvalTemplate):
    eval_name = "is_helpful"
    eval_id = "84"

class IsCode(EvalTemplate):
    eval_name = "is_code"
    eval_id = "85"

class IsCSV(EvalTemplate):
    eval_name = "is_csv"
    eval_id = "86"

class FuzzyMatch(EvalTemplate):
    eval_name = "fuzzy_match"
    eval_id = "87"

class AnswerRefusal(EvalTemplate):
    eval_name = "answer_refusal"
    eval_id = "88"

class DetectHallucinationMissingInfo(EvalTemplate):
    eval_name = "detect_hallucination_missing_info"
    eval_id = "89"

class NoHarmfulTherapeuticGuidance(EvalTemplate):
    eval_name = "no_harmful_therapeutic_guidance"
    eval_id = "90"

class ClinicallyInappropriateTone(EvalTemplate):
    eval_name = "clinically_inappropriate_tone"
    eval_id = "91"

class IsHarmfulAdvice(EvalTemplate):
    eval_name = "is_harmful_advice"
    eval_id = "92"

class ContentSafety(EvalTemplate):
    eval_name = "content_safety_violation"
    eval_id = "93"

class IsGoodSummary(EvalTemplate):
    eval_name = "is_good_summary"
    eval_id = "94"

class IsFactuallyConsistent(EvalTemplate):
    eval_name = "is_factually_consistent"
    eval_id = "95"

class IsCompliant(EvalTemplate):
    eval_name = "is_compliant"
    eval_id = "96"

class IsInformalTone(EvalTemplate):
    eval_name = "is_informal_tone"
    eval_id = "97"

class EvaluateFunctionCalling(EvalTemplate):
    eval_name = "evaluate_function_calling"
    eval_id = "98"

class TaskCompletion(EvalTemplate):
    eval_name = "task_completion"
    eval_id = "99"

class CaptionHallucination(EvalTemplate):
    eval_name = "caption_hallucination"
    eval_id = "100"

class BleuScore(EvalTemplate):
    eval_name = "bleu_score"
    eval_id = "101"