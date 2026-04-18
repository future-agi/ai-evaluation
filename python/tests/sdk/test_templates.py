"""Comprehensive tests for fi.evals.templates module."""

import pytest
from fi.evals.templates import (
    EvalTemplate,
    ConversationCoherence,
    ConversationResolution,
    ContentModeration,
    ContextAdherence,
    ContextRelevance,
    Completeness,
    ChunkAttribution,
    ChunkUtilization,
    PII,
    Toxicity,
    Tone,
    Sexist,
    PromptInjection,
    PromptAdherence,
    DataPrivacyCompliance,
    IsJson,
    OneLine,
    ContainsValidLink,
    IsEmail,
    Groundedness,
    Ranking,
    SummaryQuality,
    FactualAccuracy,
    TranslationAccuracy,
    CulturalSensitivity,
    BiasDetection,
    LLMFunctionCalling,
    AudioTranscriptionEvaluator,
    AudioQualityEvaluator,
    NoRacialBias,
    NoGenderBias,
    NoAgeBias,
    NoOpenAIReference,
    NoApologies,
    IsPolite,
    IsConcise,
    IsHelpful,
    FuzzyMatch,
    AnswerRefusal,
    DetectHallucinationMissingInfo,
    NoHarmfulTherapeuticGuidance,
    ClinicallyInappropriateTone,
    IsHarmfulAdvice,
    ContentSafety,
    IsGoodSummary,
    IsFactuallyConsistent,
    IsCompliant,
    IsInformalTone,
    EvaluateFunctionCalling,
    TaskCompletion,
    CaptionHallucination,
    BleuScore,
)


class TestEvalTemplateBase:
    """Tests for base EvalTemplate class."""

    def test_eval_template_init(self):
        """Test EvalTemplate initialization."""
        template = EvalTemplate()
        assert template.config == {}

    def test_eval_template_with_config(self):
        """Test EvalTemplate with custom config."""
        config = {"model": "gpt-4", "threshold": 0.8}
        template = EvalTemplate(config=config)
        assert template.config == config

    def test_eval_template_repr(self):
        """Test EvalTemplate string representation."""
        # Note: EvalTemplate base class doesn't have eval_name, so we test a subclass
        template = Groundedness()
        repr_str = repr(template)
        assert "EvalTemplate" in repr_str
        assert "groundedness" in repr_str


class TestConversationTemplates:
    """Tests for conversation-related templates."""

    def test_conversation_coherence(self):
        """Test ConversationCoherence template."""
        template = ConversationCoherence()
        assert template.eval_name == "conversation_coherence"
        assert template.eval_id == "1"

    def test_conversation_resolution(self):
        """Test ConversationResolution template."""
        template = ConversationResolution()
        assert template.eval_name == "conversation_resolution"
        assert template.eval_id == "2"


class TestSafetyTemplates:
    """Tests for safety-related templates."""

    def test_content_moderation(self):
        """Test ContentModeration template."""
        template = ContentModeration()
        assert template.eval_name == "content_moderation"
        assert template.eval_id == "4"

    def test_pii(self):
        """Test PII template."""
        template = PII()
        assert template.eval_name == "pii"
        assert template.eval_id == "14"

    def test_toxicity(self):
        """Test Toxicity template."""
        template = Toxicity()
        assert template.eval_name == "toxicity"
        assert template.eval_id == "15"

    def test_sexist(self):
        """Test Sexist template."""
        template = Sexist()
        assert template.eval_name == "sexist"
        assert template.eval_id == "17"

    def test_prompt_injection(self):
        """Test PromptInjection template."""
        template = PromptInjection()
        assert template.eval_name == "prompt_injection"
        assert template.eval_id == "18"

    def test_data_privacy_compliance(self):
        """Test DataPrivacyCompliance template."""
        template = DataPrivacyCompliance()
        assert template.eval_name == "data_privacy_compliance"
        assert template.eval_id == "22"

    def test_content_safety(self):
        """Test ContentSafety template."""
        template = ContentSafety()
        assert template.eval_name == "content_safety_violation"
        assert template.eval_id == "93"


class TestRAGTemplates:
    """Tests for RAG-related templates."""

    def test_context_adherence(self):
        """Test ContextAdherence template."""
        template = ContextAdherence()
        assert template.eval_name == "context_adherence"
        assert template.eval_id == "5"

    def test_context_relevance(self):
        """Test ContextRelevance template."""
        template = ContextRelevance()
        assert template.eval_name == "context_relevance"
        assert template.eval_id == "9"

    def test_completeness(self):
        """Test Completeness template."""
        template = Completeness()
        assert template.eval_name == "completeness"
        assert template.eval_id == "10"

    def test_chunk_attribution(self):
        """Test ChunkAttribution template."""
        template = ChunkAttribution()
        assert template.eval_name == "chunk_attribution"
        assert template.eval_id == "11"

    def test_chunk_utilization(self):
        """Test ChunkUtilization template."""
        template = ChunkUtilization()
        assert template.eval_name == "chunk_utilization"
        assert template.eval_id == "12"

    def test_groundedness(self):
        """Test Groundedness template."""
        template = Groundedness()
        assert template.eval_name == "groundedness"
        assert template.eval_id == "47"


class TestBiasTemplates:
    """Tests for bias detection templates."""

    def test_bias_detection(self):
        """Test BiasDetection template."""
        template = BiasDetection()
        assert template.eval_name == "bias_detection"
        assert template.eval_id == "69"

    def test_no_racial_bias(self):
        """Test NoRacialBias template."""
        template = NoRacialBias()
        assert template.eval_name == "no_racial_bias"
        assert template.eval_id == "77"

    def test_no_gender_bias(self):
        """Test NoGenderBias template."""
        template = NoGenderBias()
        assert template.eval_name == "no_gender_bias"
        assert template.eval_id == "78"

    def test_no_age_bias(self):
        """Test NoAgeBias template."""
        template = NoAgeBias()
        assert template.eval_name == "no_age_bias"
        assert template.eval_id == "79"

    def test_cultural_sensitivity(self):
        """Test CulturalSensitivity template."""
        template = CulturalSensitivity()
        assert template.eval_name == "cultural_sensitivity"
        assert template.eval_id == "68"


class TestToneTemplates:
    """Tests for tone-related templates."""

    def test_tone(self):
        """Test Tone template."""
        template = Tone()
        assert template.eval_name == "tone"
        assert template.eval_id == "16"

    def test_is_polite(self):
        """Test IsPolite template."""
        template = IsPolite()
        assert template.eval_name == "is_polite"
        assert template.eval_id == "82"

    def test_is_concise(self):
        """Test IsConcise template."""
        template = IsConcise()
        assert template.eval_name == "is_concise"
        assert template.eval_id == "83"

    def test_no_apologies(self):
        """Test NoApologies template."""
        template = NoApologies()
        assert template.eval_name == "no_apologies"
        assert template.eval_id == "81"

    def test_no_openai_reference(self):
        """Test NoOpenAIReference template."""
        template = NoOpenAIReference()
        assert template.eval_name == "no_openai_reference"
        assert template.eval_id == "80"

    def test_is_informal_tone(self):
        """Test IsInformalTone template."""
        template = IsInformalTone()
        assert template.eval_name == "is_informal_tone"
        assert template.eval_id == "97"


class TestFormatTemplates:
    """Tests for format validation templates."""

    def test_is_json(self):
        """Test IsJson template."""
        template = IsJson()
        assert template.eval_name == "is_json"
        assert template.eval_id == "23"

    def test_one_line(self):
        """Test OneLine template."""
        template = OneLine()
        assert template.eval_name == "one_line"
        assert template.eval_id == "38"

    def test_contains_valid_link(self):
        """Test ContainsValidLink template."""
        template = ContainsValidLink()
        assert template.eval_name == "contains_valid_link"
        assert template.eval_id == "39"

    def test_is_email(self):
        """Test IsEmail template."""
        template = IsEmail()
        assert template.eval_name == "is_email"
        assert template.eval_id == "40"


class TestQualityTemplates:
    """Tests for quality-related templates."""

    def test_factual_accuracy(self):
        """Test FactualAccuracy template."""
        template = FactualAccuracy()
        assert template.eval_name == "factual_accuracy"
        assert template.eval_id == "66"

    def test_is_helpful(self):
        """Test IsHelpful template."""
        template = IsHelpful()
        assert template.eval_name == "is_helpful"
        assert template.eval_id == "84"

    def test_summary_quality(self):
        """Test SummaryQuality template."""
        template = SummaryQuality()
        assert template.eval_name == "summary_quality"
        assert template.eval_id == "64"

    def test_is_good_summary(self):
        """Test IsGoodSummary template."""
        template = IsGoodSummary()
        assert template.eval_name == "is_good_summary"
        assert template.eval_id == "94"

    def test_is_factually_consistent(self):
        """Test IsFactuallyConsistent template."""
        template = IsFactuallyConsistent()
        assert template.eval_name == "is_factually_consistent"
        assert template.eval_id == "95"

    def test_prompt_adherence(self):
        """Test PromptAdherence template."""
        template = PromptAdherence()
        assert template.eval_name == "prompt_adherence"
        assert template.eval_id == "65"


class TestHallucinationTemplates:
    """Tests for hallucination detection templates."""

    def test_detect_hallucination_missing_info(self):
        """Test DetectHallucinationMissingInfo template."""
        template = DetectHallucinationMissingInfo()
        assert template.eval_name == "detect_hallucination_missing_info"
        assert template.eval_id == "89"

    def test_caption_hallucination(self):
        """Test CaptionHallucination template."""
        template = CaptionHallucination()
        assert template.eval_name == "caption_hallucination"
        assert template.eval_id == "100"


class TestTranslationTemplates:
    """Tests for translation-related templates."""

    def test_translation_accuracy(self):
        """Test TranslationAccuracy template."""
        template = TranslationAccuracy()
        assert template.eval_name == "translation_accuracy"
        assert template.eval_id == "67"


class TestFunctionCallingTemplates:
    """Tests for function calling templates."""

    def test_llm_function_calling(self):
        """Test LLMFunctionCalling template."""
        template = LLMFunctionCalling()
        assert template.eval_name == "llm_function_calling"
        assert template.eval_id == "72"

    def test_evaluate_function_calling(self):
        """Test EvaluateFunctionCalling template."""
        template = EvaluateFunctionCalling()
        assert template.eval_name == "evaluate_function_calling"
        assert template.eval_id == "98"


class TestAudioTemplates:
    """Tests for audio-related templates."""

    def test_audio_transcription_evaluator(self):
        """Test AudioTranscriptionEvaluator template."""
        template = AudioTranscriptionEvaluator()
        assert template.eval_name == "audio_transcription"
        assert template.eval_id == "73"

    def test_audio_quality_evaluator(self):
        """Test AudioQualityEvaluator template."""
        template = AudioQualityEvaluator()
        assert template.eval_name == "audio_quality"
        assert template.eval_id == "75"


class TestMedicalTemplates:
    """Tests for medical/clinical templates."""

    def test_no_harmful_therapeutic_guidance(self):
        """Test NoHarmfulTherapeuticGuidance template."""
        template = NoHarmfulTherapeuticGuidance()
        assert template.eval_name == "no_harmful_therapeutic_guidance"
        assert template.eval_id == "90"

    def test_clinically_inappropriate_tone(self):
        """Test ClinicallyInappropriateTone template."""
        template = ClinicallyInappropriateTone()
        assert template.eval_name == "clinically_inappropriate_tone"
        assert template.eval_id == "91"

    def test_is_harmful_advice(self):
        """Test IsHarmfulAdvice template."""
        template = IsHarmfulAdvice()
        assert template.eval_name == "is_harmful_advice"
        assert template.eval_id == "92"


class TestOtherTemplates:
    """Tests for other templates."""

    def test_ranking(self):
        """Test Ranking template."""
        template = Ranking()
        assert template.eval_name == "eval_ranking"
        assert template.eval_id == "61"

    def test_fuzzy_match(self):
        """Test FuzzyMatch template."""
        template = FuzzyMatch()
        assert template.eval_name == "fuzzy_match"
        assert template.eval_id == "87"

    def test_answer_refusal(self):
        """Test AnswerRefusal template."""
        template = AnswerRefusal()
        assert template.eval_name == "answer_refusal"
        assert template.eval_id == "88"

    def test_is_compliant(self):
        """Test IsCompliant template."""
        template = IsCompliant()
        assert template.eval_name == "is_compliant"
        assert template.eval_id == "96"

    def test_task_completion(self):
        """Test TaskCompletion template."""
        template = TaskCompletion()
        assert template.eval_name == "task_completion"
        assert template.eval_id == "99"

    def test_bleu_score(self):
        """Test BleuScore template."""
        template = BleuScore()
        assert template.eval_name == "bleu_score"
        assert template.eval_id == "101"


class TestTemplateWithConfig:
    """Tests for templates with custom configurations."""

    def test_template_config_passed(self):
        """Test that config is properly passed to templates."""
        config = {"threshold": 0.9, "model": "custom-model"}
        template = Groundedness(config=config)
        assert template.config == config
        assert template.config["threshold"] == 0.9

    def test_template_empty_config(self):
        """Test template with empty config."""
        template = Toxicity(config={})
        assert template.config == {}

    def test_template_none_config(self):
        """Test template with None config (should use default)."""
        template = PII(config=None)
        # None should be treated as empty dict in __init__
        assert template.config is None or template.config == {}


class TestAllTemplatesExist:
    """Verify all documented templates are importable and have correct structure."""

    @pytest.fixture
    def all_template_classes(self):
        """Return list of all template classes."""
        return [
            ConversationCoherence,
            ConversationResolution,
            ContentModeration,
            ContextAdherence,
            ContextRelevance,
            Completeness,
            ChunkAttribution,
            ChunkUtilization,
            PII,
            Toxicity,
            Tone,
            Sexist,
            PromptInjection,
            PromptAdherence,
            DataPrivacyCompliance,
            IsJson,
            OneLine,
            ContainsValidLink,
            IsEmail,
            Groundedness,
            Ranking,
            SummaryQuality,
            FactualAccuracy,
            TranslationAccuracy,
            CulturalSensitivity,
            BiasDetection,
            LLMFunctionCalling,
            AudioTranscriptionEvaluator,
            AudioQualityEvaluator,
            NoRacialBias,
            NoGenderBias,
            NoAgeBias,
            NoOpenAIReference,
            NoApologies,
            IsPolite,
            IsConcise,
            IsHelpful,
            FuzzyMatch,
            AnswerRefusal,
            DetectHallucinationMissingInfo,
            NoHarmfulTherapeuticGuidance,
            ClinicallyInappropriateTone,
            IsHarmfulAdvice,
            ContentSafety,
            IsGoodSummary,
            IsFactuallyConsistent,
            IsCompliant,
            IsInformalTone,
            EvaluateFunctionCalling,
            TaskCompletion,
            CaptionHallucination,
            BleuScore,
        ]

    def test_all_templates_have_eval_name(self, all_template_classes):
        """Test all templates have eval_name attribute."""
        for template_class in all_template_classes:
            assert hasattr(template_class, 'eval_name'), f"{template_class.__name__} missing eval_name"
            assert template_class.eval_name, f"{template_class.__name__} has empty eval_name"

    def test_all_templates_have_eval_id(self, all_template_classes):
        """Test all templates have eval_id attribute."""
        for template_class in all_template_classes:
            assert hasattr(template_class, 'eval_id'), f"{template_class.__name__} missing eval_id"
            assert template_class.eval_id, f"{template_class.__name__} has empty eval_id"

    def test_all_templates_inherit_from_eval_template(self, all_template_classes):
        """Test all templates inherit from EvalTemplate."""
        for template_class in all_template_classes:
            assert issubclass(template_class, EvalTemplate), \
                f"{template_class.__name__} does not inherit from EvalTemplate"

    def test_all_templates_instantiable(self, all_template_classes):
        """Test all templates can be instantiated."""
        for template_class in all_template_classes:
            try:
                instance = template_class()
                assert instance is not None
            except Exception as e:
                pytest.fail(f"Failed to instantiate {template_class.__name__}: {e}")

    def test_unique_eval_ids(self, all_template_classes):
        """Test all templates have unique eval_ids."""
        eval_ids = [cls.eval_id for cls in all_template_classes]
        assert len(eval_ids) == len(set(eval_ids)), "Duplicate eval_ids found"

    def test_unique_eval_names(self, all_template_classes):
        """Test all templates have unique eval_names."""
        eval_names = [cls.eval_name for cls in all_template_classes]
        assert len(eval_names) == len(set(eval_names)), "Duplicate eval_names found"

    def test_template_count(self, all_template_classes):
        """Test we have the expected number of templates (57)."""
        assert len(all_template_classes) == 57, \
            f"Expected 57 templates, found {len(all_template_classes)}"
