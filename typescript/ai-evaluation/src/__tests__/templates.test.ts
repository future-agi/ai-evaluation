import { Templates } from '../templates';
import { EvalTemplate } from '../types';

describe('Templates', () => {
    describe('Template Structure', () => {
        it('should have all templates as objects with eval_id and eval_name', () => {
            Object.entries(Templates).forEach(([key, template]) => {
                expect(template).toHaveProperty('eval_id');
                expect(template).toHaveProperty('eval_name');
                expect(typeof template.eval_id).toBe('string');
                expect(typeof template.eval_name).toBe('string');
            });
        });

        it('should have unique eval_ids across all templates', () => {
            const evalIds = Object.values(Templates).map(t => t.eval_id);
            const uniqueIds = new Set(evalIds);
            expect(uniqueIds.size).toBe(evalIds.length);
        });

        it('should have unique eval_names across all templates', () => {
            const evalNames = Object.values(Templates).map(t => t.eval_name);
            const uniqueNames = new Set(evalNames);
            expect(uniqueNames.size).toBe(evalNames.length);
        });
    });

    describe('Quality & Accuracy Templates', () => {
        it('should have FactualAccuracy template', () => {
            expect(Templates.FactualAccuracy).toBeDefined();
            expect(Templates.FactualAccuracy.eval_id).toBe('66');
            expect(Templates.FactualAccuracy.eval_name).toBe('factual_accuracy');
        });

        it('should have Groundedness template', () => {
            expect(Templates.Groundedness).toBeDefined();
            expect(Templates.Groundedness.eval_id).toBe('47');
            expect(Templates.Groundedness.eval_name).toBe('groundedness');
        });

        it('should have ContextAdherence template', () => {
            expect(Templates.ContextAdherence).toBeDefined();
            expect(Templates.ContextAdherence.eval_id).toBe('5');
            expect(Templates.ContextAdherence.eval_name).toBe('context_adherence');
        });

        it('should have ContextRelevance template', () => {
            expect(Templates.ContextRelevance).toBeDefined();
            expect(Templates.ContextRelevance.eval_id).toBe('9');
            expect(Templates.ContextRelevance.eval_name).toBe('context_relevance');
        });

        it('should have Completeness template', () => {
            expect(Templates.Completeness).toBeDefined();
            expect(Templates.Completeness.eval_id).toBe('10');
            expect(Templates.Completeness.eval_name).toBe('completeness');
        });
    });

    describe('Safety & Content Templates', () => {
        it('should have Toxicity template', () => {
            expect(Templates.Toxicity).toBeDefined();
            expect(Templates.Toxicity.eval_id).toBe('15');
            expect(Templates.Toxicity.eval_name).toBe('toxicity');
        });

        it('should have PII template', () => {
            expect(Templates.PII).toBeDefined();
            expect(Templates.PII.eval_id).toBe('14');
            expect(Templates.PII.eval_name).toBe('pii');
        });

        it('should have ContentSafety template', () => {
            expect(Templates.ContentSafety).toBeDefined();
            expect(Templates.ContentSafety.eval_id).toBe('93');
            expect(Templates.ContentSafety.eval_name).toBe('content_safety_violation');
        });

        it('should have ContentModeration template', () => {
            expect(Templates.ContentModeration).toBeDefined();
            expect(Templates.ContentModeration.eval_id).toBe('4');
            expect(Templates.ContentModeration.eval_name).toBe('content_moderation');
        });

        it('should have PromptInjection template', () => {
            expect(Templates.PromptInjection).toBeDefined();
            expect(Templates.PromptInjection.eval_id).toBe('18');
            expect(Templates.PromptInjection.eval_name).toBe('prompt_injection');
        });

        it('should have DataPrivacyCompliance template', () => {
            expect(Templates.DataPrivacyCompliance).toBeDefined();
            expect(Templates.DataPrivacyCompliance.eval_id).toBe('22');
            expect(Templates.DataPrivacyCompliance.eval_name).toBe('data_privacy_compliance');
        });
    });

    describe('Bias & Fairness Templates', () => {
        it('should have BiasDetection template', () => {
            expect(Templates.BiasDetection).toBeDefined();
            expect(Templates.BiasDetection.eval_id).toBe('69');
            expect(Templates.BiasDetection.eval_name).toBe('bias_detection');
        });

        it('should have NoRacialBias template', () => {
            expect(Templates.NoRacialBias).toBeDefined();
            expect(Templates.NoRacialBias.eval_id).toBe('77');
            expect(Templates.NoRacialBias.eval_name).toBe('no_racial_bias');
        });

        it('should have NoGenderBias template', () => {
            expect(Templates.NoGenderBias).toBeDefined();
            expect(Templates.NoGenderBias.eval_id).toBe('78');
            expect(Templates.NoGenderBias.eval_name).toBe('no_gender_bias');
        });

        it('should have NoAgeBias template', () => {
            expect(Templates.NoAgeBias).toBeDefined();
            expect(Templates.NoAgeBias.eval_id).toBe('79');
            expect(Templates.NoAgeBias.eval_name).toBe('no_age_bias');
        });

        it('should have CulturalSensitivity template', () => {
            expect(Templates.CulturalSensitivity).toBeDefined();
            expect(Templates.CulturalSensitivity.eval_id).toBe('68');
            expect(Templates.CulturalSensitivity.eval_name).toBe('cultural_sensitivity');
        });
    });

    describe('Tone & Style Templates', () => {
        it('should have Tone template', () => {
            expect(Templates.Tone).toBeDefined();
            expect(Templates.Tone.eval_id).toBe('16');
            expect(Templates.Tone.eval_name).toBe('tone');
        });

        it('should have IsPolite template', () => {
            expect(Templates.IsPolite).toBeDefined();
            expect(Templates.IsPolite.eval_id).toBe('82');
            expect(Templates.IsPolite.eval_name).toBe('is_polite');
        });

        it('should have IsConcise template', () => {
            expect(Templates.IsConcise).toBeDefined();
            expect(Templates.IsConcise.eval_id).toBe('83');
            expect(Templates.IsConcise.eval_name).toBe('is_concise');
        });

        it('should have IsInformalTone template', () => {
            expect(Templates.IsInformalTone).toBeDefined();
            expect(Templates.IsInformalTone.eval_id).toBe('97');
            expect(Templates.IsInformalTone.eval_name).toBe('is_informal_tone');
        });

        it('should have NoApologies template', () => {
            expect(Templates.NoApologies).toBeDefined();
            expect(Templates.NoApologies.eval_id).toBe('81');
            expect(Templates.NoApologies.eval_name).toBe('no_apologies');
        });

        it('should have NoOpenAIReference template', () => {
            expect(Templates.NoOpenAIReference).toBeDefined();
            expect(Templates.NoOpenAIReference.eval_id).toBe('80');
            expect(Templates.NoOpenAIReference.eval_name).toBe('no_openai_reference');
        });
    });

    describe('Format & Structure Templates', () => {
        it('should have IsJson template', () => {
            expect(Templates.IsJson).toBeDefined();
            expect(Templates.IsJson.eval_id).toBe('23');
            expect(Templates.IsJson.eval_name).toBe('is_json');
        });

        it('should have IsCSV template', () => {
            expect(Templates.IsCSV).toBeDefined();
            expect(Templates.IsCSV.eval_id).toBe('86');
            expect(Templates.IsCSV.eval_name).toBe('is_csv');
        });

        it('should have IsCode template', () => {
            expect(Templates.IsCode).toBeDefined();
            expect(Templates.IsCode.eval_id).toBe('85');
            expect(Templates.IsCode.eval_name).toBe('is_code');
        });

        it('should have IsEmail template', () => {
            expect(Templates.IsEmail).toBeDefined();
            expect(Templates.IsEmail.eval_id).toBe('40');
            expect(Templates.IsEmail.eval_name).toBe('is_email');
        });

        it('should have OneLine template', () => {
            expect(Templates.OneLine).toBeDefined();
            expect(Templates.OneLine.eval_id).toBe('38');
            expect(Templates.OneLine.eval_name).toBe('one_line');
        });

        it('should have ContainsValidLink template', () => {
            expect(Templates.ContainsValidLink).toBeDefined();
            expect(Templates.ContainsValidLink.eval_id).toBe('39');
            expect(Templates.ContainsValidLink.eval_name).toBe('contains_valid_link');
        });

        it('should have NoValidLinks template', () => {
            expect(Templates.NoValidLinks).toBeDefined();
            expect(Templates.NoValidLinks.eval_id).toBe('42');
            expect(Templates.NoValidLinks.eval_name).toBe('no_valid_links');
        });
    });

    describe('Conversation & Interaction Templates', () => {
        it('should have ConversationCoherence template', () => {
            expect(Templates.ConversationCoherence).toBeDefined();
            expect(Templates.ConversationCoherence.eval_id).toBe('1');
            expect(Templates.ConversationCoherence.eval_name).toBe('conversation_coherence');
        });

        it('should have ConversationResolution template', () => {
            expect(Templates.ConversationResolution).toBeDefined();
            expect(Templates.ConversationResolution.eval_id).toBe('2');
            expect(Templates.ConversationResolution.eval_name).toBe('conversation_resolution');
        });

        it('should have TaskCompletion template', () => {
            expect(Templates.TaskCompletion).toBeDefined();
            expect(Templates.TaskCompletion.eval_id).toBe('99');
            expect(Templates.TaskCompletion.eval_name).toBe('task_completion');
        });

        it('should have AnswerRefusal template', () => {
            expect(Templates.AnswerRefusal).toBeDefined();
            expect(Templates.AnswerRefusal.eval_id).toBe('88');
            expect(Templates.AnswerRefusal.eval_name).toBe('answer_refusal');
        });
    });

    describe('Specialized Templates', () => {
        it('should have TranslationAccuracy template', () => {
            expect(Templates.TranslationAccuracy).toBeDefined();
            expect(Templates.TranslationAccuracy.eval_id).toBe('67');
            expect(Templates.TranslationAccuracy.eval_name).toBe('translation_accuracy');
        });

        it('should have SummaryQuality template', () => {
            expect(Templates.SummaryQuality).toBeDefined();
            expect(Templates.SummaryQuality.eval_id).toBe('64');
            expect(Templates.SummaryQuality.eval_name).toBe('summary_quality');
        });

        it('should have Ranking template', () => {
            expect(Templates.Ranking).toBeDefined();
            expect(Templates.Ranking.eval_id).toBe('61');
            expect(Templates.Ranking.eval_name).toBe('eval_ranking');
        });

        it('should have PromptAdherence template', () => {
            expect(Templates.PromptAdherence).toBeDefined();
            expect(Templates.PromptAdherence.eval_id).toBe('65');
            expect(Templates.PromptAdherence.eval_name).toBe('prompt_adherence');
        });

        it('should have LLMFunctionCalling template', () => {
            expect(Templates.LLMFunctionCalling).toBeDefined();
            expect(Templates.LLMFunctionCalling.eval_id).toBe('72');
            expect(Templates.LLMFunctionCalling.eval_name).toBe('llm_function_calling');
        });

        it('should have AudioTranscriptionEvaluator template', () => {
            expect(Templates.AudioTranscriptionEvaluator).toBeDefined();
            expect(Templates.AudioTranscriptionEvaluator.eval_id).toBe('73');
            expect(Templates.AudioTranscriptionEvaluator.eval_name).toBe('audio_transcription');
        });

        it('should have AudioQualityEvaluator template', () => {
            expect(Templates.AudioQualityEvaluator).toBeDefined();
            expect(Templates.AudioQualityEvaluator.eval_id).toBe('75');
            expect(Templates.AudioQualityEvaluator.eval_name).toBe('audio_quality');
        });
    });

    describe('Medical/Clinical Templates', () => {
        it('should have NoHarmfulTherapeuticGuidance template', () => {
            expect(Templates.NoHarmfulTherapeuticGuidance).toBeDefined();
            expect(Templates.NoHarmfulTherapeuticGuidance.eval_id).toBe('90');
            expect(Templates.NoHarmfulTherapeuticGuidance.eval_name).toBe('no_harmful_therapeutic_guidance');
        });

        it('should have ClinicallyInappropriateTone template', () => {
            expect(Templates.ClinicallyInappropriateTone).toBeDefined();
            expect(Templates.ClinicallyInappropriateTone.eval_id).toBe('91');
            expect(Templates.ClinicallyInappropriateTone.eval_name).toBe('clinically_inappropriate_tone');
        });

        it('should have IsHarmfulAdvice template', () => {
            expect(Templates.IsHarmfulAdvice).toBeDefined();
            expect(Templates.IsHarmfulAdvice.eval_id).toBe('92');
            expect(Templates.IsHarmfulAdvice.eval_name).toBe('is_harmful_advice');
        });
    });

    describe('Hallucination Templates', () => {
        it('should have DetectHallucinationMissingInfo template', () => {
            expect(Templates.DetectHallucinationMissingInfo).toBeDefined();
            expect(Templates.DetectHallucinationMissingInfo.eval_id).toBe('89');
            expect(Templates.DetectHallucinationMissingInfo.eval_name).toBe('detect_hallucination_missing_info');
        });

        it('should have CaptionHallucination template', () => {
            expect(Templates.CaptionHallucination).toBeDefined();
            expect(Templates.CaptionHallucination.eval_id).toBe('100');
            expect(Templates.CaptionHallucination.eval_name).toBe('caption_hallucination');
        });
    });

    describe('Other Templates', () => {
        it('should have Sexist template', () => {
            expect(Templates.Sexist).toBeDefined();
            expect(Templates.Sexist.eval_id).toBe('17');
            expect(Templates.Sexist.eval_name).toBe('sexist');
        });

        it('should have NotGibberishText template', () => {
            expect(Templates.NotGibberishText).toBeDefined();
            expect(Templates.NotGibberishText.eval_id).toBe('19');
            expect(Templates.NotGibberishText.eval_name).toBe('not_gibberish_text');
        });

        it('should have SafeForWorkText template', () => {
            expect(Templates.SafeForWorkText).toBeDefined();
            expect(Templates.SafeForWorkText.eval_id).toBe('20');
            expect(Templates.SafeForWorkText.eval_name).toBe('safe_for_work_text');
        });

        it('should have FuzzyMatch template', () => {
            expect(Templates.FuzzyMatch).toBeDefined();
            expect(Templates.FuzzyMatch.eval_id).toBe('87');
            expect(Templates.FuzzyMatch.eval_name).toBe('fuzzy_match');
        });

        it('should have IsGoodSummary template', () => {
            expect(Templates.IsGoodSummary).toBeDefined();
            expect(Templates.IsGoodSummary.eval_id).toBe('94');
            expect(Templates.IsGoodSummary.eval_name).toBe('is_good_summary');
        });

        it('should have IsFactuallyConsistent template', () => {
            expect(Templates.IsFactuallyConsistent).toBeDefined();
            expect(Templates.IsFactuallyConsistent.eval_id).toBe('95');
            expect(Templates.IsFactuallyConsistent.eval_name).toBe('is_factually_consistent');
        });

        it('should have IsCompliant template', () => {
            expect(Templates.IsCompliant).toBeDefined();
            expect(Templates.IsCompliant.eval_id).toBe('96');
            expect(Templates.IsCompliant.eval_name).toBe('is_compliant');
        });

        it('should have IsHelpful template', () => {
            expect(Templates.IsHelpful).toBeDefined();
            expect(Templates.IsHelpful.eval_id).toBe('84');
            expect(Templates.IsHelpful.eval_name).toBe('is_helpful');
        });

        it('should have EvaluateFunctionCalling template', () => {
            expect(Templates.EvaluateFunctionCalling).toBeDefined();
            expect(Templates.EvaluateFunctionCalling.eval_id).toBe('98');
            expect(Templates.EvaluateFunctionCalling.eval_name).toBe('evaluate_function_calling');
        });

        it('should have ChunkAttribution template', () => {
            expect(Templates.ChunkAttribution).toBeDefined();
            expect(Templates.ChunkAttribution.eval_id).toBe('11');
            expect(Templates.ChunkAttribution.eval_name).toBe('chunk_attribution');
        });

        it('should have ChunkUtilization template', () => {
            expect(Templates.ChunkUtilization).toBeDefined();
            expect(Templates.ChunkUtilization.eval_id).toBe('12');
            expect(Templates.ChunkUtilization.eval_name).toBe('chunk_utilization');
        });

        it('should have BleuScore template', () => {
            expect(Templates.BleuScore).toBeDefined();
            expect(Templates.BleuScore.eval_id).toBe('101');
            expect(Templates.BleuScore.eval_name).toBe('bleu_score');
        });
    });

    describe('Template Count', () => {
        it('should have 47 templates defined', () => {
            const templateCount = Object.keys(Templates).length;
            expect(templateCount).toBe(47);
        });
    });

    describe('Template Usage', () => {
        it('should be usable with Evaluator.evaluate', () => {
            // Test that template objects can be used as expected
            const template: EvalTemplate = Templates.Groundedness;
            expect(template.eval_name).toBe('groundedness');
            expect(template.eval_id).toBe('47');
        });

        it('should allow accessing template by key', () => {
            const templateKey: keyof typeof Templates = 'Toxicity';
            const template = Templates[templateKey];
            expect(template.eval_name).toBe('toxicity');
        });

        it('should allow iterating over all templates', () => {
            const templateNames: string[] = [];
            Object.values(Templates).forEach(template => {
                templateNames.push(template.eval_name);
            });
            expect(templateNames).toContain('groundedness');
            expect(templateNames).toContain('toxicity');
            expect(templateNames).toContain('factual_accuracy');
        });
    });
});
