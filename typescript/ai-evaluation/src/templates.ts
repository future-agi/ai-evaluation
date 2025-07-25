// TypeScript translation of the Python EvalTemplate classes

interface EvalTemplate {
  eval_id: string;
  eval_name: string;
  description?: string;
  eval_tags?: string[];
  required_keys?: string[];
  output?: string;
  eval_type_id?: string;
  config_schema?: Record<string, any>;
  criteria?: string;
  choices?: string[];
  multi_choice?: boolean;
}

const Templates: Record<string, EvalTemplate> = {
  ConversationCoherence: {
    eval_name: "conversation_coherence",
    eval_id: "1"
  },
  ConversationResolution: {
    eval_name: "conversation_resolution",
    eval_id: "2"
  },
  ContentModeration: {
    eval_name: "content_moderation",
    eval_id: "4"
  },
  ContextAdherence: {
    eval_name: "context_adherence",
    eval_id: "5"
  },
  ContextRelevance: {
    eval_name: "context_relevance",
    eval_id: "9"
  },
  Completeness: {
    eval_name: "completeness",
    eval_id: "10"
  },
  ChunkAttribution: {
    eval_name: "chunk_attribution",
    eval_id: "11"
  },
  ChunkUtilization: {
    eval_name: "chunk_utilization",
    eval_id: "12"
  },
  PII: {
    eval_name: "pii",
    eval_id: "14"
  },
  Toxicity: {
    eval_name: "toxicity",
    eval_id: "15"
  },
  Tone: {
    eval_name: "tone",
    eval_id: "16"
  },
  Sexist: {
    eval_name: "sexist",
    eval_id: "17"
  },
  PromptInjection: {
    eval_name: "prompt_injection",
    eval_id: "18"
  },
  NotGibberishText: {
    eval_name: "not_gibberish_text",
    eval_id: "19"
  },
  SafeForWorkText: {
    eval_name: "safe_for_work_text",
    eval_id: "20"
  },
  PromptAdherence: {
    eval_name: "prompt_adherence",
    eval_id: "65"
  },
  DataPrivacyCompliance: {
    eval_name: "data_privacy_compliance",
    eval_id: "22"
  },
  IsJson: {
    eval_name: "is_json",
    eval_id: "23"
  },
  OneLine: {
    eval_name: "one_line",
    eval_id: "38"
  },
  ContainsValidLink: {
    eval_name: "contains_valid_link",
    eval_id: "39"
  },
  IsEmail: {
    eval_name: "is_email",
    eval_id: "40"
  },
  NoValidLinks: {
    eval_name: "no_valid_links",
    eval_id: "42"
  },
  Groundedness: {
    eval_name: "groundedness",
    eval_id: "47"
  },
  Ranking: {
    eval_name: "eval_ranking",
    eval_id: "61"
  },
  SummaryQuality: {
    eval_name: "summary_quality",
    eval_id: "64"
  },
  FactualAccuracy: {
    eval_name: "factual_accuracy",
    eval_id: "66"
  },
  TranslationAccuracy: {
    eval_name: "translation_accuracy",
    eval_id: "67"
  },
  CulturalSensitivity: {
    eval_name: "cultural_sensitivity",
    eval_id: "68"
  },
  BiasDetection: {
    eval_name: "bias_detection",
    eval_id: "69"
  },
  LLMFunctionCalling: {
    eval_name: "llm_function_calling",
    eval_id: "72"
  },
  AudioTranscriptionEvaluator: {
    eval_name: "audio_transcription",
    eval_id: "73"
  },
  AudioQualityEvaluator: {
    eval_name: "audio_quality",
    eval_id: "75"
  },
  NoRacialBias: {
    eval_name: "no_racial_bias",
    eval_id: "77"
  },
  NoGenderBias: {
    eval_name: "no_gender_bias",
    eval_id: "78"
  },
  NoAgeBias: {
    eval_name: "no_age_bias",
    eval_id: "79"
  },
  NoOpenAIReference: {
    eval_name: "no_openai_reference",
    eval_id: "80"
  },
  NoApologies: {
    eval_name: "no_apologies",
    eval_id: "81"
  },
  IsPolite: {
    eval_name: "is_polite",
    eval_id: "82"
  },
  IsConcise: {
    eval_name: "is_concise",
    eval_id: "83"
  },
  IsHelpful: {
    eval_name: "is_helpful",
    eval_id: "84"
  },
  IsCode: {
    eval_name: "is_code",
    eval_id: "85"
  },
  IsCSV: {
    eval_name: "is_csv",
    eval_id: "86"
  },
  FuzzyMatch: {
    eval_name: "fuzzy_match",
    eval_id: "87"
  },
  AnswerRefusal: {
    eval_name: "answer_refusal",
    eval_id: "88"
  },
  DetectHallucinationMissingInfo: {
    eval_name: "detect_hallucination_missing_info",
    eval_id: "89"
  },
  NoHarmfulTherapeuticGuidance: {
    eval_name: "no_harmful_therapeutic_guidance",
    eval_id: "90"
  },
  ClinicallyInappropriateTone: {
    eval_name: "clinically_inappropriate_tone",
    eval_id: "91"
  },
  IsHarmfulAdvice: {
    eval_name: "is_harmful_advice",
    eval_id: "92"
  },
  ContentSafety: {
    eval_name: "content_safety_violation",
    eval_id: "93"
  },
  IsGoodSummary: {
    eval_name: "is_good_summary",
    eval_id: "94"
  },
  IsFactuallyConsistent: {
    eval_name: "is_factually_consistent",
    eval_id: "95"
  },
  IsCompliant: {
    eval_name: "is_compliant",
    eval_id: "96"
  },
  IsInformalTone: {
    eval_name: "is_informal_tone",
    eval_id: "97"
  },
  EvaluateFunctionCalling: {
    eval_name: "evaluate_function_calling",
    eval_id: "98"
  },
  TaskCompletion: {
    eval_name: "task_completion",
    eval_id: "99"
  },
  CaptionHallucination: {
    eval_name: "caption_hallucination",
    eval_id: "100"
  },
  BleuScore: {
    eval_name: "bleu_score",
    eval_id: "101"
  }
};


export {
    Templates,
    EvalTemplate
}