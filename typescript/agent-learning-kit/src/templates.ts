// Template registry for cloud evals.
//
// Pure metadata — no Input schema enforcement. The cloud registry
// (src/core/cloudRegistry.ts) fetches required_keys from the api and
// maps user inputs dynamically at call time.
//
// To call a template by name without the metadata object, just pass the
// string directly:
//   evaluator.evaluate("customer_agent_query_handling", { conversation: [...] })
//
// Renamed templates get a module-level alias from the old key to the new
// entry (see bottom of file) so existing user code keeps working. Truly
// removed templates are gone — callers get a clean "undefined" signal
// instead of a silent runtime 400 from the api.

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
  FuzzyMatch: {
    eval_name: "fuzzy_match",
    eval_id: "87"
  },
  AnswerRefusal: {
    eval_name: "answer_refusal",
    eval_id: "88"
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
  },

  // -------------------- New in the api revamp --------------------
  NoLLMReference: { eval_name: "no_llm_reference", eval_id: "" },
  DetectHallucination: { eval_name: "detect_hallucination", eval_id: "" },
  ContainsCode: { eval_name: "contains_code", eval_id: "" },
  TextToSQL: { eval_name: "text_to_sql", eval_id: "" },
  GroundTruthMatch: { eval_name: "ground_truth_match", eval_id: "" },
  PromptInstructionAdherence: { eval_name: "prompt_instruction_adherence", eval_id: "" },
  ProtectFlash: { eval_name: "protect_flash", eval_id: "" },
  ImageInstructionAdherence: { eval_name: "image_instruction_adherence", eval_id: "" },
  SyntheticImageEvaluator: { eval_name: "synthetic_image_evaluator", eval_id: "" },
  OCREvaluation: { eval_name: "ocr_evaluation", eval_id: "" },
  ASRAccuracy: { eval_name: "ASR/STT_accuracy", eval_id: "" },
  TTSAccuracy: { eval_name: "TTS_accuracy", eval_id: "" },

  // Customer-agent family
  CustomerAgentClarificationSeeking: { eval_name: "customer_agent_clarification_seeking", eval_id: "" },
  CustomerAgentContextRetention: { eval_name: "customer_agent_context_retention", eval_id: "" },
  CustomerAgentConversationQuality: { eval_name: "customer_agent_conversation_quality", eval_id: "" },
  CustomerAgentHumanEscalation: { eval_name: "customer_agent_human_escalation", eval_id: "" },
  CustomerAgentInterruptionHandling: { eval_name: "customer_agent_interruption_handling", eval_id: "" },
  CustomerAgentLanguageHandling: { eval_name: "customer_agent_language_handling", eval_id: "" },
  CustomerAgentLoopDetection: { eval_name: "customer_agent_loop_detection", eval_id: "" },
  CustomerAgentObjectionHandling: { eval_name: "customer_agent_objection_handling", eval_id: "" },
  CustomerAgentPromptConformance: { eval_name: "customer_agent_prompt_conformance", eval_id: "" },
  CustomerAgentQueryHandling: { eval_name: "customer_agent_query_handling", eval_id: "" },
  CustomerAgentTerminationHandling: { eval_name: "customer_agent_termination_handling", eval_id: "" },
};

// -------------------- Back-compat aliases for renamed templates --------------------
// Old name → current entry so existing user code keeps working.
Templates.NoOpenAIReference = Templates.NoLLMReference;
Templates.DetectHallucinationMissingInfo = Templates.DetectHallucination;
Templates.LLMFunctionCalling = Templates.EvaluateFunctionCalling;
Templates.AudioTranscriptionEvaluator = Templates.ASRAccuracy;


export {
    Templates,
    EvalTemplate
}