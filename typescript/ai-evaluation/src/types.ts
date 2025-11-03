interface ConfigParam {
  type: string;
  default?: any;
}

interface ConfigPossibleValues {
  min_length?: number;
  validations?: string[];
  eval_prompt?: string;
  substring?: string;
  model?: string;
  code?: string;
  keywords?: string[];
  keyword?: string;
  failure_threshold?: number;
  headers?: Record<string, string>;
  case_sensitive?: boolean;
  comparator?: string;
  payload?: Record<string, any>;
  url?: string;
  input?: string;
  max_length?: number;
  multi_choice?: boolean;
  system_prompt?: string;
  pattern?: string;
  grading_criteria?: string;
  _schema?: string;
  rule_prompt?: string;
  choices?: string[];
}

interface DatapointFieldAnnotation {
  /**
   * The annotations to be logged for the datapoint field.
   */
  field_name: string;
  text: string;
  annotation_type: string;
  annotation_note: string;
}

interface EvalResultMetric {
  /**
   * Represents the LLM evaluation result metric.
   */
  id: string | number;
  value: string | number | any[];
}

interface EvalResult {
  /**
   * Represents the LLM evaluation result.
   */
  data?: Record<string, any> | any[];
  failure?: boolean;
  reason: string;
  runtime: number;
  metadata?: string | any[] | Record<string, any>;
  metrics: EvalResultMetric[];
}

interface BatchRunResult {
  /**
   * Represents the result of a batch run of LLM evaluation.
   */
  eval_results: (EvalResult | null)[];
}

enum RequiredKeys {
  text = "text",
  response = "response",
  query = "query",
  context = "context",
  expected_response = "expected_response",
  expected_text = "expected_text",
  document = "document",
  input = "input",
  output = "output",
  prompt = "prompt",
  image_url = "image_url",
  input_image_url = "input_image_url",
  output_image_url = "output_image_url",
  actual_json = "actual_json",
  expected_json = "expected_json",
  messages = "messages"
}

enum EvalTags {
  CONVERSATION = "CONVERSATION",
  HALLUCINATION = "HALLUCINATION",
  RAG = "RAG",
  FUTURE_EVALS = "FUTURE_EVALS",
  LLMS = "LLMS",
  CUSTOM = "CUSTOM",
  FUNCTION = "FUNCTION",
  IMAGE = "IMAGE",
  SAFETY = "SAFETY",
  TEXT = "TEXT"
}

enum Comparator {
  COSINE = "CosineSimilarity",
  LEVENSHTEIN = "NormalisedLevenshteinSimilarity",
  JARO_WINKLER = "JaroWincklerSimilarity",
  JACCARD = "JaccardSimilarity",
  SORENSEN_DICE = "SorensenDiceSimilarity"
}


export {
    ConfigParam,
    ConfigPossibleValues,
    DatapointFieldAnnotation,
    EvalResultMetric,
    EvalResult,
    BatchRunResult,
    RequiredKeys,
    EvalTags,
    Comparator
}