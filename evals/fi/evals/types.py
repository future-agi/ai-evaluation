from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ConfigParam(BaseModel):
    type: str
    default: Optional[Any] = None


class ConfigPossibleValues(BaseModel):
    min_length: Optional[int] = None
    validations: Optional[List[str]] = None
    eval_prompt: Optional[str] = None
    substring: Optional[str] = None
    model: Optional[str] = None
    code: Optional[str] = None
    keywords: Optional[List[str]] = None
    keyword: Optional[str] = None
    failure_threshold: Optional[float] = None
    headers: Optional[Dict[str, str]] = None
    case_sensitive: Optional[bool] = None
    comparator: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    input: Optional[str] = None
    max_length: Optional[int] = None
    multi_choice: Optional[bool] = None
    system_prompt: Optional[str] = None
    pattern: Optional[str] = None
    grading_criteria: Optional[str] = None
    _schema: Optional[str] = None
    rule_prompt: Optional[str] = None
    choices: Optional[List[str]] = None


class DatapointFieldAnnotation(BaseModel):
    """
    The annotations to be logged for the datapoint field.
    """

    field_name: str
    text: str
    annotation_type: str
    annotation_note: str


class EvalResultMetric(BaseModel):
    """
    Represents the LLM evaluation result metric.
    """

    id: Union[str, int, float]
    value: Union[str, int, float, List[Any]]


class EvalResult(BaseModel):
    """
    Represents the LLM evaluation result.
    """

    data: Optional[Union[Dict[str, Any], List[Any]]] = None
    failure: Optional[bool]
    reason: str
    runtime: int
    metadata: Optional[Union[str, List[Any], Dict[str, Any]]] = None
    metrics: List[EvalResultMetric]


class BatchRunResult(BaseModel):
    """
    Represents the result of a batch run of LLM evaluation.
    """

    eval_results: List[Optional[EvalResult]]


class RequiredKeys(Enum):
    text = "text"
    response = "response"
    query = "query"
    context = "context"
    expected_response = "expected_response"
    expected_text = "expected_text"
    document = "document"
    input = "input"
    output = "output"
    prompt = "prompt"
    image_url = "image_url"
    input_image_url = "input_image_url"
    output_image_url = "output_image_url"
    actual_json = "actual_json"
    expected_json = "expected_json"
    messages = "messages"


class EvalTags(Enum):
    CONVERSATION = "CONVERSATION"
    HALLUCINATION = "HALLUCINATION"
    RAG = "RAG"
    FUTURE_EVALS = "FUTURE_EVALS"
    LLMS = "LLMS"
    CUSTOM = "CUSTOM"
    FUNCTION = "FUNCTION"
    IMAGE = "IMAGE"
    SAFETY = "SAFETY"
    TEXT = "TEXT"


class Comparator(Enum):
    COSINE = "CosineSimilarity"
    LEVENSHTEIN = "NormalisedLevenshteinSimilarity"
    JARO_WINKLER = "JaroWincklerSimilarity"
    JACCARD = "JaccardSimilarity"
    SORENSEN_DICE = "SorensenDiceSimilarity"
