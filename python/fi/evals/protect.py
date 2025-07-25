import copy
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fi.api.types import HttpMethod, RequestConfig
# from fi.evals import EvalClient  # Removed, as EvalClient does not exist
from fi.evals.evaluator import EvalResponseHandler, Evaluator
from fi.evals.templates import (
    DataPrivacyCompliance,
    PromptInjection,
    Sexist,
    Tone,
    Toxicity,
    BiasDetection,
)
from fi.testcases.mllm_test_case import MLLMTestCase
from fi.utils.routes import Routes
from fi.utils.utils import get_keys_from_env, get_base_url_from_env
from fi.utils.errors import InvalidAuthError, SDKException, InvalidValueType, MissingRequiredKey


PROTECT_FLASH_ID = "76"

class Protect:
    """Client for protecting against unwanted content using various metrics"""

    def __init__(self, 
                 fi_api_key: Optional[str] = None, 
                 fi_secret_key: Optional[str] = None, 
                 fi_base_url: Optional[str] = None, 
                 evaluator: Optional[Evaluator] = None):
        """
        Initialize Protect Class

        Args:
            evaluator: Instance of Evaluator to use for evaluations. If None, creates a new one.
        """
        env_api_key, env_secret_key = get_keys_from_env() 
        fi_api_key = env_api_key or fi_api_key
        fi_secret_key = env_secret_key or fi_secret_key
        fi_base_url = get_base_url_from_env() or fi_base_url
        if not fi_api_key or not fi_secret_key:
            raise InvalidAuthError("API key or secret key is missing for Protect initialization.")
        
        self.evaluator = evaluator if evaluator is not None else Evaluator(
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url
        )

        # Map metric names to their corresponding template classes
        self.metric_map = {
            "content_moderation": Toxicity,
            "bias_detection": BiasDetection,
            "security": PromptInjection,
            "data_privacy_compliance": DataPrivacyCompliance,
        }

    def _check_rule_sync(
        self, rule: Dict, test_case: MLLMTestCase
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
        """
        Synchronous version of rule checking

        Returns:
            Tuple[str, bool, Optional[str], Optional[str]]:
        """
        # thread_name = threading.current_thread().name
        # start_time = time.time()
        # print(f"Starting rule check for {rule['metric']} in thread {thread_name} at {start_time}")

        template_class = self.metric_map[rule["metric"]]
        if rule["metric"] == "Data Privacy":
            template = template_class(
                config={"call_type": "protect", "check_internet": False}
            )
            # template = template_class(config={"check_internet": False})
        else:
            template = template_class(config={"call_type": "protect"})
            # template = template_class(config={})

        payload = {
            "inputs": [test_case.model_dump()],
            "config": {
                template.eval_id: template.config
            },
        }

        eval_result = self.evaluator.request(
            config=RequestConfig(
                method=HttpMethod.POST,
                url=f"{self.evaluator._base_url}/{Routes.evaluate.value}",
                json=payload,
                timeout=3000
            ),
            response_handler=EvalResponseHandler,
        )


        # end_time = time.time()
        # print(f"Completed rule check for {rule['metric']} in thread {thread_name} at {end_time} (took {end_time - start_time:.2f}s)")

        reason_text: Optional[str] = None

        if eval_result.eval_results:
            result = eval_result.eval_results[0]
            detected_values = [result.output]

            should_trigger = False
            if rule["type"] == "any":
                should_trigger = any(
                    value in rule["contains"] for value in detected_values
                )
            elif rule["type"] == "all":
                should_trigger = all(
                    value in rule["contains"] for value in detected_values
                )

            if should_trigger:
                if rule["_internal_reason_flag"]:
                    # message = rule['action'] + f' Reason: {result.reason}'
                    message = rule["action"]
                    reason_text = result.reason
                else:
                    message = rule["action"]
                return rule["metric"], True, message, reason_text

        return rule["metric"], False, None, None

    def _process_rules_batch(
        self, rules: List[Dict], test_case: MLLMTestCase, remaining_time: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Process a batch of rules in parallel

        Args:
            rules: List of rules to process
            test_case: Test case to evaluate
            remaining_time: Time remaining for processing

        Returns:
            Tuple[List[str], List[str], List[str]]:
                (failure_messages, completed_rules, uncompleted_rules)
        """
        # print(f"\nProcessing batch of {len(rules)} rules")
        # batch_start = time.time()

        completed_rules = []
        uncompleted_rules = [rule["metric"] for rule in rules]
        failure_messages = []
        failure_reasons = []
        failed_rule = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all rules to the thread pool
            future_to_rule = {
                executor.submit(self._check_rule_sync, rule, test_case): rule["metric"]
                for rule in rules
            }

            try:
                # Wait for futures to complete with timeout
                for future in as_completed(future_to_rule, timeout=remaining_time):
                    rule_name = future_to_rule[future]
                    try:
                        metric, triggered, message, reason_text = future.result()
                        # Update tracking lists
                        completed_rules.append(metric)
                        if rule_name in uncompleted_rules:
                            uncompleted_rules.remove(rule_name)

                        if triggered:
                            failure_messages.append(message)
                            if reason_text:
                                failure_reasons.append(reason_text)
                            failed_rule.append(rule_name)
                            # Cancel remaining futures if a rule fails
                            for f_key, f_val in future_to_rule.items():
                                if not f_key.done():
                                    f_key.cancel()

                    except Exception as e:
                        if rule_name in uncompleted_rules:
                            # uncompleted_rules.remove(rule_name) # Errored rule should remain uncompleted
                            pass 

            except TimeoutError:
                # print(
                #     f"Timeout reached. {len(completed_rules)} rules completed, "
                #     f"{len(uncompleted_rules)} rules incomplete"
                # )
                all_submitted_rules = [r["metric"] for r in rules]
                uncompleted_rules = [r for r in all_submitted_rules if r not in completed_rules]

        # batch_end = time.time()
        # print(f"Batch processing completed in {batch_end - batch_start:.2f}s\n")

        return (
            failure_messages,
            completed_rules,
            uncompleted_rules,
            failure_reasons,
            failed_rule,
        )

    def _is_url(self, text: str) -> bool:
        """
        Check if the input text is a URL or URL-like string.
        
        Args:
            text: String to check
            
        Returns:
            bool: True if input appears to be a URL, False otherwise
        """
        # Check if it's an explicit URL with a scheme
        parsed_url = urlparse(text)
        if parsed_url.scheme in ['http', 'https']:
            return True
            
        # Check for URL-like patterns without scheme
        text_lower = text.lower()
        # Check for common TLDs
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.io', '.co']
        has_tld = any(tld in text_lower for tld in common_tlds)
        
        # Check for patterns like "www." at the beginning
        starts_with_www = text_lower.startswith('www.')
        
        # Check for domain-like pattern (example.com)
        has_domain_pattern = '.' in text_lower and not text_lower.startswith('.') and not text_lower.endswith('.')
        
        return (has_tld and has_domain_pattern) or starts_with_www

    def _is_only_url(self, text: str) -> bool:
        """
        Check if the input text is solely a URL without additional content.
        
        Args:
            text: String to check
            
        Returns:
            bool: True if the entire input appears to be just a URL, False otherwise
        """
        # Remove whitespace for checking
        text = text.strip()
        
        # If the string contains spaces, it's not only a URL
        if ' ' in text:
            return False
            
        # Check if what remains is a URL
        return self._is_url(text)

    def protect(
        self,
        inputs: str,
        protect_rules: Optional[List[Dict]] = None,
        action: str = "Response cannot be generated as the input fails the checks",
        reason: bool = False,
        timeout: float = 30000, #milliseconds
        use_flash: bool = False,
    ) -> List[str]:
        """
        Evaluate input strings against protection rules

        Args:
            inputs: Text or list of texts to check for harmful content
            timeout: Time limit for evaluation in milliseconds (default: 30000)
            protect_rules: Rules to check against. Each rule needs:
                metric: What to check (e.g. 'content_moderation', 'bias_detection')
                contains: Values to look for
                type: 'any' or 'all' matching required
                action: Message to show if rule fails
                reason: Include explanation in message (optional)
            use_flash: Use fast binary classification instead of detailed rules

        Returns:
            Dictionary containing:
                status: "passed" or "failed"
                completed_rules: List of rules that were checked
                uncompleted_rules: List of rules that couldn't be checked
                failed_rule: The rule that triggered the failure (if any)
                messages: The action message for the failed rule or the input if passed
                reasons: The reason for failure or "All checks passed"
                time_taken: Elapsed time for the evaluation

        Raises:
            ValueError: If inputs or protect_rules don't match the required structure
            TypeError: If inputs contains non-string objects
        """


        timeout_seconds = timeout / 1000.0

        # When using ProtectFlash and no protect_rules provided, create default rules
        if use_flash and not protect_rules:
            protect_rules = [{"metric": "content_moderation"}]
        elif use_flash and protect_rules:
            print("Note: When using ProtectFlash, Rules are not considered as it performs binary harmful/not harmful classification only.")

        # Ensure protect_rules is a list
        if protect_rules is None:
            protect_rules = []
            
        protect_rules_copy = copy.deepcopy(protect_rules)

        # Validate inputs
        if inputs is None:
            raise InvalidValueType(value_name="inputs", value=inputs, correct_type="string or list of strings")

        # This check can be more specific if we only expect str initially that gets converted
        if not isinstance(inputs, (str, list)):
             raise InvalidValueType(value_name="inputs", value=inputs, correct_type="string or list of strings")

        # Convert single string to list for uniform processing
        if isinstance(inputs, str):
            inputs_list = [inputs]
        else:
            inputs_list = inputs # Already a list

        if not inputs_list: # Check after potential conversion
            raise InvalidValueType(value_name="inputs", value=inputs_list, correct_type="non-empty string or non-empty list of strings")

        # Validate each input is a non-empty string
        for i, input_text in enumerate(inputs_list):
            if not isinstance(input_text, str):
                raise InvalidValueType(
                    value_name=f"input at index {i}", 
                    value=input_text, 
                    correct_type="string"
                )
            if not input_text.strip():
                raise InvalidValueType(
                    value_name=f"input at index {i}", 
                    value=input_text, 
                    correct_type="non-empty string or string with non-whitespace characters"
                )

        # If using ProtectFlash, we can use a simpler approach by directly calling the API with protect_flash=True
        if use_flash:
            # Create a test case with appropriate payload
            test_case = MLLMTestCase(input=inputs, call_type="protect")
            
            # Prepare the protect API call with protect_flash flag
            template_class = self.metric_map[protect_rules_copy[0]["metric"]]
            template = template_class(config={"call_type": "protect"})
            
            # Ensure action is set in the rule for consistency with standard protect
            # This allows the user to provide action either as a parameter or in the rule itself
            if "action" not in protect_rules_copy[0]:
                protect_rules_copy[0]["action"] = action
            
            # Call the evaluator with protect_flash=True
            # Custom payload for ProtectFlash that includes the protect_flash flag 
            payload = {
                "inputs": [test_case.model_dump()],
                "config": {
                    PROTECT_FLASH_ID: template.config
                },
                "protect_flash": True  # This is the key flag that enables ProtectFlash
            }
            
            # Make a direct request using the evaluator's request method
            response = self.evaluator.request(
                config=RequestConfig(
                    method=HttpMethod.POST,
                    url=f"{self.evaluator._base_url}/{Routes.evaluate.value}",
                    json=payload,
                    timeout=timeout / 1000 or self.evaluator._default_timeout,
                ),
                response_handler=EvalResponseHandler,
            )
            # Process the response
            if hasattr(response, "eval_results") and response.eval_results:
                result = response.eval_results[0]
                is_harmful = result.output
                elapsed_time = result.runtime / 1000 if result.runtime else 0
                
                ans = {
                    "status": "failed" if is_harmful else "passed",
                    "completed_rules": ["ProtectFlash"],  # Use ProtectFlash instead of rule metric
                    "uncompleted_rules": [],
                    "failed_rule": "ProtectFlash" if is_harmful else None,  # Use ProtectFlash instead of rule metric
                    "messages": protect_rules_copy[0]["action"] if is_harmful else inputs[0],
                    "reasons": (
                        f"Content detected as harmful." if is_harmful else "All checks passed"
                    ),
                    "time_taken": elapsed_time,
                }
                return ans
            else:
                # Return a default response if no results
                 return {
                    "status": "error",
                    "messages": "Evaluation failed",
                    "completed_rules": [],
                    "uncompleted_rules": ["ProtectFlash"],
                    "failed_rule": None,
                    "reasons": "No evaluation results returned",
                    "time_taken": 0,
                }

        
        # Original implementation for standard Protect (non-flash)
        # Convert inputs to MLLMTestCase instances with call_type="protect"
        test_cases = [MLLMTestCase(input=input_text, call_type="protect") for input_text in inputs_list]

        # Validate protect_rules_copy
        if not isinstance(protect_rules_copy, list):
            raise InvalidValueType(value_name="protect_rules", value=protect_rules_copy, correct_type="list")

        if not protect_rules_copy:
            raise InvalidValueType(value_name="protect_rules", value=protect_rules_copy, correct_type="non-empty list")

        valid_metrics = set(self.metric_map.keys())
        valid_types = {"any", "all"}

        for i, rule in enumerate(protect_rules_copy):

            if not isinstance(rule, dict):
                raise InvalidValueType(value_name=f"Rule at index {i}", value=rule, correct_type="dictionary")

            # Check required keys
            required_keys = {"metric"}
            missing_keys = required_keys - set(rule.keys())
            if missing_keys:
                # Using MissingRequiredKey from our errors module
                raise MissingRequiredKey(field_name=f"Rule at index {i}", missing_key=', '.join(missing_keys))

            # Validate metric name first, as other validations might depend on it
            if rule["metric"] not in valid_metrics:
                raise InvalidValueType(
                    value_name=f"metric in Rule at index {i}", 
                    value=rule["metric"], 
                    correct_type=f"one of {list(valid_metrics)}"
                )

         
            is_tone_metric = rule["metric"] == "Tone"

            if is_tone_metric:
                if "contains" not in rule:
                    raise MissingRequiredKey(field_name=f"Rule for Tone metric at index {i}", missing_key="contains")
                if not isinstance(rule["contains"], list):
                    raise InvalidValueType(value_name=f"'contains' in Tone rule at index {i}", value=rule["contains"], correct_type="list")
                if not rule["contains"]:
                    raise InvalidValueType(value_name=f"'contains' in Tone rule at index {i}", value=rule["contains"], correct_type="non-empty list")
                
                # Type for Tone metric
                if "type" not in rule:
                    rule["type"] = "any" # Default if not present
                elif rule["type"] not in valid_types:
                    raise InvalidValueType(
                        value_name=f"'type' in Tone rule at index {i}", 
                        value=rule["type"], 
                        correct_type=f"one of {valid_types}"
                    )
            else: # For non-Tone metrics
                if "contains" in rule:
                    # This indicates an invalid configuration for a non-Tone metric
                    raise SDKException(f"'contains' should not be specified for {rule['metric']} metric at index {i}. Provide it only for 'Tone' metric.")
                if "type" in rule:
                    raise SDKException(f"'type' should not be specified for {rule['metric']} metric at index {i}. Provide it only for 'Tone' metric.")
                
                # Set default values for internal processing of non-Tone metrics
                rule["contains"] = ["Failed"] # Predefined internal value to check against for non-Tone metrics
                rule["type"] = "any" # Default type for non-Tone metrics

            # Validate action 
            if "action" not in rule:
                rule["action"] = action # Default action if not specified

            # 'reason' should not be in the input rule, it's a parameter to the protect method itself.
            if "reason" in rule:
                raise InvalidValueType(value_name=f"key in rule at index {i}", value="reason", correct_type="not to be part of the rule, it is a global parameter")
            # Set the global reason for this rule processing from the method's parameter
            rule["_internal_reason_flag"] = reason # Use a different key to avoid conflict

        # results = []
        BATCH_SIZE = 5  # Maximum number of concurrent rule checks
        if len(protect_rules_copy) < BATCH_SIZE:
            BATCH_SIZE = len(protect_rules_copy)

        # total_timeout = timeout # Original line, timeout is in ms
        total_timeout_for_processing_seconds = timeout_seconds 
        start_time = time.time() 

        all_failure_messages = []
        all_completed_rules = []
        all_uncompleted_rules = []
        all_failure_reasons = []
        # try:
        bool_check_fail = False
        for test_case in test_cases:
            for i in range(0, len(protect_rules_copy), BATCH_SIZE):
                # Calculate remaining time
                elapsed_time = time.time() - start_time # This is in seconds
                # remaining_time = max(0, total_timeout - elapsed_time) # BUG: total_timeout was ms
                remaining_time_seconds = max(0, total_timeout_for_processing_seconds - elapsed_time)

                if remaining_time_seconds <= 0:
                    # Add remaining rules to uncompleted list
                    remaining_rules = [
                        rule["metric"] for rule in protect_rules_copy[i:]
                    ]
                    all_uncompleted_rules.extend(remaining_rules)
                    break

                rules_batch = protect_rules_copy[i : i + BATCH_SIZE]
                (
                    messages,
                    completed,
                    uncompleted,
                    failure_reasons,
                    failed_rule,
                ) = self._process_rules_batch(rules_batch, test_case, remaining_time_seconds)

                all_completed_rules.extend(completed)
                all_uncompleted_rules.extend(uncompleted)
                all_failure_reasons.extend(failure_reasons)
                if messages:
                    all_failure_messages.extend(messages)
                    bool_check_fail = True
                    break

        final_processing_duration_seconds = time.time() - start_time

        ans = {
            "status": "failed" if all_failure_messages else "passed",
            "completed_rules": all_completed_rules,
            "uncompleted_rules": all_uncompleted_rules,
            "failed_rule": failed_rule,
            "messages": (
                all_failure_messages[0] if all_failure_messages else "All checks passed"
            ),
            "reasons": (
                all_failure_reasons if all_failure_reasons else "All checks passed"
            ),
            "time_taken": final_processing_duration_seconds, # Use final calculated duration in seconds
        }

        if len(ans["uncompleted_rules"]) == len(protect_rules_copy):
            ans["reason"] = "No checks completed"

        if bool_check_fail:
            ans["status"] = "failed"
        else:
            ans["status"] = "passed"
            # ans['messages'] = inputs

        if ans["status"] == "passed":
            ans["messages"] = inputs_list[0]

        return ans

protect = lambda inputs, protect_rules, action="Response cannot be generated as the input fails the checks", reason=False, timeout=30000: Protect().protect(inputs, protect_rules, action, reason, timeout)