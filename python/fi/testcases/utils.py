from typing import List, Union

from fi.testcases.unified_test_case import TestCase


def check_valid_test_cases_type(
    test_cases: Union[List[TestCase], List[TestCase]]
):
    """
    Validate test cases for mixed types.
    Now simplified since all test cases are the same type, but we check their detected types.
    """
    llm_test_case_count = 0
    conversational_test_case_count = 0
    
    for test_case in test_cases:
        if not isinstance(test_case, TestCase):
            raise TypeError("All test cases must be instances of TestCase")
            
        # Check the auto-detected or manually set test case type
        test_type = test_case.test_case_type
        
        if test_type == "conversational":
            conversational_test_case_count += 1
        else:
            # LLM, multimodal, and general types are all treated as non-conversational
            llm_test_case_count += 1

    if llm_test_case_count > 0 and conversational_test_case_count > 0:
        raise ValueError(
            "You cannot supply a mixture of conversational and non-conversational test cases in the same batch."
        )
