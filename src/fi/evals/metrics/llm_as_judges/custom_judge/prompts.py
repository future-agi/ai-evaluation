DEFAULT_USER_PROMPT_TEMPLATE = """
### GRADING CRITERIA ###
{{ grading_criteria }}

### OUTPUT FORMAT ###
You MUST return a valid JSON object with two keys: "score" (a float between 0.0 and 1.0) and "reason" (a brief explanation of your score).

{% if few_shot_examples %}
### EXAMPLES ###
{% for example in few_shot_examples -%}
---
INPUT:
{{ example.inputs | tojson(indent=2) }}

EXPECTED JUDGEMENT:
{{ example.output }}
---
{% endfor %}
{% endif %}

### TASK ###
Based on the grading criteria, please evaluate the following input.

### INPUT ###
{{ task_input | tojson(indent=2) }}
"""
