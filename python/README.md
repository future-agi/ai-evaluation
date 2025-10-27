# Future AGI

![Company Logo](https://fi-content.s3.ap-south-1.amazonaws.com/Logo.png)

Welcome to Future AGI - Empowering GenAI Teams with Advanced Performance Management

# Overview

Future AGI provides a cutting-edge platform designed to help GenAI teams maintain peak model accuracy in production environments.
Our solution is purpose-built, scalable, and delivers results 10x faster than traditional methods.

**Key Features**

* **_Simplified GenAI Performance Management_**: Streamline your workflow and focus on developing cutting-edge AI models.
* **_Instant Evaluation_**: Score outputs without human-in-the-loop or ground truth, increasing QA team efficiency by up to 10x.
* **_Advanced Error Analytics_**: Gain ready-to-use insights with comprehensive error tagging and segmentation.
* **_Configurable Metrics_**: Define custom metrics tailored to your specific use case for precise model evaluation.

# Quickstart
---
title: Quickstart
---

This guide will walk you through setting up an evaluation in **Future AGI**, allowing you to assess AI models and workflows efficiently. You can run evaluations via the **Future AGI platform** or using the **Python SDK**.

## Access API Key

To authenticate while running evals, you will need Future AGI's API keys, which you can get access by following below steps:

- Go to your Future AGI dashboard
- Click on **Keys** under **Developer** option from left column

- Copy both, **API Key** and **Secret Key**

---

## Setup Evaluator 

Install the Future AGI Python SDK using below command:

```python
pip install ai-evaluation
```

Then initialise the Evaluator:

```python
from fi.evals import Evaluator

evaluator = Evaluator(
    fi_api_key="your_api_key",
    fi_secret_key="your_secret_key",
)
```

We recommend you to set the `FI_API_KEY` and `FI_SECRET_KEY` environment variables before using the `Evaluator` class, instead of passing them as parameters.

---


## Running Your First Eval

This section walks you through the process of running your first evaluation using the Future AGI evaluation framework. To get started, we'll use **Tone Evaluation** as an example.

### a. Using Python SDK

**Define the Test Case**

Create a test case containing the **text input** that will be evaluated for tone.

```python
from fi.testcases import TestCase

test_case = TestCase(
    input='''
    Dear Sir, I hope this email finds you well. 
    I look forward to any insights or advice you might have 
    whenever you have a free moment.
    '''
)

```

You can also directly send the data through a dictionary with valid keys. However, it is recommended to use the `TestCase` class when working with Future AGI Evaluations.


**Configure the Evaluation Template**

For **Tone Evaluation**, we use the **Tone Evaluation Template** to analyse the sentiment and emotional tone of the input.

```python
from fi.evals.templates import Tone

tone_eval = Tone() # This is the evaluation template to use provided by Future AGI
```

[Click here to read more about all the Evals provided by Future AGI](https://docs.futureagi.com/future-agi/products/evaluation/eval-definition/overview)

**Run the Evaluation**

Execute the evaluation and retrieve the results.

```python
result = evaluator.evaluate(eval_templates=[tone_eval], inputs=[test_case])
tone_result = result.eval_results[0].metrics[0].value
```


To Evaluate the data on your own evaluation template which you have created, you can use the `evaluate` function with the `eval_templates` parameter.

```python
from fi.evals import evaluate

result = evaluate(eval_templates="name-of-your-eval", inputs={
    "input": "your_input_text",
    "output": "your_output_text"
})

print(result.eval_results[0].metrics[0].value)
```

### b. Using Web Interface

**Select a Dataset**

Before running an evaluation, ensure you have selected a dataset. If no dataset is available, follow the steps to **Add Dataset** on the Future AGI platform.

[Read more about all the ways you can add dataset](https://docs.futureagi.com/future-agi/products/dataset/overview)

**Access the Evaluation Panel**

- Navigate to your dataset.
- Click on the **Evaluate** button in the top-right menu.
- This will open the evaluation configuration panel.

**Starting a New Evaluation**

- Click on the **Add Evaluation** button.
- You will be directed to the Evaluation List page. 
You can either create your own evaluation or select from the available templates built by Future AGI.
- Click on one of the available templates.
- Write the name of the evaluation and select the required dataset column.
<Tip>
Checkmark on **Error Localization** if you want to localize the errors in the dataset when the datapoint is evaluated and fails the evaluation.
</Tip>
- Click on the **Add & Run** button.


## Creating a New Evaluation

Future AGI provides a wide range of evaluation templates to choose from. You can create your own evaluation to tailor your needs by following below simple steps:

- Click on the **Create your own eval** button after clicking on the **Add Evaluation** button.
- Write the name of the evaluation, this name will be used to identify the evaluation in the evaluation list. only lower case letters, numbers and underscores are allowed in the name. 
- Select either **Use Future AGI Agents** or **Use other LLMs**

**Future AGI Agents** are our own proprietary models trained on a vast variety of datasets to perform evaluations. These models vary in capabilities and are suited for different use cases:
- **TURING_LARGE** – Flagship evaluation model that delivers best-in-class accuracy across multimodal inputs (text, images, audio). Recommended when maximal precision outweighs latency constraints.

- **TURING_SMALL** – Compact variant that preserves high evaluation fidelity while lowering computational cost. Supports text and image evaluations.

- **TURING_FLASH** – Latency-optimised version of TURING, providing high-accuracy assessments for text and image inputs with fast response times.

- **PROTECT** – Real-time guardrailing model for safety, policy compliance, and content-risk detection. Offers very low latency on text and audio streams and permits user-defined rule sets.

- **PROTECT_FLASH** – Ultra-fast binary guardrail for text content. Designed for first-pass filtering where millisecond-level turnaround is critical.

- In the Rule Prompt, you can write the rules that the evaluation should follow. Use `{{}}` to create a key (variable), that variable will be used in future when you configure the evaluation.
- Choose Output Type As either Pass/Fail or Percentage or Deterministic Choices
    - **Pass/Fail**: The evaluation will return either Pass or Fail.
    - **Percentage**: The evaluation will return a Score between 0 and 100.
    - **Deterministic Choices**: The evaluation will return a categorical choice from the list of choices.
- Select the Tags for the evaluation that are suitable to use case.
- Write the description of the evaluation that will be used to identify the evaluation in the evaluation list.
- Checkmark on **Check Internet** to power your evaluation with the latest information.
- Click on the **Create Evaluation** button.

---