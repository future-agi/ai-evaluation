# CI/CD Integration Examples

This directory contains examples for integrating AI evaluations into your CI/CD pipeline.

## GitHub Actions

### Quick Setup

1. **Copy the workflow file** to your repository:
   ```bash
   mkdir -p .github/workflows
   cp examples/ci-cd/.github/workflows/evaluation.yml .github/workflows/
   ```

2. **Add secrets** to your GitHub repository:
   - Go to Settings → Secrets and variables → Actions
   - Add `FI_API_KEY` with your Future AGI API key
   - Add `FI_SECRET_KEY` with your Future AGI secret key

3. **Create your evaluation config**:
   ```bash
   fi init --template rag
   # Edit fi-evaluation.yaml as needed
   ```

4. **Commit and push**:
   ```bash
   git add .github/workflows/evaluation.yml fi-evaluation.yaml
   git commit -m "Add AI evaluation pipeline"
   git push
   ```

### Workflow Features

The provided workflow includes:

- **Automatic triggers**: Runs on push to main/master and on PRs
- **Manual trigger**: Can be run manually via workflow_dispatch
- **JUnit reporting**: Results appear in GitHub's test summary
- **Artifact upload**: Results saved as downloadable artifacts
- **HTML reports**: Visual reports for review

### Customization

#### Change evaluation config

Edit the `fi run` command to use a specific config:

```yaml
- name: Run evaluations
  run: fi run -c path/to/your/config.yaml
```

#### Assertions (Quality Gates)

Assertions allow you to define quality gates that must pass for the CI build to succeed.
When assertions fail, the `fi run` command exits with code 2, causing the build to fail.

```yaml
# fi-evaluation.yaml
assertions:
  # Per-template assertions with multiple conditions
  - template: "groundedness"
    conditions:
      - "pass_rate >= 0.85"      # 85% pass rate required
      - "avg_score >= 0.7"        # Average score >= 0.7
    on_fail: "error"              # error = fail build, warn = warning only

  # Global assertions across all evaluations
  - global: true
    conditions:
      - "total_pass_rate >= 0.90" # Overall 90% pass rate
      - "failed_count < 10"       # Max 10 failures allowed
    on_fail: "error"

# Alternative: Simple threshold shortcuts
thresholds:
  default_pass_rate: 0.80         # Default for all templates
  fail_fast: true                 # Stop on first failure
  overrides:
    groundedness: 0.85            # Higher threshold for groundedness
    pii: 0.99                     # Near-perfect for PII detection
```

**Supported metrics:**
- `pass_rate`, `total_pass_rate` - Percentage of passing evaluations
- `avg_score`, `min_score`, `max_score` - Score statistics (for numeric outputs)
- `p50_score`, `p90_score`, `p95_score` - Score percentiles
- `passed_count`, `failed_count`, `total_count` - Evaluation counts
- `runtime_avg`, `runtime_p95` - Runtime metrics in milliseconds

**Exit codes:**
- `0` - All evaluations and assertions passed
- `2` - One or more assertions failed
- `3` - Assertions passed but with warnings (when using `--strict`)

**CLI flags:**
```bash
fi run --check          # Enable assertion checking (default)
fi run --no-check       # Disable assertion checking
fi run --fail-fast      # Stop on first assertion failure
fi run --strict         # Treat warnings as errors
```

#### Matrix builds

Enable the `matrix-evaluate` job to run different configs in parallel:

```yaml
matrix-evaluate:
  if: true  # Change from false to true
```

## GitLab CI

Create `.gitlab-ci.yml`:

```yaml
stages:
  - evaluate

evaluation:
  stage: evaluate
  image: python:3.11
  variables:
    FI_API_KEY: $FI_API_KEY
    FI_SECRET_KEY: $FI_SECRET_KEY
  before_script:
    - pip install ai-evaluation
  script:
    - fi run --output json -O results.json
    - fi export --last -o results.xml -f junit
  artifacts:
    when: always
    paths:
      - results.json
      - results.xml
    reports:
      junit: results.xml
```

## CircleCI

Create `.circleci/config.yml`:

```yaml
version: 2.1

jobs:
  evaluate:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install ai-evaluation
      - run:
          name: Run evaluations
          command: |
            fi run --output json -O results.json
            fi export --last -o results.xml -f junit
      - store_test_results:
          path: results.xml
      - store_artifacts:
          path: results.json

workflows:
  evaluation:
    jobs:
      - evaluate
```

## Jenkins

Create `Jenkinsfile`:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.11'
        }
    }

    environment {
        FI_API_KEY = credentials('fi-api-key')
        FI_SECRET_KEY = credentials('fi-secret-key')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install ai-evaluation'
            }
        }

        stage('Evaluate') {
            steps {
                sh 'fi run --output json -O results.json'
                sh 'fi export --last -o results.xml -f junit'
            }
        }
    }

    post {
        always {
            junit 'results.xml'
            archiveArtifacts artifacts: 'results.json'
        }
    }
}
```

## Azure Pipelines

Create `azure-pipelines.yml`:

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: pip install ai-evaluation
    displayName: 'Install dependencies'

  - script: |
      fi run --output json -O results.json
      fi export --last -o results.xml -f junit
    displayName: 'Run evaluations'
    env:
      FI_API_KEY: $(FI_API_KEY)
      FI_SECRET_KEY: $(FI_SECRET_KEY)

  - task: PublishTestResults@2
    inputs:
      testResultsFormat: 'JUnit'
      testResultsFiles: 'results.xml'
    condition: always()

  - task: PublishBuildArtifacts@1
    inputs:
      pathToPublish: 'results.json'
      artifactName: 'evaluation-results'
    condition: always()
```

## Local Execution for Fast CI

For faster CI feedback, you can run heuristic metrics locally without API calls. This is useful for quick validation checks before running full cloud evaluations.

### Available Local Metrics

- **String metrics**: `contains`, `regex`, `equals`, `starts_with`, `ends_with`, `length_*`
- **JSON metrics**: `is_json`, `contains_json`, `json_schema`
- **Similarity metrics**: `bleu_score`, `rouge_score`, `levenshtein_similarity`

### Example: Local Pre-check

```yaml
jobs:
  local-check:
    name: Quick Local Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ai-evaluation
      - name: Run local evaluations
        run: |
          python -c "
          from fi.evals.local import LocalEvaluator
          import json

          evaluator = LocalEvaluator()
          test_data = json.load(open('test_data.json'))

          result = evaluator.evaluate_batch([
              {'metric_name': 'is_json', 'inputs': test_data},
              {'metric_name': 'length_between', 'inputs': test_data, 'config': {'min_length': 10, 'max_length': 5000}},
          ])

          passed = sum(1 for r in result.results.eval_results if r.output == 1.0)
          total = len(result.results.eval_results)
          print(f'Passed: {passed}/{total}')
          exit(0 if passed == total else 1)
          "

  cloud-evaluation:
    name: Full Cloud Evaluation
    needs: local-check  # Only run if local checks pass
    runs-on: ubuntu-latest
    # ... full evaluation workflow
```

See the [Local Execution Guide](../../docs/local-execution.md) for more details.

## Best Practices

1. **Store credentials securely**: Never commit API keys. Use secrets/environment variables.

2. **Run on PRs**: Catch issues before they're merged.

3. **Use thresholds**: Define minimum acceptable scores and fail builds that don't meet them.

4. **Archive results**: Keep evaluation history for tracking over time.

5. **Separate configs**: Use different configs for different evaluation suites (rag, safety, etc.).

6. **Schedule nightly runs**: In addition to PR checks, run comprehensive evaluations nightly.

7. **Use local checks first**: Run fast local metrics before expensive cloud evaluations.

## Example Config

See `example-config.yaml` for a complete CI-ready evaluation configuration.
