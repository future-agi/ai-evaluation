/**
 * Backend route paths — single source of truth for url suffixes.
 *
 * Mirrors python/fi/utils/routes.py. Relative paths (no leading slash).
 * Template placeholders like {template_id} are replaced at call time.
 */
export const Routes = {
    // ---------- Health / logging ----------
    healthcheck: "healthcheck",
    log_model: "sdk/api/v1/log/model/",

    // ---------- Eval run (sync + async) ----------
    evaluate: "sdk/api/v1/eval/",
    evaluatev2: "sdk/api/v1/new-eval/",
    evaluate_template: "sdk/api/v1/eval/{eval_id}/",
    get_eval_templates: "sdk/api/v1/get-evals/",
    get_eval_result: "sdk/api/v1/new-eval/",
    evaluate_pipeline: "sdk/api/v1/evaluate-pipeline/",
    configure_evaluations: "sdk/api/v1/configure-evaluations/",

    // ---------- Eval templates (revamp) ----------
    eval_template_list: "model-hub/eval-templates/list/",
    eval_template_list_charts: "model-hub/eval-templates/list-charts/",
    eval_template_detail: "model-hub/eval-templates/{template_id}/detail/",
    eval_template_create_v2: "model-hub/eval-templates/create-v2/",
    eval_template_update_v2: "model-hub/eval-templates/{template_id}/update/",
    eval_template_delete: "model-hub/delete-eval-template/",
    eval_template_bulk_delete: "model-hub/eval-templates/bulk-delete/",
    eval_template_duplicate: "model-hub/duplicate-eval-template/",
    eval_template_usage: "model-hub/eval-templates/{template_id}/usage/",
    eval_template_feedback_list: "model-hub/eval-templates/{template_id}/feedback-list/",

    // Versions
    eval_template_version_list: "model-hub/eval-templates/{template_id}/versions/",
    eval_template_version_create: "model-hub/eval-templates/{template_id}/versions/create/",
    eval_template_version_set_default:
        "model-hub/eval-templates/{template_id}/versions/{version_id}/set-default/",
    eval_template_version_restore:
        "model-hub/eval-templates/{template_id}/versions/{version_id}/restore/",

    // Composite
    composite_eval_create: "model-hub/eval-templates/create-composite/",
    composite_eval_detail: "model-hub/eval-templates/{template_id}/composite/",
    composite_eval_execute: "model-hub/eval-templates/{template_id}/composite/execute/",
    composite_eval_execute_adhoc: "model-hub/eval-templates/composite/execute-adhoc/",

    // Ground truth
    ground_truth_list: "model-hub/eval-templates/{template_id}/ground-truth/",
    ground_truth_upload: "model-hub/eval-templates/{template_id}/ground-truth/upload/",
    ground_truth_config: "model-hub/eval-templates/{template_id}/ground-truth-config/",
    ground_truth_mapping: "model-hub/ground-truth/{ground_truth_id}/mapping/",
    ground_truth_role_mapping: "model-hub/ground-truth/{ground_truth_id}/role-mapping/",
    ground_truth_data: "model-hub/ground-truth/{ground_truth_id}/data/",
    ground_truth_status: "model-hub/ground-truth/{ground_truth_id}/status/",
    ground_truth_search: "model-hub/ground-truth/{ground_truth_id}/search/",
    ground_truth_embed: "model-hub/ground-truth/{ground_truth_id}/embed/",
    ground_truth_delete: "model-hub/ground-truth/{ground_truth_id}/",

    // Playground
    eval_playground: "model-hub/eval-playground/",
    eval_playground_feedback: "model-hub/eval-playground/feedback/",
    eval_sdk_code: "model-hub/eval-sdk-code/",

    // ---------- Legacy dataset (kept for BC with existing callers) ----------
    dataset: "model-hub/develops",
    dataset_names: "model-hub/develops/get-datasets-names/",
    dataset_empty: "model-hub/develops/create-empty-dataset/",
    dataset_local: "model-hub/develops/create-dataset-from-local-file/",
    dataset_huggingface: "model-hub/develops/create-dataset-from-huggingface/",
    dataset_table: "model-hub/develops/{dataset_id}/get-dataset-table/",
    dataset_delete: "model-hub/develops/delete_dataset/",
    dataset_add_rows: "model-hub/develops/{dataset_id}/add_rows/",
    dataset_add_columns: "model-hub/develops/{dataset_id}/add_columns/",
    dataset_add_run_prompt_column: "model-hub/develops/add_run_prompt_column/",
    dataset_add_evaluation: "model-hub/develops/{dataset_id}/add_user_eval/",
    dataset_optimization_create: "model-hub/optimisation/create/",
    dataset_eval_stats: "model-hub/dataset/{dataset_id}/eval-stats/",

    // ---------- Prompt templates ----------
    generate_prompt: "model-hub/prompt-templates/generate-prompt/",
    improve_prompt: "model-hub/prompt-templates/improve-prompt/",
    run_template: "model-hub/prompt-templates/{template_id}/run_template/",
    create_template: "model-hub/prompt-templates/create-draft/",
    delete_template: "model-hub/prompt-templates/{template_id}",
    get_template_by_id: "model-hub/prompt-templates/{template_id}",
    list_templates: "model-hub/prompt-templates/",
    commit_template: "model-hub/prompt-templates/{template_id}/commit/",

    // ---------- Knowledge base ----------
    knowledge_base: "model-hub/knowledge-base/",
    knowledge_base_list: "model-hub/knowledge-base/list/",
    knowledge_base_files: "model-hub/knowledge-base/files/",
} as const;

export type RouteKey = keyof typeof Routes;
