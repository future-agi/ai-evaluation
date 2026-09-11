/**
 * Ensure the Routes enum has every entry referenced by callers.
 *
 * Mirror of python/tests/contract/test_routes_enum.py.
 */
import { Routes } from "../../core/routes";


const REQUIRED_BY_EVALUATOR = [
    "evaluatev2",
    "get_eval_templates",
    "get_eval_result",
    "evaluate_pipeline",
    "configure_evaluations",
    "evaluate",
];

const REQUIRED_BY_MANAGER = [
    "eval_template_list",
    "eval_template_list_charts",
    "eval_template_detail",
    "eval_template_create_v2",
    "eval_template_update_v2",
    "eval_template_delete",
    "eval_template_bulk_delete",
    "eval_template_duplicate",
    "eval_template_usage",
    "eval_template_feedback_list",
    "eval_template_version_list",
    "eval_template_version_create",
    "eval_template_version_set_default",
    "eval_template_version_restore",
    "composite_eval_create",
    "composite_eval_detail",
    "composite_eval_execute",
    "ground_truth_list",
    "ground_truth_upload",
    "ground_truth_config",
    "ground_truth_mapping",
    "ground_truth_role_mapping",
    "ground_truth_data",
    "ground_truth_status",
    "ground_truth_search",
    "ground_truth_embed",
    "ground_truth_delete",
    "eval_playground",
];


describe("Routes enum completeness", () => {
    test.each(REQUIRED_BY_EVALUATOR)(
        "Evaluator route '%s' exists",
        (name) => {
            expect(Routes).toHaveProperty(name);
        }
    );

    test.each(REQUIRED_BY_MANAGER)(
        "EvalTemplateManager route '%s' exists",
        (name) => {
            expect(Routes).toHaveProperty(name);
        }
    );
});
