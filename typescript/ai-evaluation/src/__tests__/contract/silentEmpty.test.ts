/**
 * Regression test for the silent-empty bug — api 4xx must surface a failed
 * EvalResult with readable error text, never an empty batch.
 *
 * Mirror of python/tests/contract/test_silent_empty.py.
 */
import { Evaluator } from "../../evaluator";


describe("Evaluator silent-empty regression", () => {
    const ORIGINAL_ENV = { ...process.env };

    beforeEach(() => {
        process.env.FI_API_KEY = "fake";
        process.env.FI_SECRET_KEY = "fake";
        process.env.FI_BASE_URL = "http://fake";
    });

    afterEach(() => {
        process.env = { ...ORIGINAL_ENV };
        jest.restoreAllMocks();
    });

    test("api 4xx returns failed EvalResult, not empty batch", async () => {
        const ev = new Evaluator();

        // Stub the HTTP request path to throw like a 400 would.
        jest.spyOn(ev, "request").mockRejectedValueOnce(
            new Error("Evaluation failed with a 400 Bad Request")
        );

        const batch = await ev.evaluate("toxicity", { output: "x" }, { modelName: "turing_flash" });

        expect(batch.eval_results).toHaveLength(1);
        const r = batch.eval_results[0]!;
        expect(r.output).toBeNull();
        expect(r.reason).toContain("400");
        expect(r.name).toBe("toxicity");
    });

    test("api 5xx returns failed EvalResult", async () => {
        const ev = new Evaluator();

        jest.spyOn(ev, "request").mockRejectedValueOnce(
            new Error("Error in evaluation: 500")
        );

        const batch = await ev.evaluate("toxicity", { output: "x" }, { modelName: "turing_flash" });

        expect(batch.eval_results).toHaveLength(1);
        expect(batch.eval_results[0]!.reason).toContain("500");
    });
});
