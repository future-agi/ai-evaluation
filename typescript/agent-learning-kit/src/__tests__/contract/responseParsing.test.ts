/**
 * Pin response parsing — both legacy and revamped schemas decode.
 *
 * Mirror of python/tests/contract/test_response_parsing.py.
 */
import { AxiosResponse } from "axios";

import { EvalResponseHandler } from "../../evaluator";


function mockResponse(data: any, status = 200): AxiosResponse {
    return {
        data,
        status,
        statusText: "OK",
        headers: {},
        config: {} as any,
    } as AxiosResponse;
}


describe("EvalResponseHandler._parseSuccess", () => {
    test("revamp snake_case response decodes", () => {
        const payload = {
            status: true,
            result: [
                {
                    evaluations: [
                        {
                            name: "toxicity",
                            output: "Passed",
                            reason: "fine",
                            runtime: 1234,
                            output_type: "Pass/Fail",
                            eval_id: "uuid-1",
                        },
                    ],
                },
            ],
        };
        const res = EvalResponseHandler._parseSuccess(mockResponse(payload));
        expect(res.eval_results).toHaveLength(1);
        const r = res.eval_results[0]!;
        expect(r.name).toBe("toxicity");
        expect(r.output).toBe("Passed");
        expect(r.output_type).toBe("Pass/Fail");
        expect(r.eval_id).toBe("uuid-1");
    });

    test("empty result list returns empty batch (no crash)", () => {
        const res = EvalResponseHandler._parseSuccess(
            mockResponse({ status: true, result: [] })
        );
        expect(res.eval_results).toEqual([]);
    });

    test("unwrapped async eval result (no evaluations[] array)", () => {
        const payload = {
            status: true,
            result: [
                {
                    name: "toxicity",
                    output: "Pending",
                    reason: "processing",
                    runtime: 0,
                    output_type: "",
                    eval_id: "async-uuid",
                },
            ],
        };
        const res = EvalResponseHandler._parseSuccess(mockResponse(payload));
        expect(res.eval_results).toHaveLength(1);
        expect(res.eval_results[0]!.eval_id).toBe("async-uuid");
    });
});
