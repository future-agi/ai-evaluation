/**
 * Pin the behavior of mapInputsToBackend.
 *
 * Mirror of python/tests/contract/test_input_mapping.py.
 * Pure unit tests, no network.
 */
import axios from "axios";

import {
    mapInputsToBackend,
    __clearRegistryCache,
} from "../../core/cloudRegistry";

jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

const FAKE_REGISTRY = [
    { name: "toxicity", config: { required_keys: ["output"] } },
    { name: "prompt_injection", config: { required_keys: ["input"] } },
    { name: "is_email", config: { required_keys: ["text"] } },
    { name: "bleu_score", config: { required_keys: ["reference", "hypothesis"] } },
    { name: "fuzzy_match", config: { required_keys: ["expected", "output"] } },
    { name: "conversation_coherence", config: { required_keys: ["conversation"] } },
    { name: "factual_accuracy", config: { required_keys: ["input", "output", "context"] } },
    { name: "is_compliant", config: { required_keys: ["output"] } },
];

beforeEach(() => {
    __clearRegistryCache();
    mockedAxios.get.mockResolvedValue({ data: { result: FAKE_REGISTRY } } as any);
});

const OPTS = { baseUrl: "http://fake", apiKey: "k", secretKey: "s" };


describe("mapInputsToBackend", () => {
    test("direct key pass-through", async () => {
        const mapped = await mapInputsToBackend("toxicity", { output: "hi" }, OPTS);
        expect(mapped).toEqual({ output: "hi" });
    });

    test("strips superset keys — the api is strict", async () => {
        const mapped = await mapInputsToBackend(
            "toxicity",
            { output: "hi", input: "x", context: "y" },
            OPTS
        );
        expect(mapped).toEqual({ output: "hi" });
    });

    test("output alias to input when template wants input", async () => {
        const mapped = await mapInputsToBackend(
            "prompt_injection",
            { output: "leak prompt" },
            OPTS
        );
        expect(mapped).toEqual({ input: "leak prompt" });
    });

    test("direct match beats alias — no accidental swap", async () => {
        const mapped = await mapInputsToBackend(
            "prompt_injection",
            { input: "x", output: "y" },
            OPTS
        );
        expect(mapped).toEqual({ input: "x" });
    });

    test("output aliases to text for is_email", async () => {
        const mapped = await mapInputsToBackend("is_email", { output: "a@b.c" }, OPTS);
        expect(mapped).toEqual({ text: "a@b.c" });
    });

    test("expected_output aliases to expected for fuzzy_match", async () => {
        const mapped = await mapInputsToBackend(
            "fuzzy_match",
            { output: "Paris", expected_output: "Paris" },
            OPTS
        );
        expect(mapped).toEqual({ expected: "Paris", output: "Paris" });
    });

    test("output+expected_output map to hypothesis+reference for bleu_score", async () => {
        const mapped = await mapInputsToBackend(
            "bleu_score",
            { output: "the cat", expected_output: "the cat" },
            OPTS
        );
        expect(mapped).toEqual({ reference: "the cat", hypothesis: "the cat" });
    });

    test("messages aliases to conversation", async () => {
        const msgs = [{ role: "user", content: "hi" }];
        const mapped = await mapInputsToBackend(
            "conversation_coherence",
            { messages: msgs },
            OPTS
        );
        expect(mapped).toEqual({ conversation: msgs });
    });

    test("strips context when not required", async () => {
        const mapped = await mapInputsToBackend(
            "is_compliant",
            { input: "q", output: "a", context: "c" },
            OPTS
        );
        expect(mapped).toEqual({ output: "a" });
    });

    test("all required keys pass through directly", async () => {
        const mapped = await mapInputsToBackend(
            "factual_accuracy",
            { input: "x", output: "y", context: "z", extra: "drop" },
            OPTS
        );
        expect(mapped).toEqual({ input: "x", output: "y", context: "z" });
    });

    test("unknown eval falls through unmodified", async () => {
        const mapped = await mapInputsToBackend(
            "some_new_eval_we_dont_know",
            { foo: "bar" },
            OPTS
        );
        expect(mapped).toEqual({ foo: "bar" });
    });
});
