import { Evaluator, list_evaluations } from '../evaluator';

// Increase default timeout since network calls may be slow
jest.setTimeout(30_000);

// Helper to decide if we have credentials for running end-to-end
const hasCredentials = Boolean(process.env.FI_API_KEY && process.env.FI_SECRET_KEY);

// Use describe.skip if credentials are missing so test run passes quickly in CI without secrets
const describeMaybe = hasCredentials ? describe : describe.skip;

describeMaybe('Evaluator – end-to-end (real network)', () => {
    const evaluator = new Evaluator({
        fiApiKey: process.env.FI_API_KEY,
        fiSecretKey: process.env.FI_SECRET_KEY,
        fiBaseUrl: process.env.FI_BASE_URL, // optional – falls back to prod
        timeout: 25_000,
    });

    it('should evaluate "factual_accuracy" successfully', async () => {
        const inputs = {
            input: 'What is the capital of France?',
            output: 'Paris is the capital of France',
        };

        const result = await evaluator.evaluate('factual_accuracy', inputs, {
            modelName: 'turing_flash',
        });

        expect(result).toBeDefined();
        expect(Array.isArray(result.eval_results)).toBe(true);
        expect(result.eval_results.length).toBeGreaterThan(0);
    });

    it('should list available evaluations', async () => {
        const evaluations = await evaluator.list_evaluations();

        expect(evaluations).toBeDefined();
        expect(Array.isArray(evaluations)).toBe(true);
        expect(evaluations.length).toBeGreaterThan(0);
    });
}); 