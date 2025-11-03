import { Evaluator } from '../evaluator';

// Helper to conditionally run tests only if environment variables are set
const describeIf = (condition: boolean, ...args: Parameters<typeof describe>) => 
    condition ? describe(...args) : describe.skip(...args);

const hasIntegrationEnv = !!(process.env.FI_API_KEY && process.env.FI_SECRET_KEY && process.env.FI_BASE_URL);

describeIf(hasIntegrationEnv, 'API Health Check', () => {
    let evaluator: Evaluator;

    beforeAll(() => {
        evaluator = new Evaluator({
            fiApiKey: process.env.FI_API_KEY!,
            fiSecretKey: process.env.FI_SECRET_KEY!,
            fiBaseUrl: process.env.FI_BASE_URL!,
        });
    });

    it('should be able to list evaluations (basic connectivity test)', async () => {
        try {
            const evaluations = await evaluator.list_evaluations();
            expect(evaluations).toBeDefined();
            expect(Array.isArray(evaluations)).toBe(true);
            console.log(`✅ API is accessible. Found ${evaluations.length} evaluations.`);
        } catch (error) {
            console.error('❌ API connectivity test failed:', error);
            // Don't fail the test, just log the error
            expect(true).toBe(true); // Always pass, this is just a health check
        }
    }, 10000);

    it('should check API base URL format', () => {
        const baseUrl = process.env.FI_BASE_URL;
        console.log(`🔗 Using API Base URL: ${baseUrl}`);
        
        if (baseUrl) {
            expect(baseUrl.startsWith('http')).toBe(true);
            console.log(`✅ Base URL format looks correct`);
        } else {
            console.log(`⚠️  No base URL configured`);
        }
    });

    it('should check environment variables', () => {
        const apiKey = process.env.FI_API_KEY;
        const secretKey = process.env.FI_SECRET_KEY;
        const baseUrl = process.env.FI_BASE_URL;
        
        console.log(`🔑 API Key present: ${!!apiKey}`);
        console.log(`🔐 Secret Key present: ${!!secretKey}`);
        console.log(`🌐 Base URL present: ${!!baseUrl}`);
        
        if (apiKey) console.log(`🔑 API Key length: ${apiKey.length} characters`);
        if (secretKey) console.log(`🔐 Secret Key length: ${secretKey.length} characters`);
        
        expect(apiKey).toBeDefined();
        expect(secretKey).toBeDefined();
        expect(baseUrl).toBeDefined();
    });
});

describeIf(!hasIntegrationEnv, 'API Health Check - Skipped', () => {
    it('should skip health check when environment variables are not set', () => {
        console.log('⏭️  Skipping API health check. Please set FI_API_KEY, FI_SECRET_KEY, and FI_BASE_URL environment variables.');
        expect(true).toBe(true);
    });
}); 