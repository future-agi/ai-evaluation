/**
 * End-to-end integration tests for the revamped evals TypeScript SDK.
 *
 * Runs against a live backend (defaults to ws2-backend at
 * http://localhost:8003). Uses the test account to auto-fetch API keys.
 *
 * Run directly (skips jest mocks):
 *   npx ts-node -T -O '{"module":"commonjs"}' src/__tests__/evals_revamp.integration.test.ts
 * Or via jest (will pick up the .integration.test.ts suffix in jest config).
 */

import axios from 'axios';

const BASE_URL = process.env.FI_BASE_URL || 'http://localhost:8003';
const EMAIL = process.env.FI_TEST_EMAIL || 'kartik.nvj@futureagi.com';
const PASSWORD = process.env.FI_TEST_PASSWORD || 'test@123';

async function ensureAuth(): Promise<void> {
    if (process.env.FI_API_KEY && process.env.FI_SECRET_KEY) return;
    const tokenResp = await axios.post(`${BASE_URL}/accounts/token/`, {
        email: EMAIL,
        password: PASSWORD,
    });
    const access = tokenResp.data.access as string;
    const keysResp = await axios.get(`${BASE_URL}/accounts/keys/`, {
        headers: { Authorization: `Bearer ${access}` },
    });
    const data = keysResp.data.data as { api_key: string; secret_key: string };
    process.env.FI_API_KEY = data.api_key;
    process.env.FI_SECRET_KEY = data.secret_key;
    process.env.FI_BASE_URL = BASE_URL;
}

type Outcome = { name: string; ok: boolean; error?: string };
const results: Outcome[] = [];

async function test(name: string, fn: () => Promise<void>): Promise<void> {
    const t0 = Date.now();
    try {
        await fn();
        const ms = Date.now() - t0;
        console.log(`  PASS  ${name}  (${ms}ms)`);
        results.push({ name, ok: true });
    } catch (err: any) {
        const ms = Date.now() - t0;
        const msg = err?.message || String(err);
        console.log(`  FAIL  ${name}  (${ms}ms)\n    ${msg}`);
        if (err?.stack) console.log(err.stack);
        results.push({ name, ok: false, error: msg });
    }
}

function assertEq(actual: unknown, expected: unknown, label: string): void {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(
            `${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`
        );
    }
}

function assertTrue(cond: boolean, label: string): void {
    if (!cond) throw new Error(`${label} was false`);
}

function randomName(prefix: string): string {
    return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

async function main(): Promise<void> {
    console.log(`Using backend: ${BASE_URL}`);
    console.log(`Test user: ${EMAIL}\n`);

    await ensureAuth();

    // Import SDK AFTER auth so env vars are picked up by the clients.
    const { Evaluator, EvalTemplateManager } = await import('../index');

    const evaluator = new Evaluator();
    const manager = new EvalTemplateManager();

    let templateId: string | undefined;
    let compositeId: string | undefined;
    const toCleanup: string[] = [];

    await test('Evaluator.evaluate parses revamped snake_case response', async () => {
        const result = await evaluator.evaluate(
            'tone',
            { output: 'I absolutely love this product!' },
            {}
        );
        assertTrue(!!result, 'BatchRunResult returned');
        assertTrue(result.eval_results.length === 1, 'one eval result');
        const er = result.eval_results[0]!;
        assertEq(er.name, 'tone', 'name');
        assertTrue(!!er.eval_id, 'eval_id populated');
        assertTrue(!!er.output_type, 'output_type populated');
        assertTrue(er.output !== undefined, 'output populated');
        assertTrue(!!er.reason && er.reason.length > 0, 'reason populated');
    });

    await test('Evaluator.evaluate for pass/fail system eval', async () => {
        const result = await evaluator.evaluate(
            'is_json',
            { text: '{"ok": true, "n": 1}' },
            {}
        );
        const er = result.eval_results[0]!;
        assertTrue(!!er.output_type, `output_type present (${er.output_type})`);
    });

    await test('EvalTemplateManager.listTemplates paginated', async () => {
        const resp = await manager.listTemplates({ page: 0, pageSize: 5 });
        assertTrue('items' in resp, 'items key');
        assertTrue(Array.isArray(resp.items), 'items is array');
        assertTrue(resp.items.length <= 5, 'page size respected');
    });

    await test('EvalTemplateManager.listTemplates system filter', async () => {
        const resp = await manager.listTemplates({
            ownerFilter: 'system',
            pageSize: 3,
        });
        assertTrue(resp.items.length > 0, 'system has items');
        assertTrue(
            resp.items.every((i: any) => i.owner === 'system'),
            'all items owned by system'
        );
    });

    await test('createTemplate + getTemplate + updateTemplate', async () => {
        const name = randomName('sdk-ts-llm');
        const created = await manager.createTemplate({
            name,
            instructions: 'Does the {{output}} sound polite? Answer pass or fail.',
            evalType: 'llm',
            model: 'turing_large',
            outputType: 'pass_fail',
            passThreshold: 0.5,
            description: 'ts sdk smoke test',
        });
        assertEq(created.name, name, 'name matches');
        assertEq(created.version, 'V1', 'V1');
        templateId = created.id;
        toCleanup.push(templateId!);

        const detail = await manager.getTemplate(templateId!);
        assertEq(detail.id, templateId, 'detail id');
        assertEq(detail.eval_type, 'llm', 'eval_type');
        assertEq(detail.output_type, 'pass_fail', 'output_type');
        assertEq(detail.pass_threshold, 0.5, 'pass_threshold');

        const upd = await manager.updateTemplate(templateId!, {
            description: 'updated via ts sdk',
            pass_threshold: 0.7,
        });
        assertTrue(upd.updated === true, 'update updated=true');

        const detail2 = await manager.getTemplate(templateId!);
        assertEq(detail2.description, 'updated via ts sdk', 'description updated');
        assertEq(detail2.pass_threshold, 0.7, 'pass_threshold updated');
    });

    await test('Version CRUD', async () => {
        assertTrue(!!templateId, 'prior test created template');
        const list = await manager.listVersions(templateId!);
        assertTrue(list.versions.length >= 1, 'has V1');

        const v1Id = list.versions[list.versions.length - 1].id;

        const v2 = await manager.createVersion(templateId!);
        assertTrue(v2.version_number > 1, 'v2 number > 1');

        const setDef = await manager.setDefaultVersion(templateId!, v2.id);
        assertEq(setDef.is_default, true, 'is_default');

        const restored = await manager.restoreVersion(templateId!, v1Id);
        assertTrue(
            restored.version_number > v2.version_number,
            'restore bumps number'
        );
    });

    await test('Composite create + execute', async () => {
        const a = await manager.createTemplate({
            name: randomName('sdk-ts-child-a'),
            instructions: 'Rate the {{output}} for clarity from 0 to 1.',
            outputType: 'percentage',
        });
        const b = await manager.createTemplate({
            name: randomName('sdk-ts-child-b'),
            instructions: 'Rate the {{output}} for politeness from 0 to 1.',
            outputType: 'percentage',
        });
        toCleanup.push(a.id, b.id);

        const composite = await manager.createComposite({
            name: randomName('sdk-ts-composite'),
            childTemplateIds: [a.id, b.id],
            aggregationFunction: 'weighted_avg',
            compositeChildAxis: 'percentage',
            childWeights: { [a.id]: 1.0, [b.id]: 2.0 },
            description: 'ts sdk composite smoke test',
        });
        compositeId = composite.id;
        toCleanup.push(compositeId!);
        assertEq(composite.template_type, 'composite', 'template_type');
        assertEq(composite.children.length, 2, 'two children');

        const detail = await manager.getComposite(compositeId!);
        assertEq(detail.id, compositeId, 'detail id');
        assertEq(detail.children.length, 2, 'two children in detail');

        const run = await manager.executeComposite(compositeId!, {
            mapping: { output: "Hello, I hope you're having a wonderful day!" },
        });
        assertEq(run.composite_id, compositeId, 'exec composite id');
        assertEq(run.total_children, 2, 'total_children=2');
        assertTrue(
            run.completed_children + run.failed_children === 2,
            'every child accounted for'
        );
    });

    await test('Evaluator.submit returns execution handle and wait() completes', async () => {
        const { Evaluator: E } = await import('../index');
        const ev = new E();
        const handle = await ev.submit(
            'tone',
            { output: 'I absolutely love this product!' },
            {}
        );
        assertTrue(!!handle && typeof handle.wait === 'function', 'Execution returned');
        assertEq(handle.kind, 'eval', 'kind=eval');
        assertTrue(!!handle.id, 'execution id populated');
        assertTrue(
            handle.status === 'pending' || handle.status === 'processing',
            `initial status non-terminal (got ${handle.status})`
        );

        await handle.wait({ timeout: 120, pollInterval: 2 });
        assertEq(handle.status, 'completed', 'terminal status');
        assertTrue(handle.result !== null, 'result populated');
        const res = handle.result as any;
        assertEq(res.name, 'tone', 'result name');
        assertTrue(res.output !== undefined, 'result output');
    });

    await test('Evaluator.getExecution re-attaches by id', async () => {
        const handle = await evaluator.submit(
            'tone',
            { output: 'Have a great day!' },
            {}
        );
        const refetched = await evaluator.getExecution(handle.id);
        assertEq(refetched.id, handle.id, 'id matches');
        assertEq(refetched.kind, 'eval', 'kind=eval');
        await refetched.wait({ timeout: 120, pollInterval: 2 });
        assertEq(refetched.status, 'completed', 'completed');
        assertTrue(refetched.result !== null, 'result populated');
    });

    await test('Evaluator.submit with errorLocalizer=true completes', async () => {
        const handle = await evaluator.submit(
            'toxicity',
            { output: 'You are a worthless idiot, I hate you!' },
            { errorLocalizer: true }
        );
        await handle.wait({ timeout: 180, pollInterval: 3 });
        assertEq(handle.status, 'completed', 'completed');
        const res = handle.result as any;
        assertTrue(
            res && res.error_localizer_enabled === true,
            'error_localizer_enabled=true'
        );
    });

    await test('EvalTemplateManager.submitComposite returns handle and wait() completes', async () => {
        assertTrue(!!compositeId, 'prior test created composite');
        const handle = manager.submitComposite(compositeId!, {
            mapping: { output: "Thanks, have a lovely day!" },
        });
        assertTrue(!!handle && typeof handle.wait === 'function', 'Execution returned');
        assertEq(handle.kind, 'composite', 'kind=composite');
        assertTrue(!!handle.id, 'local execution id populated');
        assertTrue(
            handle.status === 'pending' || handle.status === 'processing',
            `initial status non-terminal (got ${handle.status})`
        );

        await handle.wait({ timeout: 180, pollInterval: 2 });
        assertEq(handle.status, 'completed', 'terminal status');
        const res = handle.result as any;
        assertTrue(res && typeof res === 'object', 'result is object');
        assertEq(res.composite_id, compositeId, 'composite_id matches');
        assertEq(res.total_children, 2, 'total_children=2');
    });

    await test('Cleanup created templates', async () => {
        for (const id of toCleanup) {
            try {
                await manager.deleteTemplate(id);
            } catch (err: any) {
                console.log(`    warning: failed to delete ${id}: ${err?.message}`);
            }
        }
    });

    const passed = results.filter((r) => r.ok).length;
    const failed = results.length - passed;
    console.log(`\nSummary: ${passed} passed, ${failed} failed, ${results.length} total`);
    if (failed > 0) process.exit(1);
}

// Jest-aware: if running under jest, wrap as a single test.
declare const jest: any;
if (typeof jest !== 'undefined') {
    // Reference Jest globals via any so this file also type-checks when
    // executed directly via ts-node (outside Jest).
    const jestTest = (globalThis as any).test as any;
    jestTest('evals revamp end-to-end', async () => {
        await main();
    }, 180_000);
} else {
    main().catch((err) => {
        console.error(err);
        process.exit(1);
    });
}
