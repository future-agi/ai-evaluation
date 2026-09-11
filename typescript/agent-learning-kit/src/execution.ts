/**
 * Execution handles for async eval and composite runs.
 *
 * An `Execution` is a lightweight view into a (possibly still-running)
 * eval on the backend (for single evals) or a background promise inside
 * the calling process (for composite evals, which the backend runs
 * synchronously).
 */

import { EvalResult } from './types';

export type ExecutionKind = 'eval' | 'composite';
export type ExecutionStatus =
    | 'pending'
    | 'processing'
    | 'completed'
    | 'failed';

const STATUS_MAP: Record<string, ExecutionStatus> = {
    pending: 'pending',
    PENDING: 'pending',
    processing: 'processing',
    PROCESSING: 'processing',
    running: 'processing',
    completed: 'completed',
    COMPLETED: 'completed',
    failed: 'failed',
    FAILED: 'failed',
};

export function normalizeStatus(raw: string | undefined | null): ExecutionStatus {
    if (!raw) return 'pending';
    return STATUS_MAP[raw] ?? (raw.toLowerCase() as ExecutionStatus);
}

export class ExecutionError extends Error {
    public readonly executionId: string;
    constructor(executionId: string, message?: string) {
        super(message ?? `Execution ${executionId} failed`);
        this.name = 'ExecutionError';
        this.executionId = executionId;
    }
}

export interface WaitOptions {
    /** Max total seconds to wait (default 300). */
    timeout?: number;
    /** Seconds between polls (default 2). */
    pollInterval?: number;
    /** If true (default), throw ExecutionError on terminal status="failed". */
    raiseOnFailure?: boolean;
}

type Refresher = () => Promise<Execution>;

const sleep = (ms: number): Promise<void> =>
    new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Handle to an in-flight or completed eval execution.
 *
 * For single evals, `id` is a server-side UUID resumable from any process
 * via `Evaluator.getExecution`. For composite evals, `id` is a client-side
 * UUID — the execution lives in a background Promise inside the calling
 * process, so the handle must be kept alive by the caller.
 */
export class Execution {
    public id: string;
    public kind: ExecutionKind;
    public status: ExecutionStatus;
    public result: EvalResult | Record<string, any> | null;
    public errorMessage: string | null;
    public errorLocalizer: Record<string, any> | null;

    // Closure that (re)fetches the latest state from the source.
    // Hidden from JSON.stringify via a non-enumerable property.
    private _refresher: Refresher | null;

    constructor(init: {
        id: string;
        kind: ExecutionKind;
        status?: ExecutionStatus;
        result?: EvalResult | Record<string, any> | null;
        errorMessage?: string | null;
        errorLocalizer?: Record<string, any> | null;
        refresher?: Refresher;
    }) {
        this.id = init.id;
        this.kind = init.kind;
        this.status = init.status ?? 'pending';
        this.result = init.result ?? null;
        this.errorMessage = init.errorMessage ?? null;
        this.errorLocalizer = init.errorLocalizer ?? null;
        this._refresher = init.refresher ?? null;
    }

    /** Internal: swap in a fresh refresher (used by Evaluator.getExecution). */
    public _setRefresher(fn: Refresher | null): void {
        this._refresher = fn;
    }

    public isDone(): boolean {
        return this.status === 'completed' || this.status === 'failed';
    }

    public async refresh(): Promise<Execution> {
        if (!this._refresher) return this;
        const updated = await this._refresher();
        this.status = updated.status;
        this.result = updated.result;
        this.errorMessage = updated.errorMessage;
        this.errorLocalizer = updated.errorLocalizer;
        return this;
    }

    public async wait(opts: WaitOptions = {}): Promise<Execution> {
        const timeout = (opts.timeout ?? 300) * 1000;
        const pollInterval = (opts.pollInterval ?? 2) * 1000;
        const raiseOnFailure = opts.raiseOnFailure ?? true;

        if (this.isDone()) {
            if (this.status === 'failed' && raiseOnFailure) {
                throw new ExecutionError(this.id, this.errorMessage ?? undefined);
            }
            return this;
        }

        const deadline = Date.now() + timeout;
        while (true) {
            await sleep(pollInterval);
            await this.refresh();
            if (this.isDone()) break;
            if (Date.now() > deadline) {
                throw new Error(
                    `Execution ${this.id} did not complete within ${opts.timeout ?? 300}s ` +
                        `(last status: ${this.status})`
                );
            }
        }

        if (this.status === 'failed' && raiseOnFailure) {
            throw new ExecutionError(this.id, this.errorMessage ?? undefined);
        }
        return this;
    }
}
