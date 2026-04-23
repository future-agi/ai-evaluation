import { setTimeout as sleep } from "timers/promises";

/**
 * ThreadPoolExecutor-equivalent that blocks on submit() when the work queue is full.
 */
export class BoundedExecutor {
    private workers = new Set<Promise<any>>();
    private queue: Array<() => Promise<any>> = [];
    private isShutdown = false;
    private semaphore: number;
    private readonly maxWorkers: number;

    constructor(bound: number, maxWorkers: number) {
        this.semaphore = bound + maxWorkers;
        this.maxWorkers = maxWorkers;
    }

    async submit<T>(fn: (...args: any[]) => Promise<T>, ...args: any[]): Promise<T> {
        if (this.isShutdown) {
            throw new Error("Executor has been shutdown");
        }
        await this.acquireSemaphore();

        return new Promise<T>((resolve, reject) => {
            const task = async () => {
                try {
                    resolve(await fn(...args));
                } catch (err) {
                    reject(err);
                } finally {
                    this.releaseSemaphore();
                }
            };

            if (this.workers.size < this.maxWorkers) {
                this.startWorker(task);
            } else {
                this.queue.push(task);
            }
        });
    }

    async shutdown(wait = true): Promise<void> {
        this.isShutdown = true;
        if (wait) {
            await Promise.all(Array.from(this.workers));
        }
    }

    private async acquireSemaphore(): Promise<void> {
        while (this.semaphore <= 0) {
            await sleep(1);
        }
        this.semaphore--;
    }

    private releaseSemaphore(): void {
        this.semaphore++;
    }

    private startWorker(task: () => Promise<any>): void {
        const workerPromise = this.runWorker(task);
        this.workers.add(workerPromise);
        workerPromise.finally(() => this.workers.delete(workerPromise));
    }

    private async runWorker(initialTask: () => Promise<any>): Promise<void> {
        let currentTask: (() => Promise<any>) | undefined = initialTask;
        while (currentTask && !this.isShutdown) {
            await currentTask();
            currentTask = this.queue.shift();
        }
    }
}
