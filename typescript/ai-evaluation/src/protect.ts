import {
    APIKeyAuth,
    HttpMethod,
    RequestConfig,
    Routes,
    InvalidAuthError,
    SDKException,
    InvalidValueType,
    MissingRequiredKey,
} from '@future-agi/sdk';
import { EvalResponseHandler, Evaluator } from './evaluator';
import { Templates } from './templates';
import { BatchRunResult } from './types';

const PROTECT_FLASH_ID = "76";

export class Protect {
    public evaluator: Evaluator;
    private metric_map: Record<string, any>;

    constructor(options: {
        fiApiKey?: string;
        fiSecretKey?: string;
        fiBaseUrl?: string;
        evaluator?: Evaluator;
    } = {}) {
        if (options.evaluator) {
            this.evaluator = options.evaluator;
        } else {
            const fiApiKey = process.env.FI_API_KEY || options.fiApiKey;
            const fiSecretKey = process.env.FI_SECRET_KEY || options.fiSecretKey;
            const fiBaseUrl = process.env.FI_BASE_URL || options.fiBaseUrl;

            if (!fiApiKey || !fiSecretKey) {
                throw new InvalidAuthError("API key or secret key is missing for Protect initialization.");
            }
            
            this.evaluator = new Evaluator({ fiApiKey, fiSecretKey, fiBaseUrl });
        }

        this.metric_map = {
            "Toxicity": Templates.Toxicity,
            "Tone": Templates.Tone,
            "Sexism": Templates.Sexist,
            "Prompt Injection": Templates.PromptInjection,
            "Data Privacy": Templates.DataPrivacyCompliance,
        };
    }

    private async _check_rule_sync(
        rule: Record<string, any>,
        testCase: { input: string; call_type: string },
        timeoutSeconds: number
    ): Promise<[string, boolean, string | undefined, string | undefined]> {
        const templateInfo = this.metric_map[rule.metric];
        const templateConfig: Record<string, any> = { call_type: "protect" };

        if (rule.metric === "Data Privacy") {
            templateConfig.check_internet = false;
        }

        const payload = {
            inputs: [testCase],
            config: {
                [templateInfo.eval_id]: templateConfig
            },
        };

        const timeoutMs = Math.max(0, timeoutSeconds * 1000);
        const evalResult = await this.evaluator.request(
            {
                method: HttpMethod.POST,
                url: `${this.evaluator.baseUrl}/${Routes.evaluate}`,
                json: payload,
                timeout: timeoutMs
            },
            EvalResponseHandler,
        ) as BatchRunResult;

        let reasonText: string | undefined = undefined;

        if (evalResult.eval_results && evalResult.eval_results[0]) {
            const result = evalResult.eval_results[0]!;
            const detectedValues = result.data as any[] || [];
            let shouldTrigger = false;
            if (rule.type === "any") {
                shouldTrigger = detectedValues.some((value: any) => rule.contains?.includes(value));
            } else if (rule.type === "all") {
                shouldTrigger = rule.contains?.every((value: any) => detectedValues.includes(value)) ?? false;
            }

            if (shouldTrigger) {
                const message = rule.action;
                if (rule._internal_reason_flag) {
                    reasonText = result.reason;
                }
                return [rule.metric, true, message, reasonText];
            }
        }

        return [rule.metric, false, undefined, undefined];
    }

    private async _process_rules_batch(
        rules: Record<string, any>[],
        testCase: { input: string; call_type: string },
        remainingTime: number // in seconds
    ): Promise<[string[], string[], string[], string[] | undefined, string | undefined]> {
        let failureMessages: string[] = [];
        let completedRules: string[] = [];
        let uncompletedRules: string[] = [];
        let failureReasons: string[] = [];
        let failedRule: string | undefined = undefined;

        const timeoutPromise = new Promise<'timeout'>((resolve) =>
            setTimeout(() => resolve("timeout"), remainingTime * 1000)
        );

        const rulePromises = rules.map(rule =>
            this._check_rule_sync(rule, testCase, remainingTime)
                .then(value => ({ status: 'fulfilled' as const, value, metric: rule.metric }))
                .catch(reason => ({ status: 'rejected' as const, reason, metric: rule.metric }))
        );

        const raceResult = await Promise.race([Promise.all(rulePromises), timeoutPromise]);

        if (raceResult === "timeout") {
            uncompletedRules = rules.map(r => r.metric);
            return [failureMessages, completedRules, uncompletedRules, failureReasons, failedRule];
        }

        for (const result of raceResult) {
            if (result.status === 'fulfilled') {
                const [metric, triggered, message, reason_text] = result.value;
                completedRules.push(metric);
                if (triggered && !failedRule) {
                    failedRule = metric;
                    failureMessages.push(message!);
                    if (reason_text) {
                        failureReasons.push(reason_text);
                    }
                }
            } else {
                console.error(`Rule ${result.metric} failed with error:`, result.reason);
            }
        }
        
        const allMetrics = rules.map(r => r.metric);
        uncompletedRules = allMetrics.filter(m => !completedRules.includes(m));

        return [failureMessages, completedRules, uncompletedRules, failureReasons, failedRule];
    }

    public async protect(
        inputs: string,
        protectRules: Record<string, any>[] | null = null,
        action: string = "Response cannot be generated as the input fails the checks",
        reason: boolean = false,
        timeout: number = 30000, // milliseconds
        useFlash: boolean = false
    ): Promise<Record<string, any>> {
        const timeoutSeconds = timeout / 1000.0;
        let protectRulesCopy: Record<string, any>[] = protectRules ? JSON.parse(JSON.stringify(protectRules)) : [];

        if (useFlash && protectRulesCopy.length === 0) {
            protectRulesCopy = [{ metric: "Toxicity" }];
        } else if (useFlash) {
            console.log("Note: When using ProtectFlash, Rules are not considered as it performs binary harmful/not harmful classification only.");
        }

        if (typeof inputs !== 'string') {
            throw new InvalidValueType("inputs", inputs, "string");
        }

        const input_text = inputs;
        if (!input_text.trim()) {
            throw new InvalidValueType("inputs", input_text, "non-empty string or string with non-whitespace characters");
        }

        const inputsList = [input_text];
        
        if (useFlash) {
            const testCase = { input: inputsList[0], call_type: "protect" };
            const templateInfo = this.metric_map[protectRulesCopy[0].metric];
            const payload = {
                inputs: [testCase],
                config: { [PROTECT_FLASH_ID]: { call_type: "protect" } },
                protect_flash: true
            };
            
            const response = await this.evaluator.request(
                {
                    method: HttpMethod.POST,
                    url: `${this.evaluator.baseUrl}/${Routes.evaluate}`,
                    json: payload,
                    timeout: timeoutSeconds * 1000
                },
                EvalResponseHandler,
            ) as BatchRunResult;

            if (response?.eval_results?.[0]) {
                const result = response.eval_results[0];
                const isHarmful = result.failure;
                return {
                    status: isHarmful ? "failed" : "passed",
                    completed_rules: ["ProtectFlash"],
                    uncompleted_rules: [],
                    failed_rule: isHarmful ? "ProtectFlash" : null,
                    messages: isHarmful ? (protectRulesCopy[0].action || action) : inputsList[0],
                    reasons: isHarmful ? "Content detected as harmful." : "All checks passed",
                    time_taken: result.runtime ? result.runtime / 1000 : 0,
                };
            } else {
                return { status: "error", messages: "Evaluation failed", completed_rules: [], uncompleted_rules: ["ProtectFlash"], failed_rule: null, reasons: "No evaluation results returned", time_taken: 0 };
            }
        }
        
        const testCases = inputsList.map(input_text => ({ input: input_text, call_type: "protect" }));

        // Validate rules
        if (protectRulesCopy.length === 0) {
            throw new InvalidValueType("protect_rules", protectRulesCopy, "non-empty list");
        }

        const validMetrics = new Set(Object.keys(this.metric_map));
        const validTypes = new Set(['any', 'all']);

        for (let i = 0; i < protectRulesCopy.length; i++) {
            const rule = protectRulesCopy[i];
            if (typeof rule !== 'object' || rule === null) {
                throw new InvalidValueType(`Rule at index ${i}`, rule, "dictionary");
            }
            if (!rule.metric) {
                throw new MissingRequiredKey(`Rule at index ${i}`, 'metric');
            }
            if (!validMetrics.has(rule.metric)) {
                throw new InvalidValueType(`metric in Rule at index ${i}`, rule.metric, `one of ${[...validMetrics]}`);
            }
            
            const isToneMetric = rule.metric === "Tone";

            if (isToneMetric) {
                if (!rule.contains) {
                    throw new MissingRequiredKey(`Rule for Tone metric at index ${i}`, "contains");
                }
                if (!Array.isArray(rule.contains) || rule.contains.length === 0) {
                    throw new InvalidValueType(`'contains' in Tone rule at index ${i}`, rule.contains, "non-empty list");
                }
                if (rule.type && !validTypes.has(rule.type)) {
                    throw new InvalidValueType(`'type' in Tone rule at index ${i}`, rule.type, `one of ${[...validTypes]}`);
                }
                if (!rule.type) {
                    rule.type = "any"; // Default
                }
            } else {
                if (rule.contains) {
                     throw new SDKException(`'contains' should not be specified for ${rule.metric} metric at index ${i}. Provide it only for 'Tone' metric.`);
                }
                 if (rule.type) {
                    throw new SDKException(`'type' should not be specified for ${rule.metric} metric at index ${i}. Provide it only for 'Tone' metric.`);
                }
                rule.contains = ["Failed"];
                rule.type = "any";
            }

            rule._internal_reason_flag = reason;
            if (!rule.action) rule.action = action;
        }

        const startTime = Date.now();
        let allFailureMessages: string[] = [];
        let allCompletedRules: string[] = [];
        let allUncompletedRules: string[] = [];
        let allFailureReasons: string[] = [];
        let failedRule: string | undefined = undefined;

        for (const testCase of testCases) {
            const BATCH_SIZE = 5;
            for (let i = 0; i < protectRulesCopy.length; i += BATCH_SIZE) {
                const elapsedSeconds = (Date.now() - startTime) / 1000;
                const remainingTime = Math.max(0, timeoutSeconds - elapsedSeconds);
                if (remainingTime <= 0) {
                    allUncompletedRules.push(...protectRulesCopy.slice(i).map((r: Record<string, any>) => r.metric));
                    break;
                }

                const rulesBatch = protectRulesCopy.slice(i, i + BATCH_SIZE);
                const [messages, completed, uncompleted, fReasons, fRule] = await this._process_rules_batch(rulesBatch, testCase, remainingTime);
                
                allCompletedRules.push(...completed);
                allUncompletedRules.push(...uncompleted);
                if (fReasons) allFailureReasons.push(...fReasons);
                
                if (fRule) {
                    failedRule = fRule;
                    allFailureMessages = messages;
                    break; 
                }
            }
            if (failedRule) break;
        }

        const finalProcessingDurationSeconds = (Date.now() - startTime) / 1000;

        const status = failedRule ? "failed" : "passed";
        const baseResult: any = {
            status,
            completed_rules: [...new Set(allCompletedRules)],
            uncompleted_rules: [...new Set(allUncompletedRules)],
            failed_rule: failedRule || null,
            messages: status === 'failed' ? allFailureMessages[0] : inputsList[0],
            time_taken: finalProcessingDurationSeconds,
        };

        if (reason) {
            baseResult.reasons = status === 'failed'
                ? (allFailureReasons[0] || "A protection rule was triggered.")
                : "All checks passed";
        }

        return baseResult;
    }
}

/**
 * Convenience function to evaluate input strings against protection rules.
 */
export const protect = (
    inputs: string,
    protectRules: Record<string, any>[] | null,
    action?: string,
    reason?: boolean,
    timeout?: number,
    useFlash?: boolean
): Promise<Record<string, any>> => {
    const protectClient = new Protect();
    return protectClient.protect(inputs, protectRules, action, reason, timeout, useFlash);
};