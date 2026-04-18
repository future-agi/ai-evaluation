/**
 * Cloud eval registry — source of truth for eval metadata.
 *
 * Mirrors python/fi/evals/core/cloud_registry.py. Fetches the full template
 * list from `/sdk/api/v1/get-evals/` once per (baseUrl, apiKey) tuple and
 * caches the result. Exposes helpers that turn user-supplied kwargs into
 * the exact key set the backend will accept, so the SDK never drifts from
 * the backend when evals are added/renamed.
 */
import axios from "axios";

type RegistryCacheKey = string; // `${baseUrl}|${apiKeyPrefix}`
type RegistryEntry = Record<string, any>;

const _CACHE = new Map<RegistryCacheKey, Record<string, RegistryEntry>>();

/**
 * Aliases user kwargs can use → canonical backend keys.
 * Direct key match in user_inputs always wins over an alias lookup.
 */
const KEY_ALIASES: Record<string, readonly string[]> = {
    // Bidirectional output↔input — some evals only accept one (e.g.
    // prompt_injection wants `input`, toxicity wants `output`).
    output: ["output", "response", "answer", "generated", "input"],
    input: ["input", "query", "question", "prompt_input", "output"],
    context: ["context", "contexts"],
    expected: ["expected", "expected_output", "expected_response", "ground_truth"],
    expected_value: ["expected_value", "expected_output", "expected_response", "ground_truth"],
    generated_value: ["generated_value", "output", "response", "answer"],
    reference: ["reference", "expected_output", "expected_response", "ground_truth"],
    hypothesis: ["hypothesis", "output", "response"],
    text: ["text", "output", "content"],
    conversation: ["conversation", "messages"],
    prompt: ["prompt", "instructions", "system_prompt"],
    system_prompt: ["system_prompt", "prompt", "instructions"],
    image: ["image", "image_url", "input_image_url"],
    caption: ["caption", "output"],
    instruction: ["instruction", "prompt"],
    instructions: ["instructions", "prompt"],
    images: ["images", "image_urls", "input_image_urls"],
    input_pdf: ["input_pdf", "pdf"],
    json_content: ["json_content", "json", "expected_output"],
    input_audio: ["input_audio", "audio"],
    audio: ["audio", "input_audio"],
    generated_audio: ["generated_audio", "audio", "output"],
    generated_transcript: ["generated_transcript", "transcript", "output"],
};


function cacheKey(baseUrl: string, apiKey?: string): RegistryCacheKey {
    return `${baseUrl.replace(/\/$/, "")}|${(apiKey || "").slice(0, 12)}`;
}

export interface LoadRegistryOptions {
    baseUrl: string;
    apiKey?: string;
    secretKey?: string;
    forceRefresh?: boolean;
}

export async function loadRegistry(
    opts: LoadRegistryOptions
): Promise<Record<string, RegistryEntry>> {
    const key = cacheKey(opts.baseUrl, opts.apiKey);
    if (!opts.forceRefresh && _CACHE.has(key)) {
        return _CACHE.get(key)!;
    }

    const url = `${opts.baseUrl.replace(/\/$/, "")}/sdk/api/v1/get-evals/`;
    const headers: Record<string, string> = {};
    if (opts.apiKey) headers["X-Api-Key"] = opts.apiKey;
    if (opts.secretKey) headers["X-Secret-Key"] = opts.secretKey;

    try {
        const resp = await axios.get(url, { headers, timeout: 30000 });
        const items: any[] = resp.data?.result || [];
        const byName: Record<string, RegistryEntry> = {};
        for (const item of items) {
            if (item?.name) byName[item.name] = item;
        }
        _CACHE.set(key, byName);
        return byName;
    } catch (err) {
        // Soft-fail — let the caller fall back to pass-through so the
        // api can surface its own validation error.
        console.warn(`Failed to load cloud eval registry from ${url}:`, err);
        return {};
    }
}


export async function getTemplateInfo(
    name: string,
    opts: LoadRegistryOptions
): Promise<RegistryEntry | undefined> {
    const reg = await loadRegistry(opts);
    return reg[name];
}


export async function getRequiredKeys(
    name: string,
    opts: LoadRegistryOptions
): Promise<string[]> {
    const info = await getTemplateInfo(name, opts);
    return info?.config?.required_keys ?? [];
}


/**
 * Build the exact payload the backend expects for an eval, by:
 *   1. Looking up the eval's required_keys from the cached registry.
 *   2. Taking each required key from userInputs directly if present.
 *   3. Otherwise resolving via known aliases.
 *   4. Dropping any keys the backend doesn't accept (the api is strict).
 *
 * If the eval isn't in the registry (unknown name, registry load failed),
 * falls back to passing userInputs through unmodified.
 */
export async function mapInputsToBackend(
    name: string,
    userInputs: Record<string, any>,
    opts: LoadRegistryOptions
): Promise<Record<string, any>> {
    const required = await getRequiredKeys(name, opts);
    if (required.length === 0) {
        return { ...userInputs };
    }

    const mapped: Record<string, any> = {};
    for (const key of required) {
        if (key in userInputs) {
            mapped[key] = userInputs[key];
            continue;
        }
        const aliases = KEY_ALIASES[key] || [];
        for (const alias of aliases) {
            if (alias !== key && alias in userInputs) {
                mapped[key] = userInputs[alias];
                break;
            }
        }
    }
    return mapped;
}


/** Synchronous variant — uses cached data. Throws if not preloaded. */
export function mapInputsToBackendSync(
    name: string,
    userInputs: Record<string, any>,
    baseUrl: string,
    apiKey?: string
): Record<string, any> {
    const cached = _CACHE.get(cacheKey(baseUrl, apiKey));
    if (!cached || !cached[name]) {
        return { ...userInputs };
    }
    const required: string[] = cached[name]?.config?.required_keys ?? [];
    if (!required.length) {
        return { ...userInputs };
    }
    const mapped: Record<string, any> = {};
    for (const key of required) {
        if (key in userInputs) {
            mapped[key] = userInputs[key];
            continue;
        }
        const aliases = KEY_ALIASES[key] || [];
        for (const alias of aliases) {
            if (alias !== key && alias in userInputs) {
                mapped[key] = userInputs[alias];
                break;
            }
        }
    }
    return mapped;
}


/** Clear cache (exposed for tests). */
export function __clearRegistryCache(): void {
    _CACHE.clear();
}
