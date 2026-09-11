export const SECRET_KEY_ENVVAR_NAME = "FI_SECRET_KEY";
export const API_KEY_ENVVAR_NAME = "FI_API_KEY";

export const AUTH_ENVVAR_NAME = {
    SECRET_KEY: SECRET_KEY_ENVVAR_NAME,
    API_KEY: API_KEY_ENVVAR_NAME,
};

export const DEFAULT_TIMEOUT = 200;
export const DEFAULT_MAX_WORKERS = 8;
export const DEFAULT_MAX_QUEUE = 5000;

export const DEFAULT_SETTINGS = {
    TIMEOUT: DEFAULT_TIMEOUT,
    MAX_WORKERS: DEFAULT_MAX_WORKERS,
    MAX_QUEUE: DEFAULT_MAX_QUEUE,
};

export function getBaseUrl(): string {
    return process.env.FI_BASE_URL || "https://api.futureagi.com";
}

// Back-compat alias for the snake_case helper.
export const get_base_url = getBaseUrl;
