import axios, { AxiosError, AxiosInstance, AxiosResponse } from "axios";

import { AUTH_ENVVAR_NAME, DEFAULT_SETTINGS, getBaseUrl } from "./constants";
import {
    DatasetNotFoundError,
    InvalidAuthError,
    MissingAuthError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
} from "./errors";
import { BoundedExecutor } from "./executor";
import { RequestConfig } from "./types";

/**
 * Subclass to parse + validate a typed response.
 */
export abstract class ResponseHandler<T = any, U = any> {
    static parse<T, U>(response: AxiosResponse, handlerClass: typeof ResponseHandler): T | U {
        if (!response || response.status !== 200) {
            handlerClass._handleError(response);
        }
        return handlerClass._parseSuccess(response);
    }

    static _parseSuccess(_response: AxiosResponse): any {
        throw new Error("_parseSuccess must be implemented by subclass");
    }

    static _handleError(response: AxiosResponse): never {
        const status = response?.status || 500;
        let message: string | undefined;
        if (response?.data) {
            const d = response.data as any;
            message = d.message || d.detail || d.result;
            if (!message) {
                try {
                    message = JSON.stringify(d);
                } catch {
                    /* ignore */
                }
            }
        }
        if (!message) {
            const url = response?.config?.url ? ` – ${response.config.url}` : "";
            message =
                response?.statusText && response.statusText.trim().length > 0
                    ? response.statusText
                    : `HTTP ${status}${url}`;
        }

        switch (status) {
            case 401:
            case 403:
                throw new InvalidAuthError(message);
            case 404:
                throw new DatasetNotFoundError(message);
            case 429:
                throw new RateLimitError(message);
            case 503:
                throw new ServiceUnavailableError(message);
            case 500:
            case 502:
            case 504:
                throw new ServerError(message);
            default:
                throw new Error(`HTTP ${status}: ${message}`);
        }
    }
}

export interface HttpClientConfig {
    baseUrl?: string;
    defaultHeaders?: Record<string, string>;
    timeout?: number;
    maxQueue?: number;
    maxWorkers?: number;
    retryAttempts?: number;
    retryDelay?: number;
}

export class HttpClient {
    protected readonly _baseUrl: string;
    protected readonly _axiosInstance: AxiosInstance;
    protected readonly _executor: BoundedExecutor;
    protected readonly _defaultTimeout: number;
    private readonly _defaultRetryAttempts: number;
    private readonly _defaultRetryDelay: number;

    constructor(config: HttpClientConfig = {}) {
        this._baseUrl = (config.baseUrl || getBaseUrl()).replace(/\/$/, "");
        this._defaultTimeout = config.timeout || DEFAULT_SETTINGS.TIMEOUT;
        this._defaultRetryAttempts = config.retryAttempts || 3;
        this._defaultRetryDelay = config.retryDelay || 1000;

        this._axiosInstance = axios.create({
            baseURL: this._baseUrl,
            timeout: this._defaultTimeout,
            headers: {
                "Content-Type": "application/json",
                "User-Agent": "@future-agi/ai-evaluation",
                ...config.defaultHeaders,
            },
            maxRedirects: 5,
            validateStatus: () => true,
        });

        this._executor = new BoundedExecutor(
            config.maxQueue || DEFAULT_SETTINGS.MAX_QUEUE,
            config.maxWorkers || DEFAULT_SETTINGS.MAX_WORKERS
        );
    }

    async request<T = any>(
        config: RequestConfig,
        responseHandler?: typeof ResponseHandler
    ): Promise<T | AxiosResponse> {
        const requestConfig: any = {
            method: config.method,
            url: config.url,
            headers: config.headers,
            params: config.params,
            data: config.json || config.data,
            timeout: config.timeout || this._defaultTimeout,
        };

        if (config.files && Object.keys(config.files).length > 0) {
            const formData = new FormData();
            Object.entries(config.files).forEach(([key, file]) => {
                formData.append(key, file as any);
            });
            if (config.data) {
                Object.entries(config.data).forEach(([key, value]) => {
                    formData.append(key, value as any);
                });
            }
            requestConfig.data = formData;
            requestConfig.headers = {
                ...requestConfig.headers,
                "Content-Type": "multipart/form-data",
            };
        }

        const retryAttempts = config.retry_attempts || this._defaultRetryAttempts;
        const retryDelay = config.retry_delay || this._defaultRetryDelay;

        return this._executor.submit(async () => {
            for (let attempt = 0; attempt < retryAttempts; attempt++) {
                try {
                    const response = await this._axiosInstance.request(requestConfig);
                    if (responseHandler) {
                        return ResponseHandler.parse(response, responseHandler);
                    }
                    if (response.status >= 400) {
                        ResponseHandler._handleError(response);
                    }
                    return response;
                } catch (error) {
                    // Non-retryable errors
                    if (
                        error instanceof DatasetNotFoundError ||
                        error instanceof InvalidAuthError ||
                        (error instanceof AxiosError && error.response?.status === 401)
                    ) {
                        throw error;
                    }
                    if (attempt === retryAttempts - 1) {
                        if (error instanceof AxiosError && error.response) {
                            ResponseHandler._handleError(error.response);
                        }
                        throw error;
                    }
                    await new Promise((res) => setTimeout(res, retryDelay * Math.pow(2, attempt)));
                }
            }
            throw new Error("Unexpected end of retry loop");
        });
    }

    async close(): Promise<void> {
        await this._executor.shutdown(true);
    }

    get baseUrl(): string {
        return this._baseUrl;
    }

    get defaultTimeout(): number {
        return this._defaultTimeout;
    }
}

export interface APIKeyAuthConfig extends HttpClientConfig {
    fiApiKey?: string;
    fiSecretKey?: string;
    fiBaseUrl?: string;
}

export class APIKeyAuth extends HttpClient {
    protected _fiApiKey?: string;
    protected _fiSecretKey?: string;

    constructor(config: APIKeyAuthConfig = {}) {
        const fiApiKey = config.fiApiKey || process.env[AUTH_ENVVAR_NAME.API_KEY];
        const fiSecretKey = config.fiSecretKey || process.env[AUTH_ENVVAR_NAME.SECRET_KEY];
        if (!fiApiKey || !fiSecretKey) {
            throw new MissingAuthError(fiApiKey, fiSecretKey);
        }

        super({
            ...config,
            baseUrl: config.fiBaseUrl || config.baseUrl,
            defaultHeaders: {
                "X-Api-Key": fiApiKey,
                "X-Secret-Key": fiSecretKey,
                ...config.defaultHeaders,
            },
        });

        this._fiApiKey = fiApiKey;
        this._fiSecretKey = fiSecretKey;
    }

    get fiApiKey(): string | undefined {
        return this._fiApiKey;
    }

    get fiSecretKey(): string | undefined {
        return this._fiSecretKey;
    }

    get headers(): Record<string, string> {
        return {
            "X-Api-Key": this._fiApiKey!,
            "X-Secret-Key": this._fiSecretKey!,
        };
    }
}
