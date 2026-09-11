export enum HttpMethod {
    GET = "GET",
    POST = "POST",
    PUT = "PUT",
    DELETE = "DELETE",
    PATCH = "PATCH",
}

export interface RequestConfig {
    method: HttpMethod;
    url: string;
    headers?: Record<string, string>;
    params?: Record<string, any>;
    files?: Record<string, any>;
    data?: Record<string, any>;
    json?: Record<string, any>;
    timeout?: number;
    retry_attempts?: number;
    retry_delay?: number;
}
