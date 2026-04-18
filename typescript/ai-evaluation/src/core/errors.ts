/**
 * SDK error hierarchy. Vendored from the legacy @future-agi/sdk package so
 * ai-evaluation owns its own error types.
 */

export class SDKException extends Error {
    public customMessage?: string;
    public cause?: Error;

    constructor(message?: string, cause?: Error) {
        super(
            message ||
                (cause
                    ? `An SDK error occurred, caused by: ${cause.message}`
                    : "An unknown error occurred in the SDK.")
        );
        this.name = this.constructor.name;
        this.customMessage = message;
        this.cause = cause;
        Object.setPrototypeOf(this, new.target.prototype);
    }

    getMessage(): string {
        if (this.customMessage) return this.customMessage;
        if (this.cause) return `An SDK error occurred, caused by: ${this.cause.message}`;
        return "An unknown error occurred in the SDK.";
    }

    getErrorCode(): string {
        return "UNKNOWN_SDK_ERROR";
    }
}


export class MissingAuthError extends SDKException {
    public missingApiKey: boolean;
    public missingSecretKey: boolean;

    constructor(fiApiKey?: string, fiSecretKey?: string, cause?: Error) {
        super(undefined, cause);
        this.missingApiKey = !fiApiKey;
        this.missingSecretKey = !fiSecretKey;
    }

    getMessage(): string {
        const missing: string[] = [];
        if (this.missingApiKey) missing.push("'fi_api_key'");
        if (this.missingSecretKey) missing.push("'fi_secret_key'");
        return (
            "FI Client could not obtain credentials. Pass fi_api_key and fi_secret_key " +
            "directly or set FI_API_KEY / FI_SECRET_KEY env vars.\n" +
            `Missing: ${missing.join(", ")}`
        );
    }

    getErrorCode(): string {
        return "MISSING_FI_CLIENT_AUTHENTICATION";
    }
}


export class InvalidAuthError extends SDKException {
    constructor(message?: string, cause?: Error) {
        super(
            message ||
                "Invalid FI Client Authentication, please check your API key and secret key.",
            cause
        );
    }

    getErrorCode(): string {
        return "INVALID_FI_CLIENT_AUTHENTICATION";
    }
}


export class InvalidValueType extends SDKException {
    constructor(
        public valueName: string,
        public value: unknown,
        public correctType: string,
        cause?: Error
    ) {
        super(
            `${valueName} with value ${JSON.stringify(value)} is of type ${typeof value}, but expected from ${correctType}.`,
            cause
        );
    }

    getErrorCode(): string {
        return "INVALID_VALUE_TYPE";
    }
}


export class MissingRequiredKey extends SDKException {
    constructor(public fieldName: string, public missingKey: string, cause?: Error) {
        super(
            `Missing required key '${missingKey}' in ${fieldName}. Please check your configuration or API documentation.`,
            cause
        );
    }

    getErrorCode(): string {
        return "MISSING_REQUIRED_KEY";
    }
}


export class MissingRequiredConfigForEvalTemplate extends SDKException {
    constructor(public missingKey: string, public evalTemplateName: string, cause?: Error) {
        super(
            `Missing required config '${missingKey}' for eval template '${evalTemplateName}'.`,
            cause
        );
    }

    getErrorCode(): string {
        return "MISSING_EVAL_TEMPLATE_CONFIG";
    }
}


export class DatasetNotFoundError extends SDKException {
    constructor(message?: string, cause?: Error) {
        super(message || "No existing dataset found for current dataset name.", cause);
    }
    getErrorCode(): string {
        return "DATASET_NOT_FOUND";
    }
}

export class RateLimitError extends SDKException {
    constructor(message?: string, cause?: Error) {
        super(message || "Rate limit exceeded.", cause);
    }
    getErrorCode(): string {
        return "RATE_LIMIT_EXCEEDED";
    }
}

export class ServerError extends SDKException {
    constructor(message?: string, cause?: Error) {
        super(message || "Internal server error.", cause);
    }
    getErrorCode(): string {
        return "SERVER_ERROR";
    }
}

export class ServiceUnavailableError extends SDKException {
    constructor(message?: string, cause?: Error) {
        super(message || "Service unavailable, please try again later.", cause);
    }
    getErrorCode(): string {
        return "SERVICE_UNAVAILABLE";
    }
}
