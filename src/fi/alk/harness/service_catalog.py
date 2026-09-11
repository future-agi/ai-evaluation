"""Known service semantics layered on top of provider-neutral Compose discovery.

Compose remains the source of truth for *what* runs.  This catalog only supplies the small
amount of semantics Compose cannot express: the protocol associated with a port, conventional
configuration names, and a useful readiness path.  Unknown services are still supported as TCP
capabilities; adding a profile improves ergonomics without changing the lifecycle machinery.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceProfile:
    kind: str
    port: int
    protocol: str
    configuration_names: tuple[str, ...] = ()
    readiness_path: str = ""


_PROFILES = (
    ServiceProfile("postgres", 5432, "postgres", ("DATABASE_URL", "POSTGRES_URL")),
    ServiceProfile("mysql", 3306, "mysql", ("DATABASE_URL", "MYSQL_URL")),
    ServiceProfile(
        "clickhouse",
        8123,
        "clickhouse",
        ("CLICKHOUSE_URL", "CLICKHOUSE_HTTP_URL"),
        "/ping",
    ),
    ServiceProfile("clickhouse", 9000, "tcp", ("CLICKHOUSE_NATIVE_URL",)),
    ServiceProfile("redis", 6379, "redis", ("REDIS_URL",)),
    ServiceProfile("mongodb", 27017, "mongodb", ("MONGODB_URL", "MONGO_URL")),
    ServiceProfile("rabbitmq", 5672, "amqp", ("AMQP_URL", "RABBITMQ_URL")),
    ServiceProfile("rabbitmq", 15672, "http", ("RABBITMQ_MANAGEMENT_URL",)),
    ServiceProfile("kafka", 9092, "kafka", ("KAFKA_BOOTSTRAP_SERVERS", "KAFKA_URL")),
    ServiceProfile("nats", 4222, "nats", ("NATS_URL",)),
    ServiceProfile("nats", 8222, "http", ("NATS_MONITORING_URL",)),
    ServiceProfile(
        "minio", 9000, "s3", ("S3_ENDPOINT_URL", "MINIO_URL"), "/minio/health/ready"
    ),
    ServiceProfile("minio", 9001, "http", ("MINIO_CONSOLE_URL",)),
    ServiceProfile(
        "elasticsearch", 9200, "http", ("ELASTICSEARCH_URL", "SEARCH_URL"), "/"
    ),
    ServiceProfile("qdrant", 6333, "http", ("QDRANT_URL",), "/readyz"),
    ServiceProfile("qdrant", 6334, "grpc", ("QDRANT_GRPC_URL",)),
    ServiceProfile("neo4j", 7474, "http", ("NEO4J_HTTP_URL",)),
    ServiceProfile("neo4j", 7687, "bolt", ("NEO4J_URI", "NEO4J_URL")),
    ServiceProfile("livekit", 7880, "livekit", ("LIVEKIT_URL",)),
    ServiceProfile(
        "code-executor", 8000, "http", ("CODE_EXECUTOR_URL",), "/health"
    ),
    ServiceProfile("mcp", 8000, "mcp", ("MCP_URL", "MCP_SERVER_URL")),
)


_ALIASES = {
    "mongo": "mongodb",
    "opensearch": "elasticsearch",
    "redpanda": "kafka",
    "seaweedfs": "s3",
    "code_executor": "code-executor",
}


def profile_for(service: str, image: str, port: int) -> ServiceProfile:
    """Return semantics for one endpoint, retaining unknown endpoints as TCP."""
    haystack = f"{service} {image}".lower()
    for profile in _PROFILES:
        names = {
            profile.kind,
            *[key for key, value in _ALIASES.items() if value == profile.kind],
        }
        if profile.port == port and any(name in haystack for name in names):
            return profile
    # Port is a strong signal even when a private image has an opaque registry/name.
    candidates = [profile for profile in _PROFILES if profile.port == port]
    if len(candidates) == 1:
        return candidates[0]
    return ServiceProfile("service", port, "tcp")


def address(protocol: str, host: str, port: int) -> str:
    """Canonical non-secret connector value for a discovered endpoint."""
    scheme = {
        "clickhouse": "http",
        "s3": "http",
        "mcp": "http",
        "livekit": "ws",
    }.get(protocol, protocol)
    return f"{scheme}://{host}:{port}"


__all__ = ["ServiceProfile", "address", "profile_for"]
