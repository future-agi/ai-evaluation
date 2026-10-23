"""Construct the repository chat transport used by execution and readiness."""

from urllib.parse import urljoin

from fi.simulate.agent.wrappers.http import HTTPAgentWrapper


def create_chat_wrapper(
    interface,
    base,
    *,
    timeout=30.0,
    secret_values=None,
    local_auth_values=None,
    metadata=None,
    wrapper_type=HTTPAgentWrapper,
    total_timeout=None,
):
    setup_secrets = dict(interface.setup_secrets)
    if interface.local_auth:
        secret_values = dict(secret_values or {})
        auth = interface.local_auth
        auth.reject_collisions(secret_values)
        token = (local_auth_values or {}).get(auth.server_environment)
        if not token:
            raise ValueError("runtime_local_auth_token_unavailable")
        setup_secrets[auth.template_variable] = auth.server_environment
        secret_values[auth.server_environment] = token
    options = dict(
        endpoint=urljoin(
            base.rstrip("/") + "/",
            ("/invoke" if interface.kind in {"callable", "command"} else interface.path).lstrip("/"),
        ),
        protocol="fi.alk" if interface.kind in {"callable", "command"} else interface.protocol,
        model=interface.model,
        include_tools=interface.include_tools,
        request_template=interface.request_template,
        response_path=interface.response_path,
        setup_requests=[
            item.model_dump(mode="python") for item in interface.setup_requests
        ],
        timeout=timeout,
        metadata=metadata,
    )
    if interface.kind != "websocket":
        options.update(
            headers_template=interface.headers_template,
            setup_secrets=setup_secrets,
            completed_tools=interface.completed_tools.model_dump()
            if interface.completed_tools
            else None,
            failure_rule=interface.failure_rule.model_dump()
            if interface.failure_rule
            else None,
            secret_values=secret_values,
            completed_tool_events=interface.completed_tool_events.model_dump()
            if interface.completed_tool_events
            else None,
        )
        if interface.local_auth:
            options["private_response_values"] = local_auth_values
        if total_timeout is not None:
            options["total_timeout"] = total_timeout
    return wrapper_type(**options)
