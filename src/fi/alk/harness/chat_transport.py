"""Construct the repository chat transport used by execution and readiness."""

from urllib.parse import urljoin

from fi.simulate.agent.wrappers.http import HTTPAgentWrapper


def create_chat_wrapper(
    interface,
    base,
    *,
    timeout=30.0,
    secret_values=None,
    metadata=None,
    wrapper_type=HTTPAgentWrapper,
    total_timeout=None,
):
    options = dict(
        endpoint=urljoin(
            base.rstrip("/") + "/",
            ("/invoke" if interface.kind == "callable" else interface.path).lstrip("/"),
        ),
        protocol="fi.alk" if interface.kind == "callable" else interface.protocol,
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
            failure_rule=interface.failure_rule.model_dump()
            if interface.failure_rule
            else None,
            secret_values=secret_values,
            completed_tool_events=interface.completed_tool_events.model_dump()
            if interface.completed_tool_events
            else None,
        )
        if total_timeout is not None:
            options["total_timeout"] = total_timeout
    return wrapper_type(**options)
