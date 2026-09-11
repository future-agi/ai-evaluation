from fi.alk.harness.contract import AgentContract, validate_contract


def test_an_import_entrypoint_without_module_and_callable_is_rejected_at_contract_time():
    """Bundling compiles a binding from exactly these two fields, so a half-filled entry is dead.

    Observed on a dev run: the model recorded `lookup_policy` as `construct` with no module or
    callable, the contract was accepted, and the job died three runtime-validation attempts later
    with `contract_tool_entry_incomplete`. Catching it here tells the model while it can still fix
    the entry.
    """
    from fi.alk.harness.contract import AgentContract, validate_contract

    contract = AgentContract.model_validate(
        {
            "agent": "hotel",
            "real_use_cases": ["book a room"],
            "tools": [{"name": "lookup_policy", "args": []}],
            "tool_entrypoints": [{"tool": "lookup_policy", "mode": "construct"}],
        }
    )
    problems = validate_contract(contract)
    assert any("lookup_policy" in p and "needs-module-and-callable" in p for p in problems), problems


def test_a_complete_entrypoint_and_an_unreachable_one_both_pass():
    from fi.alk.harness.contract import AgentContract, validate_contract

    contract = AgentContract.model_validate(
        {
            "agent": "hotel",
            "real_use_cases": ["book a room"],
            "tools": [{"name": "a", "args": []}, {"name": "b", "args": []}],
            "tool_entrypoints": [
                {"tool": "a", "mode": "construct", "module": "pkg.mod", "callable": "Klass.method"},
                {"tool": "b", "mode": "unreachable"},
            ],
        }
    )
    assert not [p for p in validate_contract(contract) if "needs-module-and-callable" in p]
