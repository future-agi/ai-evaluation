def check(world, calls):
    return any(call.name == "lookup_account" for call in calls)
