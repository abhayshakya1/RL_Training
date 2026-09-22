import verifiers.legacy as vf


def load_environment(**kwargs) -> vf.Environment:
    """
    Load this environment.

    Environments typically return vf.SingleTurnEnv, vf.ToolEnv, etc.
    """
    raise NotImplementedError("Implement load_environment here.")
