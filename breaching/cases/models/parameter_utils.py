"""Helpers for selecting shared/trainable parameters."""


def shared_parameters(model):
    """Return the parameter list exchanged between server and users.

    Models can override this by implementing a `shared_parameters` method.
    """
    if hasattr(model, "shared_parameters"):
        params = list(model.shared_parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            params = list(model.parameters())
    if len(params) == 0:
        raise ValueError("Model has no shared parameters.")
    return params
