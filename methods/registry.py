from collections.abc import Callable


SUPPORTED_MODELS = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm")


def get_train_handler(model_name: str) -> Callable:
    if model_name == "lf_cbm":
        from methods.lf import train_lf_cbm

        return train_lf_cbm
    if model_name in ("salf_cbm", "savlg_cbm"):
        raise NotImplementedError(
            f"{model_name} is planned in this unified codebase but has not been ported yet."
        )
    raise ValueError(f"Unsupported model_name: {model_name}")

