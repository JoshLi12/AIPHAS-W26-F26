from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from winter26.new_data_pipeline import run_new_data_pipeline

__all__ = ["run_new_data_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "run_new_data_pipeline":
        from winter26.new_data_pipeline import run_new_data_pipeline

        return run_new_data_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
