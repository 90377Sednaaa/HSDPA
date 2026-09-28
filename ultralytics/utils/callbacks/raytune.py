# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    try:
        session_fn = getattr(ray.train._internal.session, "_get_session", None) or getattr(ray.train._internal.session, "get_session", None)
        if session_fn and session_fn():
            metrics = trainer.metrics
            metrics["epoch"] = trainer.epoch
            session.report(metrics)
    except Exception:
        pass


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
