from __future__ import annotations

from typing import Any, Dict, Optional
import os


def maybe_init_wandb(args, config: Dict[str, Any]) -> Optional[Any]:
    """Initialize a Weights & Biases run if a project is provided.

    Returns a wandb Run object or None if wandb is unavailable or not requested.
    """
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None

    # Respect mode (online/offline)
    mode = str(getattr(args, "wandb_mode", "online"))
    os.environ.setdefault("WANDB_MODE", mode)

    run_name = getattr(args, "wandb_run_name", None)
    group = getattr(args, "wandb_group", None)
    tags_raw = getattr(args, "wandb_tags", None)
    tags = None
    if isinstance(tags_raw, str) and len(tags_raw.strip()) > 0:
        tags = [t.strip() for t in tags_raw.split(",") if len(t.strip()) > 0]

    try:
        # Programmatic login if key provided and we are online
        if mode != "offline":
            key = getattr(args, "wandb_api_key", None) or os.getenv("WANDB_API_KEY")
            if isinstance(key, str) and len(key) > 0:
                try:
                    wandb.login(key=key, relogin=True)
                except Exception:
                    pass
        run = wandb.init(project=project, name=run_name, group=group, tags=tags, config=config, mode=mode)
        return run
    except Exception:
        return None


def log_report(run: Optional[Any], report: Dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(report)
    except Exception:
        pass


def finish(run: Optional[Any]) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


