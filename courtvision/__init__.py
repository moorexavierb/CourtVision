"""CourtVision: NBA broadcast video to 10 Hz court tracking of every player and the ball."""
from .config import Config

__all__ = ["Config", "run"]
__version__ = "0.2.0"


def run(video, cfg=None, out_dir=None, write_outputs=True):
    """Shortcut for courtvision.pipeline.run (imported lazily: no torch until it is needed)."""
    from .pipeline import run as _run
    return _run(video, cfg, out_dir, write_outputs)
