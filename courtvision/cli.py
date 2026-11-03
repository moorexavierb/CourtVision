"""``courtvision game.mp4 --config configs/default.yaml``: writes output/<video>/tracking.parquet."""
from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="courtvision", description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="output folder (default: output.dir/<video name>)")
    ap.add_argument("--hz", type=float, default=None, help="samples per second (default from config: 10)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .config import Config
    from .pipeline import run
    cfg = Config.load(a.config)
    if a.hz:
        cfg.output.hz = a.hz
    if a.no_video:
        cfg.output.write_video = False
    r = run(a.video, cfg, a.out)
    print(r.diagnostics["dataset"])


if __name__ == "__main__":
    main()
