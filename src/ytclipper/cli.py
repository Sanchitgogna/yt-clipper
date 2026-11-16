from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import live, pipeline, scoring, source


def _load(args) -> source.Source:
    if source.is_url(args.input):
        print(f"downloading {args.input} ...", file=sys.stderr)
        try:
            return source.fetch(args.input, Path(args.work_dir), args.max_height, chat=not args.no_chat,
                                cookies=args.cookies, cookies_from_browser=args.cookies_from_browser)
        except Exception as e:  # yt-dlp raises many types; show the reason, not a traceback
            sys.exit(f"download failed: {e}")
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return source.Source(path=path, title=path.stem)


def _find(args) -> tuple[source.Source, list[scoring.Clip], dict]:
    src = _load(args)
    print("analysing ...", file=sys.stderr)
    score, signals = pipeline.analyze(src, scenes=not args.no_scenes)
    clips = scoring.pick_clips(score, args.count, args.min_len, args.max_len)
    print(f"signals used: {', '.join(signals)}", file=sys.stderr)
    return src, clips, signals


def _table(clips: list[scoring.Clip]) -> None:
    if not clips:
        print("no standout moments found")
    for c in sorted(clips, key=lambda c: -c.score):
        print(f"{live.fmt(c.start)} - {live.fmt(c.end)}  ({c.length:4.0f}s)  score {c.score:.2f}")


def cmd_analyze(args) -> None:
    _, clips, _ = _find(args)
    if args.json:
        print(json.dumps([vars(c) for c in clips], indent=2))
    else:
        _table(clips)


def cmd_clip(args) -> None:
    src, clips, _ = _find(args)
    _table(clips)
    for path in pipeline.export(src, clips, Path(args.out), vertical=args.vertical):
        print(f"wrote {path}")


def cmd_live(args) -> None:
    detector = live.LiveDetector(threshold=args.threshold, pre_roll=args.pre_roll,
                                 post_roll=args.post_roll, cooldown=args.cooldown,
                                 warmup=args.warmup)
    print(f"watching {args.input} (Ctrl-C to stop) ...", file=sys.stderr)
    clips = live.run(args.input, Path(args.out), Path(args.work_dir), detector, vertical=args.vertical)
    print(f"{len(clips)} clip(s) written to {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ytclipper", description=__doc__ or
                                "Find and clip the best moments of YouTube videos and livestreams.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, finder: bool):
        sp.add_argument("input", help="YouTube URL or local media file")
        sp.add_argument("--work-dir", default=".ytclipper", help="where downloads/segments go")
        if finder:
            sp.add_argument("-n", "--count", type=int, default=5, help="max clips (default 5)")
            sp.add_argument("--min-len", type=float, default=15, help="min clip seconds")
            sp.add_argument("--max-len", type=float, default=60, help="max clip seconds")
            sp.add_argument("--max-height", type=int, default=720, help="download resolution cap")
            sp.add_argument("--no-chat", action="store_true", help="skip chat replay download")
            sp.add_argument("--cookies", help="cookies.txt of your YouTube login (if YouTube asks to sign in)")
            sp.add_argument("--cookies-from-browser", metavar="BROWSER",
                            help="use your login from a browser, e.g. chrome or firefox")
            sp.add_argument("--no-scenes", action="store_true", help="skip scene-cut detection (faster)")

    a = sub.add_parser("analyze", help="list the best moments without cutting")
    common(a, True)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("clip", help="find the best moments of a video and cut them")
    common(c, True)
    c.add_argument("-o", "--out", default="clips")
    c.add_argument("--vertical", action="store_true", help="centre-crop to 9:16 (1080x1920)")
    c.set_defaults(func=cmd_clip)

    l = sub.add_parser("live", help="watch a livestream and clip moments as they happen")
    common(l, False)
    l.add_argument("-o", "--out", default="clips")
    l.add_argument("--vertical", action="store_true")
    l.add_argument("--threshold", type=float, default=4.0, help="spike sensitivity (lower = more clips)")
    l.add_argument("--pre-roll", type=float, default=20, help="seconds kept before the moment")
    l.add_argument("--post-roll", type=float, default=10, help="seconds kept after the moment")
    l.add_argument("--cooldown", type=float, default=30, help="min seconds between clips")
    l.add_argument("--warmup", type=float, default=60,
                   help="seconds to learn the stream's normal level before clipping")
    l.set_defaults(func=cmd_live)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
