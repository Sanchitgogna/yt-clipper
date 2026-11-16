"""Find and clip the best moments of YouTube videos and livestreams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import live, pipeline, source


def _settings(args) -> pipeline.Settings:
    return pipeline.Settings(
        count=args.count, min_len=args.min_len, max_len=args.max_len,
        max_height=args.max_height, facecam=not getattr(args, "no_facecam", False),
        scenes=args.scenes, chat=not args.no_chat, fast=not args.full_download,
        captions=not getattr(args, "no_captions", False), claude=not args.no_claude,
        auth=source.Auth(args.cookies, args.cookies_from_browser))


def _progress(stage: str, frac: float | None) -> None:
    pct = f" {frac * 100:3.0f}%" if frac is not None else ""
    print(f"\r\033[K{stage}{pct}", end="", file=sys.stderr, flush=True)


def _table(results: list[pipeline.Result]) -> None:
    print(file=sys.stderr)
    if not results:
        print("no standout moments found")
    for r in sorted(results, key=lambda r: -r.score):
        rated = f"Claude {r.rating}/10" if r.rating else f"score {r.score:.2f}"
        print(f"{live.fmt(r.start)} - {live.fmt(r.end)}  ({r.end - r.start:4.0f}s)  {rated}"
              + (f"  {r.title}" if r.title else "") + (f"  {r.path}" if r.path else ""))


def _run_with(args, s: pipeline.Settings, cut: bool) -> list[pipeline.Result]:
    work = Path(args.work_dir)
    try:
        if source.is_url(args.input):
            if cut:
                return pipeline.clip_url(args.input, Path(args.out), work, s, _progress)[1]
            return pipeline.find_url(args.input, work, s, _progress)[1]
        path = Path(args.input)
        if not path.exists():
            sys.exit(f"no such file: {path}")
        if cut:
            return pipeline.clip_file(path, Path(args.out), s, _progress)[1]
        return pipeline.find(source.Source(path=path, title=path.stem), s, _progress)
    except Exception as e:  # yt-dlp raises many types; show the reason, not a traceback
        sys.exit(f"\nfailed: {e}")


def _run(args, cut: bool) -> list[pipeline.Result]:
    s = _settings(args)
    results = _run_with(args, s, cut)
    for note in s.notes:
        print(f"\nnote: {note}", file=sys.stderr)
    return results


def cmd_analyze(args) -> None:
    results = _run(args, cut=False)
    if args.json:
        print(json.dumps([{"start": r.start, "end": r.end, "score": r.score, "title": r.title,
                           "rating": r.rating, "reason": r.reason, "text": r.text}
                          for r in results], indent=2))
    else:
        _table(results)


def cmd_clip(args) -> None:
    _table(_run(args, cut=True))


def cmd_live(args) -> None:
    detector = live.LiveDetector(threshold=args.threshold, pre_roll=args.pre_roll,
                                 post_roll=args.post_roll, cooldown=args.cooldown,
                                 warmup=args.warmup)
    print(f"watching {args.input} (Ctrl-C to stop) ...", file=sys.stderr)
    settings = pipeline.Settings(captions=not args.no_captions, facecam=not args.no_facecam)
    clips = live.run(args.input, Path(args.out), Path(args.work_dir), detector, settings=settings)
    print(f"{len(clips)} clip(s) written to {args.out}")


def cmd_ui(args) -> None:
    from . import server
    server.serve(Path(args.dir).expanduser(), args.port, open_browser=not args.no_browser)


def cmd_mcp(args) -> None:
    try:
        from . import mcp_server
        mcp_server.main()
    except ImportError:
        sys.exit("the MCP server needs an extra package: pip install 'ytclipper[mcp]'")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ytclipper", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, finder: bool):
        sp.add_argument("input", help="YouTube URL or local media file")
        sp.add_argument("--work-dir", default=".ytclipper", help="where downloads/segments go")
        if finder:
            sp.add_argument("-n", "--count", type=int, default=5, help="max clips (default 5)")
            sp.add_argument("--min-len", type=float, default=15, help="min clip seconds")
            sp.add_argument("--max-len", type=float, default=60, help="max clip seconds")
            sp.add_argument("--max-height", type=int, default=1080, help="download resolution cap")
            sp.add_argument("--no-chat", action="store_true", help="skip chat replay download")
            sp.add_argument("--cookies", help="cookies.txt of your YouTube login (if YouTube asks to sign in)")
            sp.add_argument("--cookies-from-browser", metavar="BROWSER",
                            help="use your login from a browser, e.g. chrome or firefox")
            sp.add_argument("--no-claude", action="store_true",
                            help="do not ask Claude to choose and title clips (used when ANTHROPIC_API_KEY is set)")
            sp.add_argument("--scenes", action="store_true",
                            help="also use scene-cut detection (downloads the whole video: slower)")
            sp.add_argument("--full-download", action="store_true",
                            help="download the whole video instead of only the clips")

    def looks(sp):
        sp.add_argument("--no-captions", action="store_true", help="leave the animated captions off")
        sp.add_argument("--no-facecam", action="store_true",
                        help="do not look for a streamer's webcam to put on top")

    a = sub.add_parser("analyze", help="list the best moments without cutting")
    common(a, True)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("clip", help="find the best moments of a video and cut them")
    common(c, True)
    c.add_argument("-o", "--out", default="clips")
    looks(c)
    c.set_defaults(func=cmd_clip)

    l = sub.add_parser("live", help="watch a livestream and clip moments as they happen")
    common(l, False)
    l.add_argument("-o", "--out", default="clips")
    looks(l)
    l.add_argument("--threshold", type=float, default=4.0, help="spike sensitivity (lower = more clips)")
    l.add_argument("--pre-roll", type=float, default=20, help="seconds kept before the moment")
    l.add_argument("--post-roll", type=float, default=10, help="seconds kept after the moment")
    l.add_argument("--cooldown", type=float, default=30, help="min seconds between clips")
    l.add_argument("--warmup", type=float, default=60,
                   help="seconds to learn the stream's normal level before clipping")
    l.set_defaults(func=cmd_live)

    u = sub.add_parser("ui", help="open the app in your browser")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--dir", default="~/YTClipper", help="where clips are saved")
    u.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    u.set_defaults(func=cmd_ui)

    m = sub.add_parser("mcp", help="run as an MCP server for Claude")
    m.set_defaults(func=cmd_mcp)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
