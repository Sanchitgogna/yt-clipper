"""MCP server: lets Claude find and cut highlights itself.

Claude calls `find_highlights` to get candidate moments together with what is
said in each, picks the ones worth keeping (or adjusts their times), then calls
`export_clips` to cut them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import pipeline, source

BASE = Path.home() / "YTClipper"


def _work(url: str) -> Path:
    return BASE / "cache" / ("mcp-" + hashlib.sha1(url.encode()).hexdigest()[:10])


def build():
    try:
        from mcp.server.mcpserver import MCPServer   # mcp 2.x
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x

    mcp = MCPServer("ytclipper")

    @mcp.tool()
    def find_highlights(url: str, count: int = 8, min_seconds: float = 15,
                        max_seconds: float = 60, browser_login: str = "") -> dict:
        """Find the most exciting moments of a YouTube video or finished stream.

        Returns candidate clips (start/end in seconds, a score, and the transcript of
        what is said, when captions exist), best first. Nothing is cut yet: review the
        candidates, then call export_clips with the ranges to keep. `browser_login`
        (chrome, safari, firefox...) uses that browser's YouTube login if YouTube
        demands a sign-in.
        """
        s = pipeline.Settings(count=count, min_len=min_seconds, max_len=max_seconds,
                              auth=source.Auth(cookies_from_browser=browser_login or None))
        src, results = pipeline.find_url(url, _work(url), s)
        return {"title": src.title, "candidates": [
            {"start": r.start, "end": r.end, "score": round(r.score, 2), "transcript": r.text}
            for r in sorted(results, key=lambda r: -r.score)]}

    @mcp.tool()
    def export_clips(url: str, clips: list[dict], vertical: bool = False, captions: bool = False,
                     browser_login: str = "") -> list[str]:
        """Cut clips out of a video. `clips` is a list of {"start": s, "end": s} in seconds
        (from find_highlights, or any ranges you choose). `vertical` centre-crops to 9:16
        for Shorts/Reels/TikTok; `captions` burns the spoken words into the picture. Returns the paths of the written .mp4 files.
        """
        s = pipeline.Settings(vertical=vertical, captions=captions,
                              auth=source.Auth(cookies_from_browser=browser_login or None))
        work = _work(url)
        info = source.probe(url, work, s.auth)
        title = info.get("title") or "clip"
        src = source.Source(path=work / "none", title=title, url=url)
        if captions:  # the audio and YouTube captions supply the words
            src = source.fetch(url, work, auth=s.auth, audio_only=True, chat=False)
        results = [pipeline.Result(float(c["start"]), float(c["end"]), -i) for i, c in enumerate(clips)]
        out = BASE / "clips" / pipeline.slug(title)
        done = pipeline.export_sections(url, src, results, out, work, s)
        return [str(r.path) for r in done]

    return mcp


def main() -> None:
    build().run()
