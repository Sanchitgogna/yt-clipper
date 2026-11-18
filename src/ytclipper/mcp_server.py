"""MCP server: Claude, in the Claude app, finds and cuts the highlights itself.

Runs on the user's own computer under their Claude subscription, so no API key is
needed. Downloads can take minutes, longer than a client waits for one tool call,
so work runs as background jobs: a tool starts one, and `check_job` waits up to
about 40 seconds for it before reporting progress.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import pipeline, source

BASE = Path.home() / "YTClipper"
WAIT = 40          # seconds check_job holds on before answering "still running"

INSTRUCTIONS = """YT Clipper turns YouTube videos and finished livestreams into vertical
1080x1920 clips for Shorts, Reels and TikTok, with animated captions and the streamer's
facecam on top when there is one. Everything runs on the user's computer.

How to use it:
1. Call find_moments with the link. It returns a job id at once.
2. Call check_job with that id until its status is "done" (each call waits up to ~40 s;
   long videos can take several minutes, so keep checking and tell the user the stage).
3. You get candidate moments with the transcript of what is said in each. You are the
   editor: pick the ones that work as standalone clips (a complete joke, a strong
   reaction, a surprising claim, a story with a payoff; not housekeeping, sponsor reads
   or fragments). Candidates with no transcript are loud or busy moments with no speech.
   Briefly tell the user which you picked and why.
4. Call make_clips with the job id and your picks, each with a short, specific title
   (at most 60 characters). You may adjust start/end within a candidate to tighten it.
5. Call check_job on the new job id until done, then give the user the file paths.
   open_clips_folder shows them in Finder.

If YouTube asks to sign in, retry with browser_login set to the browser the user is
logged in to YouTube with (chrome, safari, firefox, edge or brave)."""


@dataclass
class Job:
    id: str
    kind: str                         # "find" or "make"
    url: str
    status: str = "running"           # running | done | error
    stage: str = "Starting"
    progress: float | None = None
    started: float = field(default_factory=time.time)
    error: str = ""
    notes: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    src: source.Source | None = None
    candidates: list[pipeline.Result] = field(default_factory=list)
    auth: source.Auth = field(default_factory=source.Auth)
    done: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict:
        out = {"job_id": self.id, "status": self.status, "stage": self.stage,
               "elapsed_seconds": round(time.time() - self.started)}
        if self.progress is not None and self.status == "running":
            out["percent"] = round(self.progress * 100)
        if self.notes:
            out["notes"] = self.notes
        if self.error:
            out["error"] = self.error
        return out | self.result


JOBS: dict[str, Job] = {}


def _work(url: str) -> Path:
    return BASE / "cache" / ("mcp-" + hashlib.sha1(url.encode()).hexdigest()[:10])


def _auth(browser_login: str, job: Job) -> source.Auth:
    """The browser login to use, or none if it cannot be read (with a note saying why)."""
    auth = source.Auth(cookies_from_browser=browser_login.strip().lower() or None)
    if auth.cookies_from_browser and source.browser_login_problem(auth):
        job.notes.append(source.login_help(auth.cookies_from_browser))
        return source.Auth()
    return auth


def _start(kind: str, url: str, target) -> Job:
    job = Job(uuid.uuid4().hex[:8], kind, url)
    JOBS[job.id] = job

    def run() -> None:
        try:
            target(job)
            job.status, job.stage, job.progress = "done", "Done", None
        except Exception as e:
            job.status, job.error = "error", source.clean_error(e)[:600]
        finally:
            job.done.set()

    threading.Thread(target=run, daemon=True).start()
    return job


def _progress(job: Job):
    def report(stage: str, frac: float | None) -> None:
        job.stage, job.progress = stage, frac
    return report


def _fmt(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _find(job: Job, count: int, min_seconds: float, max_seconds: float, browser_login: str) -> None:
    job.stage = "Checking the link"
    job.auth = _auth(browser_login, job)
    s = pipeline.Settings(count=count, min_len=min_seconds, max_len=max_seconds,
                          claude=False, auth=job.auth)  # Claude in the chat is the judge
    if source.is_url(job.url):
        info = source.probe(job.url, _work(job.url), job.auth)
        if info.get("is_live"):
            raise RuntimeError("This stream is live right now. Live clipping runs in the YT Clipper "
                               "app (double-click YTClipper.command); here, use it once the stream "
                               "has ended and its replay is up.")
        src, results = pipeline.find_url(job.url, _work(job.url), s, _progress(job))
    else:
        path = Path(job.url).expanduser()
        if not path.is_file():
            raise RuntimeError(f"No such file: {path}")
        src = source.Source(path=path, title=path.stem)
        results = pipeline.find(src, s, _progress(job))
    job.src, job.candidates = src, results
    job.result = {"video_title": src.title, "candidates": [
        {"index": i, "start": round(r.start, 2), "end": round(r.end, 2),
         "length_seconds": round(r.end - r.start), "at": f"{_fmt(r.start)}-{_fmt(r.end)}",
         "signal_score": round(r.score, 2), "transcript": r.text or "(no speech)"}
        for i, r in enumerate(results)]}
    if not results:
        job.result["message"] = "No standout moments were found in this video."


def _make(job: Job, found: Job, clips: list[dict], captions: bool, facecam: bool) -> None:
    src = found.src
    words = [w for r in found.candidates for w in (r.words or [])]
    results = []
    for i, c in enumerate(clips):
        if "index" in c and ("start" not in c or "end" not in c):
            base = found.candidates[int(c["index"])]
            start, end = base.start, base.end
        else:
            start, end = float(c["start"]), float(c["end"])
        if not 1 <= end - start <= 180:
            raise ValueError(f"Clip {i + 1} is {end - start:.0f} s long; clips must be 1-180 s.")
        r = pipeline.Result(start, end, score=len(clips) - i, title=str(c.get("title", ""))[:80])
        inside = [w for w in words if start <= w[0] < end]
        r.words = sorted(set(words)) if inside else None   # reuse the transcript when it covers the clip
        results.append(r)
    s = pipeline.Settings(captions=captions, facecam=facecam, auth=found.auth)
    out = BASE / "clips" / pipeline.slug(src.title or "clips")
    if source.is_url(found.url):
        done = pipeline.export_sections(found.url, src, results, out, _work(found.url), s,
                                        _progress(job))
    else:
        done = pipeline.export(src, results, out, s, _progress(job))
    job.result = {"folder": str(out), "clips": [
        {"title": r.title, "file": str(r.path), "start": round(r.start, 2), "end": round(r.end, 2),
         "facecam_on_top": r.facecam} for r in done]}


def build():
    try:
        from mcp.server.mcpserver import MCPServer   # mcp 2.x
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x

    mcp = MCPServer("ytclipper", instructions=INSTRUCTIONS)

    @mcp.tool()
    def find_moments(url: str, candidates: int = 10, min_seconds: float = 15,
                     max_seconds: float = 60, browser_login: str = "") -> dict:
        """Start looking for the best moments of a YouTube video or finished livestream (or a
        local video file path). Returns a job id at once; call check_job with it to get the
        candidate moments, each with its start/end in seconds and the transcript of what is
        said. `candidates` is how many to shortlist (you then pick the best of them).
        `browser_login` (chrome, safari, firefox, edge, brave) uses that browser's YouTube
        login; only needed if YouTube asks to sign in."""
        candidates = max(2, min(12, int(candidates)))
        job = _start("find", url.strip(), lambda j: _find(
            j, candidates, max(5.0, min_seconds), max(min_seconds, max_seconds), browser_login))
        return job.view() | {"next": "Call check_job with this job_id."}

    @mcp.tool()
    def check_job(job_id: str) -> dict:
        """Wait up to about 40 seconds for a job, then report it: its stage and progress
        while running; the candidate moments (find_moments) or the finished clip files
        (make_clips) when done; or the error. Call again while status is "running"."""
        job = JOBS.get(job_id.strip())
        if not job:
            return {"error": f"No job {job_id!r}. Jobs are forgotten when the Claude app restarts."}
        job.done.wait(WAIT)
        return job.view()

    @mcp.tool()
    def make_clips(job_id: str, clips: list[dict], captions: bool = True,
                   facecam: bool = True) -> dict:
        """Cut the chosen moments into vertical 1080x1920 clips. `job_id` is the finished
        find_moments job. `clips` is a list of {"index": n, "title": "..."} using candidate
        indexes, or {"start": s, "end": s, "title": "..."} for adjusted times (seconds).
        Titles name the files. `captions` adds animated word-by-word captions; `facecam`
        puts the streamer's webcam above the gameplay when one is found. Returns a job id;
        call check_job with it to get the file paths."""
        found = JOBS.get(job_id.strip())
        if not found or found.kind != "find":
            return {"error": "Give the job_id of a find_moments job."}
        if found.status != "done":
            return {"error": "That find_moments job has not finished; call check_job first."}
        if not clips:
            return {"error": "Choose at least one clip."}
        job = _start("make", found.url, lambda j: _make(j, found, clips, captions, facecam))
        return job.view() | {"next": "Call check_job with this job_id."}

    @mcp.tool()
    def open_clips_folder(folder: str = "") -> dict:
        """Open the folder of finished clips (default: the main clips folder) in Finder."""
        path = Path(folder).expanduser() if folder else BASE / "clips"
        try:
            path.resolve().relative_to(BASE.resolve())
        except ValueError:
            return {"error": f"Only folders inside {BASE} can be opened."}
        path.mkdir(parents=True, exist_ok=True)
        opener = {"Darwin": "open", "Windows": "explorer"}.get(platform.system(), "xdg-open")
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"opened": str(path)}

    return mcp


def main() -> None:
    # The Claude app starts servers with a bare PATH; ffmpeg and node usually live here
    extra = ["/opt/homebrew/bin", "/usr/local/bin"]
    os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])
    build().run()
