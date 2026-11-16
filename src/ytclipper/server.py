"""Local web app: paste a link, get clips. Runs only on this computer."""

from __future__ import annotations

import json
import mimetypes
import platform
import subprocess
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import live, pipeline, source

INDEX = Path(__file__).parent / "web" / "index.html"
BROWSERS = {"chrome", "safari", "firefox", "edge", "brave"}


@dataclass
class Job:
    id: str
    url: str
    out_dir: Path
    status: str = "running"          # running | done | error | stopped
    stage: str = "Starting"
    progress: float | None = None
    title: str = ""
    live: bool = False
    error: str = ""
    clips: list[dict] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict:
        return {k: getattr(self, k) for k in
                ("id", "url", "status", "stage", "progress", "title", "live", "error", "clips")} | {
                "folder": str(self.out_dir)}

    def add(self, start: float, end: float, score: float, path: Path, text: str = "",
            facecam: bool = False) -> None:
        self.clips.append({"start": start, "end": end, "score": round(score, 2), "text": text,
                           "facecam": facecam,
                           "name": path.name, "file": f"/files/{self.id}/{path.name}"})


def friendly(err: Exception) -> str:
    msg = str(err)
    if "Sign in to confirm" in msg or "HTTP Error 403" in msg:
        return ("YouTube refused the download and wants a sign-in. Open Options, pick the browser "
                "where you are logged in to YouTube under “Use my YouTube login”, and try again.")
    if "ffmpeg" in msg.lower() and "not found" in msg.lower():
        return "ffmpeg is not installed. On a Mac: brew install ffmpeg"
    return msg.replace("ERROR: ", "")[:500]


class App:
    def __init__(self, base: Path):
        self.base = base
        self.jobs: dict[str, Job] = {}

    def start(self, body: dict) -> Job:
        url = str(body.get("url", "")).strip()
        if not source.is_url(url):
            raise ValueError("Paste a full link starting with https://")
        browser = body.get("browser") or None
        if browser and browser not in BROWSERS:
            raise ValueError("unknown browser")
        s = pipeline.Settings(
            count=max(1, min(20, int(body.get("count", 5)))),
            min_len=max(5.0, float(body.get("min_len", 15))),
            max_len=min(600.0, float(body.get("max_len", 60))),
            captions=bool(body.get("captions", True)),
            facecam=bool(body.get("facecam", True)),
            max_height=int(body.get("quality", 1080)),
            auth=source.Auth(cookies_from_browser=browser),
        )
        s.max_len = max(s.max_len, s.min_len)
        job_id = uuid.uuid4().hex[:10]
        job = Job(job_id, url, self.base / "clips" / job_id)
        self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, s), daemon=True).start()
        return job

    def _run(self, job: Job, s: pipeline.Settings) -> None:
        work = self.base / "cache" / job.id

        def progress(stage: str, frac: float | None) -> None:
            job.stage, job.progress = stage, frac

        try:
            progress("Reading video info", None)
            info = source.probe(job.url, work, s.auth)
            job.title = info.get("title") or ""
            if info.get("is_live"):
                self._run_live(job, s, work)
                return
            _, results = pipeline.clip_url(job.url, job.out_dir, work, s, progress)
            for r in sorted(results, key=lambda r: -r.score):
                job.add(r.start, r.end, r.score, r.path, r.text, r.facecam)
            job.status, job.stage = "done", "Done" if results else "No standout moments found"
        except Exception as e:
            job.status, job.error = "error", friendly(e)
        finally:
            pipeline.cleanup(work)

    def _run_live(self, job: Job, s: pipeline.Settings, work: Path) -> None:
        job.live, job.stage, job.progress = True, "Watching the stream for big moments", None
        args = (["--cookies-from-browser", s.auth.cookies_from_browser]
                if s.auth.cookies_from_browser else [])
        live.run(job.url, job.out_dir, work, settings=s, log=lambda *_: None,
                 stop=job.stop, ytdlp_args=args,
                 on_clip=lambda c, p: job.add(c.start, c.end, c.score, p))
        job.status = "stopped" if job.stop.is_set() else "done"
        job.stage = "Stopped" if job.stop.is_set() else "Stream ended"


def reveal(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    opener = {"Darwin": "open", "Windows": "explorer"}.get(platform.system(), "xdg-open")
    subprocess.Popen([opener, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # keep the terminal quiet
            pass

        def _json(self, data, code: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _local(self) -> bool:
            # Only this computer may talk to the app, and only pages it served itself.
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                self._json({"error": "forbidden"}, 403)
                return False
            return True

        def do_GET(self) -> None:
            if not self._local():
                return
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                return self._file(INDEX, download=False)
            if parts[:2] == ["api", "jobs"] and len(parts) == 3 and parts[2] in app.jobs:
                return self._json(app.jobs[parts[2]].view())
            if parts[0] == "files" and len(parts) == 3 and parts[1] in app.jobs:
                path = app.jobs[parts[1]].out_dir / Path(parts[2]).name
                if path.is_file():
                    return self._file(path, download="dl=1" in self.path)
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if not self._local():
                return
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._json({"error": "json only"}, 415)  # blocks cross-site form posts
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            parts = self.path.strip("/").split("/")
            if parts == ["api", "jobs"]:
                try:
                    return self._json(app.start(body).view())
                except (ValueError, TypeError) as e:
                    return self._json({"error": str(e)}, 400)
            if parts[:2] == ["api", "jobs"] and len(parts) == 4 and parts[2] in app.jobs:
                job = app.jobs[parts[2]]
                if parts[3] == "stop":
                    job.stop.set()
                    return self._json(job.view())
                if parts[3] == "open":
                    reveal(job.out_dir)
                    return self._json({"ok": True})
            self._json({"error": "not found"}, 404)

        def _file(self, path: Path, download: bool) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):  # video players seek with range requests
                a, _, b = rng[6:].partition("-")
                try:
                    if a:
                        start, end = int(a), int(b) if b else end
                    else:
                        start = max(0, size - int(b))
                except ValueError:
                    rng = None
                if rng and not 0 <= start <= end < size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(base: Path, port: int = 8765, open_browser: bool = True) -> None:
    app = App(base)
    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            import sys
            if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
                super().handle_error(request, client_address)   # a player closing a stream is normal

    httpd = Server(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"YT Clipper is running at {url}\nClips are saved in {base / 'clips'}\nPress Ctrl-C to quit.")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in app.jobs.values():
            job.stop.set()
        httpd.server_close()
