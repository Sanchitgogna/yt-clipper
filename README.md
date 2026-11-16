# YT Clipper

Paste a YouTube video or livestream link and get its best moments back as clips.
Runs on your own computer, in your browser.

![The app](docs/app.png)

## Start it

**Mac:** double-click `YTClipper.command`. The first run sets everything up, then the app
opens in your browser. (It uses Homebrew to install `ffmpeg` and `node` if they are missing.)

**Any computer** (needs Python 3.10+, `ffmpeg`, and `node` or `deno`):

```bash
pip install -e .
```

```bash
ytclipper ui
```

Clips are saved in `~/YTClipper/clips`. Livestream links are detected automatically: the
app watches the stream and adds clips as big moments happen, until you press Stop.

**Captions:** set Captions to "On the video" in Options (or `--captions`) to burn what is
said into each clip. YouTube's own captions are used when the video has them; otherwise the
speech is transcribed on your computer (`pip install -e '.[captions]'`, included by the Mac
launcher).

If YouTube answers "Sign in to confirm you're not a bot" (common on servers and VPNs), open
Options and choose your browser under "Use my YouTube login".

## How it finds "good scenes"

Every second of the video gets a highlight score built from several signals:

| Signal    | Source                                                    | Used for        |
|-----------|-----------------------------------------------------------|-----------------|
| `heatmap` | YouTube's "most replayed" graph                           | videos          |
| `chat`    | chat messages per second (live chat replay)               | past streams    |
| `speech`  | captions: fast talking, hype words, [Laughter]/[Applause] | videos          |
| `audio`   | loudness spikes (shouting, cheering, explosions)          | videos + live   |
| `scenes`  | rate of hard cuts (opt-in, `--scenes`)                    | videos          |

Each signal is normalised against the video's own typical level, weighted (audience
signals count most) and combined. The highest peaks become clips, widened to include the
build-up, and their edges are nudged to the nearest quiet moment so cuts do not land
mid-sentence.

**Why it is fast:** only the audio track is downloaded for analysis, then only the chosen
clip ranges of the video are fetched and cut, in parallel. A three-hour stream does not
need a three-hour download.

## Use it from Claude (MCP)

`ytclipper mcp` is an MCP server, so Claude can pick the clips itself: it gets the candidate
moments with their transcripts, decides which are actually good, and cuts them.

```bash
pip install -e '.[mcp]'
```

Claude Code:

```bash
claude mcp add ytclipper -- /full/path/to/yt-clipper/.venv/bin/ytclipper mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"ytclipper": {"command": "/full/path/to/yt-clipper/.venv/bin/ytclipper", "args": ["mcp"]}}}
```

Then ask: *"Find the funniest moments in <link> and make vertical clips of the best three."*

Tools: `find_highlights(url, count, min_seconds, max_seconds)` and
`export_clips(url, clips, vertical)`.

## Command line

```bash
ytclipper clip "https://www.youtube.com/watch?v=VIDEO_ID" -n 5 -o clips/ --vertical
```

```bash
ytclipper analyze "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

```bash
ytclipper live "https://www.youtube.com/watch?v=LIVE_ID" -o clips/
```

Every command also accepts a local media file instead of a URL.

## Layout

- `server.py`, `web/index.html` – the browser app
- `mcp_server.py` – tools for Claude
- `captions.py` – speech-to-text and burned-in captions
- `source.py` – yt-dlp downloads; heatmap, chat and caption signals
- `media.py` – ffmpeg: loudness, scene cuts, cutting
- `scoring.py` – normalise/combine signals, pick and snap clip windows
- `pipeline.py` – analyse + export for finished videos
- `live.py` – segment recorder and online spike detector for livestreams

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Roadmap

- Live chat rate as a signal in live mode (currently loudness only)
- Auto-generated titles
- Smart vertical reframing (follow the face/action instead of centre crop)

Only clip content you have the rights to use.
