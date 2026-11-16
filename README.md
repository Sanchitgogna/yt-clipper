# ytclipper

Automatically find the best moments in YouTube videos and livestreams and cut them into clips.

## How it finds "good scenes"

Every second of the video gets a highlight score built from several signals:

| Signal    | Source                                            | Used for        |
|-----------|---------------------------------------------------|-----------------|
| `heatmap` | YouTube's "most replayed" graph                   | videos          |
| `chat`    | chat messages per second (live chat replay)       | past streams    |
| `audio`   | loudness spikes (shouting, cheering, explosions)  | videos + live   |
| `scenes`  | rate of hard cuts / big visual changes            | videos          |

Each signal is normalised against the video's own typical level, weighted (audience
signals count most) and combined. The highest peaks become clips, widened to include
the build-up and trimmed to your length limits.

## Install

Needs Python 3.10+, `ffmpeg`, and a JavaScript runtime (`node` or `deno`) on your PATH —
yt-dlp uses it to read YouTube's formats.

```bash
pip install -e .
```

## Usage

Cut the 5 best moments of a video (or a finished stream):

```bash
ytclipper clip "https://www.youtube.com/watch?v=VIDEO_ID" -n 5 -o clips/
```

Vertical 9:16 clips for Shorts / Reels / TikTok:

```bash
ytclipper clip "https://www.youtube.com/watch?v=VIDEO_ID" --vertical --max-len 45
```

Just list the moments without cutting:

```bash
ytclipper analyze "https://www.youtube.com/watch?v=VIDEO_ID" --json
```

Watch a livestream and write clips as moments happen (Ctrl-C to stop):

```bash
ytclipper live "https://www.youtube.com/watch?v=LIVE_ID" -o clips/ --threshold 4
```

Every command also accepts a local media file instead of a URL.

If YouTube answers "Sign in to confirm you're not a bot" (common on servers and VPNs),
run from a home connection or pass your own login with `--cookies-from-browser chrome`.

## Layout

- `source.py` – yt-dlp download, heatmap and chat-replay signals
- `media.py` – ffmpeg: loudness, scene cuts, cutting
- `scoring.py` – normalise/combine signals, pick clip windows
- `pipeline.py` – analyse + export for finished videos
- `live.py` – segment recorder and online spike detector for livestreams
- `cli.py` – command line

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Roadmap

- Live chat rate as a signal in `live` mode (currently loudness only)
- Speech transcription for captions and keyword/LLM-based ranking and titles
- Smart vertical reframing (follow the face/action instead of centre crop)
- Download only the highlighted sections of long VODs

Only clip content you have the rights to use.
