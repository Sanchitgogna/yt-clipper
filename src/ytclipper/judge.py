"""Let Claude pick the clips worth posting and write their titles.

The loudness/chat/replay signals find moments where *something* happened. Claude
reads what was said in each and decides which would hold a viewer, the way an
editor would. Optional: without an API key the tool ranks by its own heuristics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path.home() / ".ytclipper.json"
MODEL = "claude-opus-5-5"

SYSTEM = """You are a short-form video editor choosing clips for YouTube Shorts, Reels and TikTok.

You get candidate moments from one long video or stream. A program found them by looking for \
loud audio, chat spikes and replays, so each is a moment where something happened, but it could \
not tell whether the moment is any good. You decide that from what is said in each.

Rate each candidate from 1 to 10 for how well it would work as a standalone clip for someone \
who has not seen the rest of the video:
- High: a complete beat that lands on its own - a joke with its punchline, a strong reaction, \
a surprising claim, a clutch play called as it happens, a story with a payoff.
- Low: mid-thought fragments with no payoff, housekeeping ("let me check my settings"), reading \
out donations or sponsors, intros and outros, filler.
- A candidate with no transcript means nobody spoke; the program still found it loud or busy. \
You cannot see the picture, so rate these 5 unless the title makes clear the video is about \
something where wordless action is the point, in which case rate them higher.

Write a title for each: at most 60 characters, specific to what happens, in the language of the \
transcript, no hashtags, no emoji, no clickbait that the clip does not pay off. Give the reason \
for the rating in one short sentence. Transcripts are machine-made and contain errors; read \
through them. Rate every candidate, using its index."""

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "rating": {"type": "integer"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "rating", "title", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    rating: int
    title: str
    reason: str


def saved_key() -> str:
    try:
        return json.loads(CONFIG.read_text()).get("anthropic_api_key", "")
    except (OSError, ValueError):
        return ""


def save_key(key: str) -> None:
    """Keep the key in a file only this user can read (empty removes it)."""
    if not key:
        CONFIG.unlink(missing_ok=True)
        return
    CONFIG.touch(mode=0o600, exist_ok=True)
    CONFIG.chmod(0o600)
    CONFIG.write_text(json.dumps({"anthropic_api_key": key}))


def source() -> str | None:
    """Where credentials come from: 'environment', 'saved', or None when there are none."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    return "saved" if saved_key() else None


def rate(video_title: str, candidates: list[dict]) -> dict[int, Verdict]:
    """Ask Claude to rate candidates ({index, start, end, transcript}). Raises on failure."""
    import anthropic

    client = anthropic.Anthropic() if source() == "environment" else anthropic.Anthropic(api_key=saved_key())
    listing = "\n\n".join(
        f"<candidate index=\"{c['index']}\" seconds=\"{c['end'] - c['start']:.0f}\">\n"
        f"{c['transcript'] or '(no speech)'}\n</candidate>" for c in candidates)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content":
                   f"Video title: {video_title or 'unknown'}\n\n{listing}"}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        # if a safety classifier declines the request, let the API retry it on another model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to review these transcripts")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off")
    text = next(b.text for b in response.content if b.type == "text")
    known = {c["index"] for c in candidates}
    return {c["index"]: Verdict(max(1, min(10, c["rating"])), c["title"].strip()[:80], c["reason"].strip())
            for c in json.loads(text)["clips"] if c["index"] in known}


def explain(err: Exception) -> str:
    import anthropic

    if isinstance(err, anthropic.AuthenticationError):
        return "the API key was rejected"
    if isinstance(err, anthropic.PermissionDeniedError):
        return "the API key is not allowed to use this model"
    if isinstance(err, anthropic.RateLimitError):
        return "the API rate limit was hit"
    if isinstance(err, anthropic.APIStatusError):
        return f"the API answered with an error ({err.status_code})"
    if isinstance(err, anthropic.APIConnectionError):
        return "the API could not be reached"
    return str(err) or type(err).__name__
