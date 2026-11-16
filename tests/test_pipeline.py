import subprocess

import numpy as np

from ytclipper import cli, live, media, pipeline, scoring, source

from .conftest import BURSTS


def test_vod_pipeline_clips_the_bursts(video, tmp_path):
    src = source.Source(path=video, title="Sample video!")
    _, signals = pipeline.analyze(src)
    assert set(signals) == {"audio", "scenes"}
    results = pipeline.find(src, pipeline.Settings(count=3, min_len=10, max_len=30))
    assert len(results) == 2
    for r, (a, b) in zip(results, BURSTS):
        assert r.start <= a and r.end >= b
    for r in pipeline.export(src, results, tmp_path, pipeline.Settings(captions=False)):
        assert abs(media.duration(r.path) - (r.end - r.start)) < 0.5


def _size(path):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True).stdout.strip()


def test_cli_makes_vertical_clips(video, tmp_path, capsys):
    cli.main(["clip", str(video), "-n", "1", "-o", str(tmp_path), "--no-captions"])
    (out,) = tmp_path.glob("*.mp4")
    assert _size(out) == "1080,1920"


def test_captions_are_drawn_on_the_clip(video, tmp_path):
    from ytclipper import render
    words = [(0.5, 0.9, "hello"), (0.9, 1.4, "there"), (3.0, 3.5, "bye")]
    plain = render.render(video, tmp_path / "plain.mp4", 10, 4)
    capped = render.render(video, tmp_path / "capped.mp4", 10, 4, words=words)
    assert _size(capped) == "1080,1920" and abs(media.duration(capped) - 4) < 0.2

    def frame(path, t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(int)

    assert np.abs(frame(capped, 1.0) - frame(plain, 1.0)).max() > 100   # words on screen
    assert np.abs(frame(capped, 2.4) - frame(plain, 2.4)).max() < 40    # gap between phrases


def test_live_detector_on_segments(video, tmp_path):
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-c", "copy", "-f", "segment",
                    "-segment_time", "10", "-reset_timestamps", "1", str(seg_dir / "seg_%06d.ts")],
                   check=True)
    det = live.LiveDetector(warmup=20, pre_roll=10, post_roll=5, cooldown=15)
    segments, clock, found = [], 0.0, []
    for path in sorted(seg_dir.glob("*.ts")):
        energy = media.audio_energy(path)
        segments.append(live.Segment(path, clock, float(len(energy))))
        clock += len(energy)
        for clip in det.update(energy):
            # a clip must be cut-able from what has been recorded so far
            out = live.cut_from_segments(segments, clip, tmp_path / f"c{len(found)}.mp4",
                                         pipeline.Settings(captions=False))
            assert out and abs(media.duration(out) - clip.length) < 1.0
            found.append(clip)
    assert len(found) == 2
    for clip, (a, b) in zip(found, BURSTS):
        assert clip.start <= a - 5 and clip.end >= b


def test_facecam_layout_stacks_webcam_over_game(video, tmp_path):
    from ytclipper import facecam, render
    layout = facecam.Layout(cam=(0.0, 0.5, 0.5, 0.5))          # bottom-left quarter of the frame
    out = render.render(video, tmp_path / "split.mp4", 5, 2, layout)
    assert _size(out) == "1080,1920"
    graph, caption_y = render._graph(1920, 1080, facecam.Layout(cam=(0.02, 0.7, 0.25, 0.25)))
    assert "vstack" in graph and 480 <= caption_y - 16 <= 760      # captions sit just under the webcam
    assert "vstack" not in render._graph(1920, 1080, facecam.Layout())[0]


def test_no_facecam_in_plain_footage(video):
    from ytclipper import facecam
    assert facecam.detect(video, 0, 20).cam is None
