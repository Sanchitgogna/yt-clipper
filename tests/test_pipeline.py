import subprocess

from ytclipper import cli, live, media, pipeline, scoring, source

from .conftest import BURSTS


def test_vod_pipeline_clips_the_bursts(video, tmp_path):
    src = source.Source(path=video, title="Sample video!")
    score, signals = pipeline.analyze(src)
    assert set(signals) == {"audio", "scenes"}
    clips = scoring.pick_clips(score, count=3, min_len=10, max_len=30)
    assert len(clips) == 2
    for clip, (a, b) in zip(clips, BURSTS):
        assert clip.start <= a and clip.end >= b
    paths = pipeline.export(src, clips, tmp_path)
    for path, clip in zip(paths, clips):
        assert abs(media.duration(path) - clip.length) < 0.5


def test_cli_vertical(video, tmp_path, capsys):
    cli.main(["clip", str(video), "-n", "1", "-o", str(tmp_path), "--no-scenes", "--vertical"])
    (out,) = tmp_path.glob("*.mp4")
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.strip()
    assert size == "1080,1920"


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
            out = live.cut_from_segments(segments, clip, tmp_path / f"c{len(found)}.mp4")
            assert out and abs(media.duration(out) - clip.length) < 1.0
            found.append(clip)
    assert len(found) == 2
    for clip, (a, b) in zip(found, BURSTS):
        assert clip.start <= a - 5 and clip.end >= b
