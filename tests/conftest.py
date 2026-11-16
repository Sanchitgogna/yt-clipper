import subprocess

import pytest

BURSTS = [(40, 45), (90, 95)]


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    """2 minutes of quiet noise with two loud bursts."""
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    enable = "+".join(f"between(t,{a},{b})" for a, b in BURSTS)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=120",
         "-f", "lavfi", "-i", "anoisesrc=d=120:a=0.01:r=44100",
         "-af", f"volume=volume=30:enable='{enable}'",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "15", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path
