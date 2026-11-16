import numpy as np

from ytclipper import scoring, source


def test_pick_clips_finds_peaks_and_respects_lengths():
    score = np.zeros(600)
    score[100:110] = 5
    score[400:405] = 3
    clips = scoring.pick_clips(score, count=5, min_len=15, max_len=30)
    assert len(clips) == 2
    for clip, (a, b) in zip(clips, [(100, 110), (400, 405)]):
        assert clip.start <= a and clip.end >= b
        assert 15 <= clip.length <= 30
    assert clips[0].score > clips[1].score


def test_long_peak_is_capped_and_edges_are_clamped():
    score = np.zeros(200)
    score[0:150] = 4
    score[190:200] = 6
    clips = scoring.pick_clips(score, count=2, min_len=15, max_len=40)
    assert all(0 <= c.start < c.end <= 200 and c.length <= 40 for c in clips)
    assert clips[-1].end == 200


def test_flat_signal_yields_nothing():
    assert scoring.pick_clips(scoring.combine({"audio": np.full(300, 40.0)})) == []


def test_combine_prefers_agreeing_signals():
    audio, chat = np.random.default_rng(0).normal(40, 1, 300), np.zeros(300)
    audio[50:60] += 20          # loud only
    audio[200:210] += 20        # loud and chat goes wild
    chat[200:210] = 30
    score = scoring.combine({"audio": audio, "chat": chat})
    assert score[200:210].mean() > score[50:60].mean() > score[120:130].mean()


def test_heatmap_and_chat_signals(tmp_path):
    hm = source.heatmap_signal([{"start_time": 0, "end_time": 10, "value": 0.2},
                                {"start_time": 10, "end_time": 20, "value": 1.0}], 30)
    assert hm[5] == 0.2 and hm[15] == 1.0 and hm[25] == 0
    f = tmp_path / "x.live_chat.json"
    f.write_text('{"replayChatItemAction": {"videoOffsetTimeMsec": "3500"}}\n'
                 'garbage\n{"replayChatItemAction": {"videoOffsetTimeMsec": "3900"}}\n')
    assert source.chat_signal(f, 10)[3] == 2
