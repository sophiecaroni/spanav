import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs


def test_get_sid_blocks_only_returns_existing(tmp_path, monkeypatch):

    sid = 'T01'

    def mock_get_group_letter(sid):
        return "T"

    def mock_get_clean_eeg_path(sid, acq, task):
        return tmp_path / f"{sid}_{acq}_filt_raw.fif"

    monkeypatch.setattr(prs, "get_group_letter", mock_get_group_letter)
    monkeypatch.setattr(io, "get_clean_eeg_path", mock_get_clean_eeg_path)

    # create only block1 and block3
    (tmp_path / f"{sid}_block1_filt_raw.fif").touch()
    (tmp_path / f"{sid}_block3_filt_raw.fif").touch()

    result = io.get_sid_blocks(sid)
    assert result == ["block1", "block3"]


