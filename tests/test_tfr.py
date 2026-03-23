import spanav_tbi.processing.tfr as tfr_mod
from unittest.mock import Mock


def test_compute_cond_tfr_dimensions(monkeypatch):
    # Mock concatenated epoched object
    epo = Mock()
    mock_get_concat = Mock(return_value=epo)
    monkeypatch.setattr(
        tfr_mod.cmp,
        "get_concat_epo_recs",
        mock_get_concat,
    )

    # Mock computation of TFR object
    tfr = Mock()
    tfr.ch_names = ["a", "b", "c", "d"]  # specify important attributed
    tfr.data.shape = (10, 4, 40, 200)  # specify important attributed  # epochs, channels, freqs, times
    mock_compute_tfr = Mock(return_value=tfr)
    monkeypatch.setattr(tfr_mod, "compute_tfr", mock_compute_tfr)

    # Run function
    result = tfr_mod.compute_cond_tfr("", list(), "")

    # Check that full epoched object was used
    mock_get_concat.assert_called_once_with("", list(), "")
    mock_compute_tfr.assert_called_once_with(epo, log=True, norm=True)

    # Check returned TFR still has all channels
    assert result.ch_names == ["a", "b", "c", "d"]
    assert result.data.shape[1] == 4

    # Check returned still has all epochs
    assert result.data.shape[0] == 10
