import spanav_eeg_utils.spectral_utils as spct
import pandas as pd
import pytest
from mne.time_frequency import BaseTFR, Spectrum
from unittest.mock import MagicMock

mock_spec = MagicMock(spec=Spectrum)
mock_tfr = MagicMock(spec=BaseTFR)


@pytest.mark.parametrize(
    'inp_df, context',
    [
        # Columns with the right names absent
        (pd.DataFrame(dict(
            colx=[],
            coly=[],
        )),
         pytest.raises(ValueError)),

        # Spectral objects columns does not contain spectral objects of accepted types (TFR or Spectrum)
        (pd.DataFrame(dict(
            objs_col=[int, float],
            objs_name=['', ''],
        )),
         pytest.raises(TypeError)),

        # Spectral objects columns contains spectral objects of accepted but different types (TFR and Spectrum)
        (pd.DataFrame(dict(
            objs_col=[mock_spec, mock_tfr],
            objs_name=['', ''],
        )),
         pytest.raises(TypeError)),

        # Multiple rows as bl_name
        (pd.DataFrame(dict(
            objs_col=[mock_spec, mock_spec],
            objs_name=['bl_name', 'bl_name'],
        )),
         pytest.raises(ValueError)),
    ]
)
def test_bl_obj_rows_raises(inp_df, context):
    with context:
        args = 'objs_name', 'objs_col', 'bl_name'
        spct.spectral_bl_corr_from_df(inp_df, *args)
