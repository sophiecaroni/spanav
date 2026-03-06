import pytest
import spanav_eeg_utils.parsing_utils as prs
from pathlib import Path
from contextlib import nullcontext


@pytest.mark.parametrize(
    "inp, expected",
    [
        # subject dirs rewritten
        (Path("/outputs/Cleaning/73T04"),
         Path("/outputs/Cleaning/sub-T04")),

        (Path("/outputs/Cleaning/T04"),
         Path("/outputs/Cleaning/sub-T04")),

        (Path("/x/y/sub-73T01"), Path("/x/y/sub-T01")),

        (Path("/x/y/WP73A01"), Path("/x/y/sub-A01")),

        # Examples with group directories never modified
        (Path("/intermediate/WP73A/sub-A01"),
         Path("/intermediate/WP73A/sub-A01")),

        (Path("/x/y/BIDS_Data_WP73T"), Path("/x/y/BIDS_Data_WP73T")),

        # Whole noisy path untouched
        (Path("/x/y/fooba"), Path("/x/y/fooba")),
    ],
)
def test_check_path_sid_cases(inp: Path, expected: Path):
    assert prs.check_path_sid(inp) == expected


def test_check_path_sid_preserves_parent_path():
    p = Path("/a/b/WP73T/73T04")
    out = prs.check_path_sid(p)
    assert out.parent == p.parent
    assert out.name == "sub-T04"


@pytest.mark.parametrize(
    "inp, expected",
    [
        (dict(acq=1, task='SpaNav'),
         'block1'),

        (dict(acq='1', task='SpaNav'),
         'block1'),

        (dict(acq='block1', task='SpaNav'),
         'block1'),

        (dict(acq='pre', task='RestEO'),
         'RestEO_pre')
    ],
)
def test_get_rec_block_dir_cases(inp, expected):
    assert prs.get_rec_acq_dir(**inp) == expected


@pytest.mark.parametrize(
    "inp, expected, context",
    [
        ('sub-T01', 'T', nullcontext()),
        ('A01', 'A', nullcontext()),
        ('T1', None, pytest.raises(ValueError)),
        ('B01', None, pytest.raises(ValueError)),

    ],
)
def test_get_group_letter_cases(inp, expected, context):
    with context:
        result = prs.get_group_letter(inp)
        assert result == expected
