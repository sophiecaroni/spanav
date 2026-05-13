import os
import re


def _get_accepted_orders():
    return [
        ('imp', 'rs', 'block', 'rs'),
        ('imp', 'rs', 'block', 'imp', 'rs'),
        ('imp', 'rs', 'block', 'rs', 'imp'),
        ('imp', 'rs', 'block', 'imp', 'block', 'imp', 'rs')
    ]


def _get_fname_template(fname: str) -> str:
    if 'im' in fname.lower():
        return 'imp'
    elif 'rs' in fname.lower():
        return 'rs'
    elif 'block' in fname.lower():
        return 'block'
    else:
        raise ValueError(f'Unknown fname format: {fname}')


def _get_vhdr_files(files: list) -> list:
    return [file for file in files if file.suffix == '.vhdr']


def check_expected_order(sid_dirs: dict) -> dict:
    ordered_files = {sid: [] for sid in sid_dirs.keys()}
    for sid, files in sid_dirs.items():
        files.sort(key=os.path.getmtime)
        accepted_orders = _get_accepted_orders()
        current_order = []
        for file in files:
            if file.suffix == '.vhdr':
                template_name = _get_fname_template(file.stem)
                if not current_order or template_name != current_order[-1]:
                    current_order.append(template_name)
            ordered_files[sid].append(file)

        exist = tuple(current_order) in accepted_orders
        assert exist, f'Wrong order detected for {sid}: {current_order}'
    print('✅ All correctly ordered')
    return ordered_files


def _get_vhdr_inner_time(fpath) -> str:
    f = open(fpath).read()
    match = re.search(r'Impedance\s+\[kOhm\]\s+at\s+(\d{2}:\d{2}:\d{2})', f)
    if match is None:
        return 'nan'
    else:
        return str(match.group(1))


def check_correct_time(sid_dirs: dict) -> None:
    for sid, fpaths in sid_dirs.items():
        has_problem = False
        print(f"\n=== {sid} === ")
        vhdr_fpaths = _get_vhdr_files(fpaths)
        imp_itime = None
        for fpath in vhdr_fpaths:
            if _get_fname_template(fpath.stem) == 'imp':
                imp_itime = _get_vhdr_inner_time(fpath)
            elif imp_itime is not None:
                itime = _get_vhdr_inner_time(fpath)
                if imp_itime != itime:
                    print(f"\t❌ Mismatch: \timpedance t={imp_itime} \t{fpath.stem} t={itime}")
                    has_problem = True
        if not has_problem:
            print(f"\t✅ All good")

