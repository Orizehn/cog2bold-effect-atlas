"""Read framewise-displacement (FD) confounds tables and emit a JSON filter.

The filter format is the same as the legacy code's:
    {"included": [{"subject": int, "session": int}, ...]}

Adapters can pass this JSON to `materialize(subject_session_filter=...)`.

Usage (CLI):
    python -m cog2bold.utils.filter_subjects \\
        --confounds-root /path/to/DFG_fmriprep_confounds_only \\
        --out filters/good_fd_0_5_mean_45prec.json \\
        --max-mean-fd 0.5 --max-percent-bad 45
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_fd_metrics(
    subject_id: int, session_id: int, confounds_root: Path,
) -> Optional[Dict[str, float]]:
    confounds_root = Path(confounds_root)
    subj_folder = confounds_root / f"sub-{subject_id}"
    if not subj_folder.exists():
        return None
    sess_folder = subj_folder / f"ses-{session_id}" / "func"
    if not sess_folder.exists():
        return None
    files = list(sess_folder.glob("*_desc-confounds_timeseries.tsv"))
    files = [
        f for f in files if (f"_ses-{session_id}_" in f.name) or ("_ses-" not in f.name)
    ]
    if not files:
        return None
    fd_all = []
    fd_threshold = 0.5
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
            if "framewise_displacement" in df.columns:
                fd_all.append(df["framewise_displacement"].values)
        except Exception:
            continue
    if not fd_all:
        return None
    fd_concat = np.concatenate(fd_all)
    n_total = len(fd_concat)
    n_bad = int(np.sum(fd_concat > fd_threshold))
    return {
        "mean_fd": float(np.nanmean(fd_concat)),
        "bad_volumes": n_bad,
        "total_volumes": n_total,
        "percent_bad": (n_bad / n_total * 100.0) if n_total else 0.0,
        "n_runs": len(files),
    }


def discover_subject_sessions(confounds_root: Path):
    confounds_root = Path(confounds_root)
    pairs = []
    for subj_dir in sorted(confounds_root.glob("sub-*")):
        m = re.match(r"sub-(\d+)$", subj_dir.name)
        if not m:
            continue
        subj = int(m.group(1))
        for sess_dir in sorted(subj_dir.glob("ses-*")):
            m2 = re.match(r"ses-(\d+)$", sess_dir.name)
            if not m2:
                continue
            sess = int(m2.group(1))
            if (sess_dir / "func").exists():
                pairs.append((subj, sess))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confounds-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-mean-fd", type=float, default=0.5)
    p.add_argument("--max-percent-bad", type=float, default=45.0)
    args = p.parse_args()

    pairs = discover_subject_sessions(args.confounds_root)
    included = []
    excluded = []
    for subj, sess in pairs:
        m = load_fd_metrics(subj, sess, args.confounds_root)
        if m is None:
            excluded.append({"subject": subj, "session": sess, "reason": "no FD"})
            continue
        if m["mean_fd"] > args.max_mean_fd:
            excluded.append({"subject": subj, "session": sess, "reason": f"mean FD={m['mean_fd']:.3f}"})
            continue
        if m["percent_bad"] > args.max_percent_bad:
            excluded.append({"subject": subj, "session": sess, "reason": f"%bad={m['percent_bad']:.1f}"})
            continue
        included.append({"subject": subj, "session": sess, **m})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"included": included, "excluded": excluded}, indent=2))
    print(f"Kept {len(included)} of {len(pairs)} (subject, session) pairs -> {args.out}")


if __name__ == "__main__":
    main()
