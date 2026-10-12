"""Streamlit Community Cloud entry point for the Cog2BOLD effect-atlas app.

Community Cloud runs this file (set "Main file path" to ``streamlit_app.py``). It switches the
app into public mode, so visitors only see the shipped run bundles, no filesystem path inputs,
a capped bootstrap and download-only reports. Then it hands over to the real app.
"""

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).parent

os.environ.setdefault("COG2BOLD_PUBLIC", "1")
os.environ.setdefault("COG2BOLD_BUNDLES", str(ROOT / "bundles"))
os.environ.setdefault("COG2BOLD_GUIDE", str(ROOT / "docs" / "effect_atlas_user_guide.pdf"))
os.environ.setdefault("COG2BOLD_MAX_BOOT", "300")
os.environ.setdefault("COG2BOLD_WORKSPACE", "/tmp/cog2bold_workspace")
os.environ.setdefault("MPLBACKEND", "Agg")

runpy.run_path(str(ROOT / "apps" / "effect_atlas.py"), run_name="__main__")
