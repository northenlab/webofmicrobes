# figure1_white_20261123.py
"""Figure 1: restore the white background of panel a (mSystems revision).

Panel a was drawn in draw.io and exported with a transparent background. When panel c was added
(r1_edits.figure1_with_c), the image was converted to RGB without first placing it on white, which
turned every transparent pixel black and hid the black labels of panel a. The corrected image
(wom/manuscript_figures/msystems_revision/lambda0/r1/fig1c/figure1_revised.png) is rebuilt from the
submitted Figure 1, the revised panel b and panel c, on white; it differs from the image in the
manuscript only where the source was transparent. This script puts it into the manuscript. The
image size is unchanged, so its frame in the document is too.

Usage:
    python figure1_white_20261123.py            # dry run
    python figure1_white_20261123.py --apply
"""

import io
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402

FIXED = (Path(__file__).resolve().parents[3] /
         "wom/manuscript_figures/msystems_revision/lambda0/r1/fig1c/figure1_revised.png")
PART = "word/media/image1.png"


def black_fraction(im: Image.Image) -> float:
    return float((np.asarray(im.convert("RGB")).sum(axis=2) == 0).mean())


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms = TrackedDocx(revision_paths.MS, "unused")
    old = Image.open(io.BytesIO(ms.read_part(PART)))
    new_bytes = FIXED.read_bytes()
    new = Image.open(io.BytesIO(new_bytes))
    assert new.size == old.size == (2048, 1523) and new.mode == "RGB", (old.size, new.size, new.mode)
    a, b = np.asarray(old.convert("RGB")).astype(int), np.asarray(new).astype(int)
    changed = np.abs(a - b).max(axis=2) > 0
    assert not changed[783:].any(), "panel c differs"
    darker = b[changed].sum(axis=1) < a[changed].sum(axis=1)
    assert not darker.any(), "a pixel became darker; only transparent areas should change, towards white"
    print(f"   {PART}: black pixels {black_fraction(old):.1%} -> {black_fraction(new):.1%}; "
          f"{changed.mean():.1%} of pixels change, all in panels a and b and all towards white")
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    ms.replace_part(PART, new_bytes)
    ms.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
