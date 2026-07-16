"""QA gate: flag deck slides whose content intrudes into the bottom margin.

Marp's default theme leaves ~78px of padding top and bottom; the pagination
number sits bottom-right inside that margin. Any other ink below the
threshold row means slide content ran past its box (Marp clips overflow
silently, so this is how cut-off slides are caught).

Usage:
    uv run --python 3.12 --with pillow python check_margin.py '<dir>/*.png'

Exit status 1 if any slide overflows, 0 if all clean.

BLIND SPOT: content pushed ENTIRELY below the canvas is never drawn, so it
leaves no ink in the margin band and passes this check while being silently
truncated (e.g. a footer + figure caption both cut off). The mandatory
visual pass over changed slides is the only catch for full truncation —
this scanner detects intrusion, not absence.
"""
import glob
import sys

from PIL import Image

THRESHOLD_Y = 690     # rows below this should be margin (of a 720px slide)
PAGENUM_X = 1150      # ignore the pagination number region (x >= this)
WHITE_CUTOFF = 245    # channel value below this counts as ink
MIN_INK = 30          # ignore stray anti-aliasing pixels

failed = False
for path in sorted(glob.glob(sys.argv[1])):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    px = img.load()
    ink = 0
    lowest = 0
    for y in range(THRESHOLD_Y, h):
        for x in range(0, min(PAGENUM_X, w)):
            r, g, b = px[x, y]
            if r < WHITE_CUTOFF or g < WHITE_CUTOFF or b < WHITE_CUTOFF:
                ink += 1
                lowest = max(lowest, y)
    name = path.rsplit("/", 1)[-1]
    if ink > MIN_INK:
        failed = True
        print(f"OVERFLOW {name}: {ink} ink px below y={THRESHOLD_Y}, lowest row {lowest}/{h}")
    else:
        print(f"ok       {name}")

sys.exit(1 if failed else 0)
