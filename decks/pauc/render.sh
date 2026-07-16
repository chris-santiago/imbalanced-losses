#!/usr/bin/env bash
# Render a Marp deck. ALWAYS passes --allow-local-files: without it Marp
# silently drops local images (SVGs/PNGs) from PDF and PNG output.
#
# Usage:
#   render.sh <deck.md> html            -> <deck>.html
#   render.sh <deck.md> pdf             -> <deck>.pdf
#   render.sh <deck.md> all             -> both
#   render.sh <deck.md> png <out-dir>   -> per-slide PNGs for the QA gate
set -euo pipefail

DECK="$1"
MODE="${2:-all}"
BASE="${DECK%.md}"
MARP=(npx -y @marp-team/marp-cli@latest "$DECK" --allow-local-files)

# Run from the deck's directory so relative asset paths resolve.
cd "$(dirname "$DECK")"
DECK="$(basename "$DECK")"
BASE="${DECK%.md}"
MARP=(npx -y @marp-team/marp-cli@latest "$DECK" --allow-local-files)

case "$MODE" in
  html) "${MARP[@]}" -o "${BASE}.html" ;;
  pdf)  "${MARP[@]}" --pdf -o "${BASE}.pdf" ;;
  all)  "${MARP[@]}" -o "${BASE}.html"
        "${MARP[@]}" --pdf -o "${BASE}.pdf" ;;
  png)  OUT_DIR="${3:?png mode needs an output dir}"
        mkdir -p "$OUT_DIR"
        "${MARP[@]}" --images png -o "${OUT_DIR}/slide.png" ;;
  *) echo "unknown mode: $MODE (html|pdf|all|png)" >&2; exit 1 ;;
esac
