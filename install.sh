#!/usr/bin/env bash
set -euo pipefail

if ! command -v magick >/dev/null 2>&1; then
    echo "ImageMagick 7 is required (missing: magick)." >&2
    exit 1
fi

python3 -m pip install -e '.[hub,generation]'
