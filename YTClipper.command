#!/bin/bash
# Double-click this file on a Mac to start YT Clipper in your browser.
cd "$(dirname "$0")" || exit 1
source scripts/mac-setup.sh
exec .venv/bin/ytclipper ui
