#!/bin/bash
# Double-click this file on a Mac to let Claude (in the Claude desktop app) make clips for you.
cd "$(dirname "$0")" || exit 1
source scripts/mac-setup.sh
.venv/bin/ytclipper connect-claude || fail "Could not update the Claude app's settings."
echo
echo "Done. In the Claude app, start a new chat and paste a YouTube link, e.g."
echo "  \"Make 3 clips from https://www.youtube.com/watch?v=...\""
echo
echo "Keep this folder where it is: the Claude app runs YT Clipper from here."
read -r -p "Press Enter to close."
