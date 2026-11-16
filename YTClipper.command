#!/bin/bash
# Double-click this file on a Mac to start YT Clipper in your browser.
cd "$(dirname "$0")" || exit 1

need() { command -v "$1" >/dev/null 2>&1; }
missing=""
need python3 || missing="$missing python"
need ffmpeg  || missing="$missing ffmpeg"
{ need node || need deno; } || missing="$missing node"
if [ -n "$missing" ]; then
  echo "YT Clipper needs:$missing"
  if need brew; then
    echo "Installing with Homebrew..."
    brew install $missing || { read -r -p "Install failed. Press Enter to close."; exit 1; }
  else
    echo "Install Homebrew from https://brew.sh, then run:  brew install$missing"
    read -r -p "Press Enter to close."
    exit 1
  fi
fi

if [ ! -x .venv/bin/ytclipper ]; then
  echo "First run: setting up (takes a minute)..."
  python3 -m venv .venv && .venv/bin/pip install -q -e . || { read -r -p "Setup failed. Press Enter to close."; exit 1; }
fi
exec .venv/bin/ytclipper ui
