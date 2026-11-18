# Shared first-run setup for the Mac launchers; sourced, not run on its own.

# Homebrew lives here on Apple Silicon and Intel Macs; a double-clicked script may not see it otherwise
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

fail() { echo; echo "$1"; read -r -p "Press Enter to close."; exit 1; }
need() { command -v "$1" >/dev/null 2>&1; }

# A Python new enough for the app (macOS's built-in one is 3.9, too old)
python_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
find_python() {
  for p in python3.13 python3.12 python3.11 python3.10 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if need "$p" && python_ok "$p"; then command -v "$p"; return; fi
  done
}

PY="$(find_python)"
missing=""
[ -z "$PY" ] && missing="$missing python@3.12"
need ffmpeg || missing="$missing ffmpeg"
{ need node || need deno; } || missing="$missing node"

if [ -n "$missing" ]; then
  echo "YT Clipper needs:$missing"
  need brew || fail "Install Homebrew from https://brew.sh first, then double-click this file again."
  echo "Installing with Homebrew (this can take a few minutes)..."
  brew install $missing || fail "Homebrew could not install:$missing"
  PY="$(find_python)"
  [ -z "$PY" ] && fail "Python 3.10 or newer is still missing. Run: brew install python@3.12"
fi

# A copy set up by an older Python, or a setup that failed halfway, is started again
if [ -x .venv/bin/python ] && ! python_ok .venv/bin/python; then rm -rf .venv; fi
if [ ! -x .venv/bin/ytclipper ]; then
  echo "First run: setting up with $("$PY" --version) (takes a few minutes)..."
  rm -rf .venv
  "$PY" -m venv .venv || fail "Could not create the Python environment."
  .venv/bin/python -m pip install -q --upgrade pip || fail "Could not update pip."
  .venv/bin/python -m pip install -q -e . || { rm -rf .venv; fail "Setup failed (see the message above)."; }
fi
