#!/usr/bin/env bash
# One-step install for macOS/Linux: bash install.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3 || true)
if [ -z "$PY" ] || ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "Python 3.11 or newer is required (https://www.python.org/downloads/)."; exit 1
fi
"$PY" -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
if ! .venv/bin/python -c 'import tkinter' 2>/dev/null; then
  echo "The app window needs Tk. On Debian/Ubuntu: sudo apt install python3-tk"
fi
printf '#!/usr/bin/env bash\ncd "$(dirname "$0")"\n.venv/bin/python -m chattrainer gui\n' > run.sh
chmod +x run.sh
echo
echo "Installed. Start the app with: ./run.sh"
echo "Then open the Train tab and press 'Install training components' once (needs an NVIDIA GPU)."
