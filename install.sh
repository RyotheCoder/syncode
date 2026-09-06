#!/usr/bin/env bash
# Syncode one-line installer: cai ban release tu GitHub vao Python hien tai.
# Dung:  curl -sSL https://raw.githubusercontent.com/Ryothecoder/syncode/main/install.sh | bash
# Tuy chon: SYNCODE_VERSION=v0.2.0 ... (mac dinh v0.1.0)
set -euo pipefail

REPO="Ryothecoder/syncode"
TAG="${SYNCODE_VERSION:-v0.1.0}"
URL="https://github.com/${REPO}/archive/refs/tags/${TAG}.zip"

command -v python3 >/dev/null || { echo "Can python3 (>=3.9). Cai truoc roi chay lai." >&2; exit 1; }
python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" \
  || { echo "Can python >= 3.9 (dang dung $(python3 --version))." >&2; exit 1; }
python3 -m pip --version >/dev/null || { echo "Thieu pip. Cai pip roi chay lai." >&2; exit 1; }

echo "Cai Syncode ${TAG} tu ${URL} ..."
python3 -m pip install --upgrade "${URL}"

command -v syncode >/dev/null && echo "OK: lenh 'syncode' da san sang." || {
  echo "Cai xong nhung chua thay lenh 'syncode' trong PATH." >&2
  echo "Thu: python3 -m pip show -f syncode | head, hoac them ~/.local/bin vao PATH." >&2
  exit 1
}

echo
echo "Buoc tiep theo (1 trong 2):"
echo "  export NVIDIA_API_KEY=\"nvapi-...\"   # key mien phi: https://build.nvidia.com"
echo "  syncode                               # vao app roi go /.key nvapi-..."
