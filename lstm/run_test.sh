set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT="${SCRIPT_DIR}"

python -m unittest discover -s "$ROOT"
