#!/usr/bin/env bash
set -euo pipefail
MG_PATH="${MG_PATH:-mg}"
PROGRAM="${1:-}"
if [[ -z "${PROGRAM}" ]]; then
  echo "Usage: $0 <path_to_mangle_program> [-- <mg_args...>]" >&2
  exit 1
fi
if ! command -v "${MG_PATH}" >/dev/null 2>&1; then
  if [[ -x "$HOME/bin/mg" ]]; then MG_PATH="$HOME/bin/mg"; else echo "mg not found in PATH or at ~/bin/mg" >&2; exit 2; fi
fi
if [[ ! -f "${PROGRAM}" ]]; then
  echo "Program not found: ${PROGRAM}" >&2
  exit 3
fi
shift || true
if [[ "${1-}" == "--" ]]; then shift; fi
exec "${MG_PATH}" "$@" -exec "$(cat "${PROGRAM}")"
