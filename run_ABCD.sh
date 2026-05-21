#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/ABCD_analysis.py"
LCG_BASE="/cvmfs/sft.cern.ch/lcg/views"

OS_MAJOR="$(
  . /etc/os-release
  echo "${VERSION_ID%%.*}"
)"
case "${OS_MAJOR}" in
  9) EL_TAG="el9" ;;
  8) EL_TAG="el8" ;;
  *) echo "ERROR: Unsupported OS major: ${OS_MAJOR}" >&2; exit 2 ;;
esac

LCG_VIEW="${LCG_VIEW:-}"

find_best_view_for_el() {
  local el_tag="$1"
  local best=""

  while IFS= read -r v; do
    if ls "${LCG_BASE}/${v}"/x86_64-"${el_tag}"-*-opt/setup.sh >/dev/null 2>&1; then
      best="${v}"
    fi
  done < <(ls -1 "${LCG_BASE}" | grep -E '^LCG_[0-9]+' | sort -V)

  [[ -n "${best}" ]] || return 1
  echo "${best}"
}

if [[ -z "${LCG_VIEW}" ]]; then
  if ! LCG_VIEW="$(find_best_view_for_el "${EL_TAG}")"; then
    echo "ERROR: No LCG view found that provides an ${EL_TAG} build." >&2
    exit 3
  fi
fi

VIEW_DIR="${LCG_BASE}/${LCG_VIEW}"
[[ -d "${VIEW_DIR}" ]] || { echo "ERROR: LCG view not found: ${VIEW_DIR}" >&2; exit 4; }

ARCH="${ARCH:-}"

pick_arch() {
  local view_dir="$1"
  local el_tag="$2"
  local candidates=(
    "x86_64-${el_tag}-gcc13-opt"
    "x86_64-${el_tag}-gcc12-opt"
    "x86_64-${el_tag}-gcc11-opt"
    "x86_64-${el_tag}-gcc10-opt"
    "x86_64-${el_tag}-gcc9-opt"
  )

  for a in "${candidates[@]}"; do
    if [[ -r "${view_dir}/${a}/setup.sh" ]]; then
      echo "${a}"
      return 0
    fi
  done

  local any
  any="$(find "${view_dir}" -maxdepth 2 -type f -name setup.sh -path "*x86_64-${el_tag}-*-opt/*" 2>/dev/null | head -n 1 || true)"
  [[ -n "${any}" ]] || return 1
  echo "$(basename "$(dirname "${any}")")"
}

if [[ -z "${ARCH}" ]]; then
  if ! ARCH="$(pick_arch "${VIEW_DIR}" "${EL_TAG}")"; then
    echo "ERROR: View ${LCG_VIEW} has no compatible arch for ${EL_TAG}" >&2
    echo "Available arches:" >&2
    ls -1 "${VIEW_DIR}" | head -n 80 >&2
    exit 5
  fi
fi

SETUP="${VIEW_DIR}/${ARCH}/setup.sh"
[[ -r "${SETUP}" ]] || { echo "ERROR: setup.sh not found: ${SETUP}" >&2; exit 6; }

#echo "[run_ABCD.sh] OS:        ${EL_TAG}"
#echo "[run_ABCD.sh] LCG view:   ${LCG_VIEW}"
#echo "[run_ABCD.sh] Arch:       ${ARCH}"

# LCG setup scripts sometimes reference unset vars; disable nounset just for sourcing.
# shellcheck disable=SC1090
set +u
source "${SETUP}"
set -u

exec python "${SCRIPT}" "$@"
