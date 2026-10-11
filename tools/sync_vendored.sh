#!/bin/bash
# Re-copy the fastbb files that the forks vendor (see VENDORED.md) from this checkout into the fork checkouts and show the diffs.
# Copies are edited only here and re-synced. Each copied .py file gets a header line naming the source path + commit.
# Usage: tools/sync_vendored.sh [raft|crestereo|sam3|all]     (fork checkouts, each on its cuda-kernels branch; nothing is committed)
#   RAFT_DIR / CRESTEREO_DIR / SAM3_DIR override the default locations (../raft, ../crestereo, ../sam3)
set -euo pipefail
FB="$(cd "$(dirname "$0")/.." && pwd)"
RAFT_DIR="${RAFT_DIR:-$FB/../raft}"; CRESTEREO_DIR="${CRESTEREO_DIR:-$FB/../crestereo}"; SAM3_DIR="${SAM3_DIR:-$FB/../sam3}"
# Encoder-only sources are selected by trim_encoder_vendor.py; source hashes name the rebuilt binaries.
# fork-name | files (paths relative to fastbb/fastbb/ = path relative to $DEST)
RAFT_FILES="normconv.py csrc/norm_conv.cu"
CRESTEREO_FILES="instnorm.py kernels.py fastbb_kernels.cu"
SAM3_FILES="attention.py kernels.py fastbb_kernels.cu cudnn_conv.py"   # trimmed to the head_dim-64 flash kernels

sync_one() {
  local name=$1 dir=$2 files=$3 DEST=${4:-fastbb_vendored}   # DEST: directory inside the fork
  [ -d "$dir/.git" ] || [ -f "$dir/.git" ] || { echo "$name: $dir is not a git checkout" >&2; return 1; }
  [ "$(git -C "$dir" branch --show-current)" = cuda-kernels ] || { echo "$name: checkout cuda-kernels before syncing" >&2; return 1; }
  [ -z "$(git -C "$dir" status --porcelain -- "$DEST")" ] || { echo "$name: $DEST has local changes; commit or move them before syncing" >&2; return 1; }
  echo "== $name ($(git -C "$dir" rev-parse --abbrev-ref HEAD)) <- $DEST/"
  rm -rf "$dir/$DEST"; mkdir -p "$dir/$DEST"
  local f src rel c latest=""
  for f in $files; do
    for src in $FB/fastbb/$f; do
      rel=${src#$FB/fastbb/}
      c=$(git -C "$FB" log -1 --format=%h -- "fastbb/$rel")
      mkdir -p "$dir/$DEST/$(dirname "$rel")"
      case $rel in
        *.py) { echo "# vendored from fastforks/fastbb: fastbb/$rel @ $c -- do not edit here, see fastbb/VENDORED.md"; cat "$src"; } > "$dir/$DEST/$rel";;
        *)    cp "$src" "$dir/$DEST/$rel";;
      esac
      latest="$latest $rel@$c"
    done
  done
  printf '"""Vendored copy of parts of fastforks/fastbb (own copy, no dependency on the fastbb package). Do not edit here:\nchange fastbb and run its tools/sync_vendored.sh.\nfiles @ last-changed fastbb commit:%s\n"""\n' \
    "$latest" > "$dir/$DEST/__init__.py"
  python "$FB/tools/trim_encoder_vendor.py" "$dir/$DEST"
  git -C "$dir" add -N "$DEST" 2>/dev/null || true
  git -C "$dir" --no-pager diff --stat -- "$DEST"
}
case "${1:-all}" in
  raft) sync_one raft "$RAFT_DIR" "$RAFT_FILES";;
  crestereo) sync_one crestereo "$CRESTEREO_DIR" "$CRESTEREO_FILES";;
  sam3) sync_one sam3 "$SAM3_DIR" "$SAM3_FILES" sam3/fastbb_vendored;;
  all) sync_one raft "$RAFT_DIR" "$RAFT_FILES"; sync_one crestereo "$CRESTEREO_DIR" "$CRESTEREO_FILES"
       sync_one sam3 "$SAM3_DIR" "$SAM3_FILES" sam3/fastbb_vendored;;
  *) echo "usage: $0 [raft|crestereo|sam3|all]" >&2; exit 2;;
esac
