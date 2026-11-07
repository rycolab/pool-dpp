#!/bin/sh
# One-time setup for the Lean formalization of Theorems 5.1 and 5.5.
# Installs elan (Lean's toolchain manager) into ~/.elan without editing your
# shell config, then fetches Mathlib and its prebuilt cache (~8 GB in .lake/).
# Versions are pinned: Lean by lean-toolchain, Mathlib by the commit in lake-manifest.json.
# Do not run `lake update`. It moves Mathlib to its latest commit, and the proofs may not
# compile there.
set -e
cd "$(dirname "$0")"

if [ ! -x "$HOME/.elan/bin/lake" ]; then
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /tmp/elan-init.sh
  sh /tmp/elan-init.sh -y --default-toolchain none --no-modify-path
fi
export PATH="$HOME/.elan/bin:$PATH"

lake exe cache get
echo "done: $(du -sh .lake | cut -f1) in .lake"
