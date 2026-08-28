#!/usr/bin/env bash
# Iterate immediate subdirectories, run calc_dihedrals.py then main.py in each

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(pwd)"
CALC="$SCRIPT_DIR/calc_dihedrals.py"
MAIN="$SCRIPT_DIR/main.py"

# Command definitions - modify these as needed
CALC_CMD='python3 "$CALC" --input "$xyz"'
MAIN_CMD='python3 "$MAIN" "$xyz" --torsions torsions.json --workers 30 --swarm-size 60'

if [[ ! -f "$CALC" || ! -f "$MAIN" ]]; then
  echo "Error: calc_dihedrals.py and/or main.py not found in $SCRIPT_DIR" >&2
  exit 1
fi

echo "Base dir: $WORK_DIR"
echo "Scripts from: $SCRIPT_DIR"

for d in "$WORK_DIR"/*/ ; do
  [ -d "$d" ] || continue
  dirn=$(basename "$d")
  echo "---- Processing: $dirn ----"
  cd "$d" || { echo "Cannot cd to $d"; continue; }

  # find first .xyz file in the directory
  xyz=$(ls *.xyz 2>/dev/null | head -n1 || true)
  if [[ -z "$xyz" ]]; then
    echo "No .xyz file found in $d — skipping"
    continue
  fi

  echo "Found XYZ: $xyz"
  echo "Running calc_dihedrals.py on $xyz (logs -> calc_dihedrals.log)"
  if ! eval $CALC_CMD > calc_dihedrals.log 2>&1; then
    echo "calc_dihedrals.py failed for $dirn — see calc_dihedrals.log"
    continue
  fi

  echo "Running main.py (logs -> main.log)"
  if ! eval $MAIN_CMD > main.log 2>&1; then
    echo "main.py failed for $dirn — see main.log"
    continue
  fi

  echo "Completed: $dirn"
  echo
done

echo "All directories processed."
