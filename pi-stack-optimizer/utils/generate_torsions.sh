#!/usr/bin/env bash
# Generate torsions from SMILES in CSV file
# Creates individual directories for each molecule with torsions.json and output.xyz

set -euo pipefail

# Default parameters
INPUT_CSV=""
OUTPUT_DIR="torsions_output"
START_MOL=1
END_MOL=0
SMILES_COL="SMILES"
ID_COL="ID"
OPT_METHOD="both"

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALC_SCRIPT="$SCRIPT_DIR/calc_dihedrals.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_CSV="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --start-mol)
            START_MOL="$2"
            shift 2
            ;;
        --end-mol)
            END_MOL="$2"
            shift 2
            ;;
        --smiles-col)
            SMILES_COL="$2"
            shift 2
            ;;
        --id-col)
            ID_COL="$2"
            shift 2
            ;;
        --opt-method)
            OPT_METHOD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -i INPUT_CSV [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  -i, --input CSV_FILE     Input CSV file with SMILES"
            echo ""
            echo "Options:"
            echo "  -o, --output-dir DIR     Output directory (default: torsions_output)"
            echo "  --start-mol N            Start from molecule N (default: 1)"
            echo "  --end-mol N              End at molecule N (0=all, default: 0)"
            echo "  --smiles-col NAME        SMILES column name (default: SMILES)"
            echo "  --id-col NAME            ID column name (default: ID)"
            echo "  --opt-method METHOD      uff, mmff, both, none (default: both)"
            echo "  -h, --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_CSV" ]]; then
    echo "Error: Input CSV required. Use -i or --input"
    exit 1
fi

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Error: File not found: $INPUT_CSV"
    exit 1
fi

if [[ ! -f "$CALC_SCRIPT" ]]; then
    echo "Error: calc_dihedrals.py not found at: $CALC_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize summary
SUMMARY="$OUTPUT_DIR/summary.csv"
echo "ID,SMILES,Status,Message,N_Torsions" > "$SUMMARY"

echo "=== Torsion Generation from SMILES ==="
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_DIR"
echo "Molecules: $START_MOL to $([ $END_MOL -eq 0 ] && echo 'end' || echo $END_MOL)"
echo ""

PROCESSED=0
SUCCESS=0
FAILED=0

# Create temporary Python script to parse CSV
TEMP_PY=$(mktemp)
cat > "$TEMP_PY" << 'PYTHON_SCRIPT'
import csv
import sys
import base64

input_csv = sys.argv[1]
start_mol = int(sys.argv[2])
end_mol = int(sys.argv[3])
smiles_col = sys.argv[4]
id_col = sys.argv[5]

with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        if i < start_mol:
            continue
        if end_mol > 0 and i > end_mol:
            break
        
        mol_id = row.get(id_col, f'mol_{i}')
        smiles = row.get(smiles_col, '')
        
        # Encode SMILES in base64 to safely pass through shell
        smiles_b64 = base64.b64encode(smiles.encode()).decode()
        print(f'{i},{mol_id},{smiles_b64}')
PYTHON_SCRIPT

# Process CSV
python3 "$TEMP_PY" "$INPUT_CSV" "$START_MOL" "$END_MOL" "$SMILES_COL" "$ID_COL" | while IFS=',' read -r row_num mol_id smiles_b64; do
    
    # Decode SMILES
    SMILES=$(python3 -c "import base64; print(base64.b64decode('$smiles_b64').decode())")
    
    ((PROCESSED++))
    
    # Skip empty SMILES
    if [[ -z "$SMILES" ]]; then
        echo "$mol_id,\"\",failed,\"Empty SMILES\",0" >> "$SUMMARY"
        ((FAILED++))
        echo "[$row_num] $mol_id: SKIPPED (empty SMILES)"
        continue
    fi
    
    # Create molecule directory
    MOL_DIR="$OUTPUT_DIR/$mol_id"
    mkdir -p "$MOL_DIR"
    
    echo "[$row_num] Processing $mol_id..."
    
    # Run calc_dihedrals.py
    cd "$MOL_DIR"
    if python3 "$CALC_SCRIPT" \
        --format smiles \
        --input "$SMILES" \
        --output torsions.json \
        --one-based \
        --opt-method "$OPT_METHOD" \
        > calc_dihedrals.log 2>&1; then
        
        # Check outputs
        if [[ -f "torsions.json" && -f "output.xyz" ]]; then
            N_TORSIONS=$(python3 -c "import json; print(len(json.load(open('torsions.json')).get('torsions', [])))" 2>/dev/null || echo "0")
            echo "$mol_id,\"${SMILES:0:50}...\",success,\"OK\",$N_TORSIONS" >> "$SUMMARY"
            ((SUCCESS++))
            echo "  ✓ Generated $N_TORSIONS torsions"
        else
            echo "$mol_id,\"${SMILES:0:50}...\",failed,\"Output files missing\",0" >> "$SUMMARY"
            ((FAILED++))
            echo "  ✗ Output files missing"
        fi
    else
        ERROR=$(head -n 1 calc_dihedrals.log)
        echo "$mol_id,\"${SMILES:0:50}...\",failed,\"${ERROR:0:100}\",0" >> "$SUMMARY"
        ((FAILED++))
        echo "  ✗ Failed (see $MOL_DIR/calc_dihedrals.log)"
    fi
    
    cd - > /dev/null
done

# Cleanup
rm -f "$TEMP_PY"

echo ""
echo "=== Summary ==="
echo "Processed: $PROCESSED"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"
echo "Results: $SUMMARY"
