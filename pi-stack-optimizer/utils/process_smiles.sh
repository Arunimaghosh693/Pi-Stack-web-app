#!/usr/bin/env bash
# Process CSV file with SMILES and run calc_dihedrals.py and optionally main.py
# Creates individual directories for each molecule

set -e

# Default parameters
INPUT_CSV=""
OUTPUT_DIR="smiles_results"
MAX_ROWS=0
START_MOL=1
END_MOL=0
SMILES_COL="SMILES"
ID_COL="ID"
VERBOSE=false
OPT_METHOD="both"
WORKERS=30
SWARM_SIZE=60
STEP_MODE="both"  # "torsions", "optimization", "both"

# Command definitions - modify these as needed
CALC_CMD_TEMPLATE='python3 "$CALC" --format smiles --input "$SMILES" --output torsions.json --one-based --opt-method $OPT_METHOD'
MAIN_CMD_TEMPLATE='python3 "$MAIN" output.xyz --torsions torsions.json --workers $WORKERS --swarm-size $SWARM_SIZE'

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALC="$SCRIPT_DIR/calc_dihedrals.py"
MAIN="$SCRIPT_DIR/main.py"

# Parse command line arguments
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
        --max-rows)
            MAX_ROWS="$2"
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
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --swarm-size)
            SWARM_SIZE="$2"
            shift 2
            ;;
        --opt-method)
            OPT_METHOD="$2"
            shift 2
            ;;
        --step-mode)
            STEP_MODE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Process CSV file with SMILES and run torsion generation and/or PSO optimization"
            echo ""
            echo "Step modes:"
            echo "  torsions      - Only generate torsions from SMILES (calc_dihedrals.py)"
            echo "  optimization  - Only run PSO optimization (requires existing torsions)"
            echo "  both          - Run both steps sequentially (default)"
            echo ""
            echo "Options:"
            echo "  -i, --input CSV_FILE     Input CSV file (required)"
            echo "  -o, --output-dir DIR     Output directory (default: smiles_results)"
            echo "  --max-rows N             Process only first N rows (0=all, default: 0)"
            echo "  --start-mol N            Start from molecule number N (1-based, default: 1)"
            echo "  --end-mol N              End at molecule number N (0=all, default: 0)"
            echo "  --smiles-col COLUMN      SMILES column name (default: SMILES)"
            echo "  --id-col COLUMN          ID column name (default: ID)"
            echo "  --workers N              Number of workers (default: 30)"
            echo "  --swarm-size N           PSO swarm size (default: 60)"
            echo "  --opt-method METHOD      Optimization method: uff,mmff,both,none (default: both)"
            echo "  --step-mode MODE         Processing mode: torsions, optimization, both (default: both)"
            echo "  -v, --verbose            Enable verbose output"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Outputs:"
            echo "  - {OUTPUT_DIR}/{ID}/     Individual directories for each molecule"
            echo "  - summary.csv            Processing summary"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check required input
if [[ -z "$INPUT_CSV" ]]; then
    echo "Error: Input CSV file is required. Use -i or --input to specify."
    exit 1
fi

# Validate step mode
if [[ "$STEP_MODE" != "torsions" && "$STEP_MODE" != "optimization" && "$STEP_MODE" != "both" ]]; then
    echo "Error: Invalid step-mode '$STEP_MODE'. Must be 'torsions', 'optimization', or 'both'."
    exit 1
fi

# Check if input file exists
if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Error: Input CSV file '$INPUT_CSV' not found"
    exit 1
fi

# Check if required scripts exist
if [[ "$STEP_MODE" == "torsions" || "$STEP_MODE" == "both" ]] && [[ ! -f "$CALC" ]]; then
    echo "Error: calc_dihedrals.py not found in $SCRIPT_DIR"
    exit 1
fi

if [[ "$STEP_MODE" == "optimization" || "$STEP_MODE" == "both" ]] && [[ ! -f "$MAIN" ]]; then
    echo "Error: main.py not found in $SCRIPT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize summary file
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"
echo "ID,SMILES,Status,Message,Directory,N_Torsions" > "$SUMMARY_FILE"

echo "Processing $INPUT_CSV..."
echo "SMILES column: $SMILES_COL"
echo "ID column: $ID_COL"
echo "Output directory: $OUTPUT_DIR"
echo "Step mode: $STEP_MODE"
if [[ "$STEP_MODE" == "optimization" || "$STEP_MODE" == "both" ]]; then
    echo "Workers: $WORKERS, Swarm size: $SWARM_SIZE"
fi
if [[ $END_MOL -gt 0 ]]; then
    echo "Processing molecules $START_MOL to $END_MOL"
else
    echo "Processing from molecule $START_MOL to end"
fi
echo ""

# Counters
PROCESSED=0
SUCCESS_COUNT=0
FAILED_COUNT=0

# Process each row using Python to properly parse CSV
python3 -c "
import csv
import sys

with open('$INPUT_CSV', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        if i < $START_MOL:
            continue
        if $END_MOL > 0 and i > $END_MOL:
            break
        if $MAX_ROWS > 0 and (i - $START_MOL + 1) > $MAX_ROWS:
            break
        mol_id = row.get('$ID_COL', f'mol_{i}')
        smiles = row.get('$SMILES_COL', '')
        # Use base64 encoding to avoid shell parsing issues with SMILES
        import base64
        smiles_b64 = base64.b64encode(smiles.encode()).decode()
        print(f'{i}|{mol_id}|{smiles_b64}')
" | while IFS='|' read -r ROW_NUM MOL_ID SMILES_B64; do
    
    # Decode SMILES from base64
    SMILES=$(python3 -c "import base64; print(base64.b64decode('$SMILES_B64').decode())" 2>/dev/null || echo "")
    
    if [[ $VERBOSE == true && $((ROW_NUM % 10)) -eq 0 ]]; then
        echo "Processing row $ROW_NUM: ID=$MOL_ID"
    fi
    
    ((PROCESSED++))
    
    # Skip if SMILES is empty
    if [[ -z "$SMILES" ]]; then
        echo "$MOL_ID,\"$SMILES\",failed,\"Empty SMILES\",,0" >> "$SUMMARY_FILE"
        ((FAILED_COUNT++))
        continue
    fi
    
    # Create individual directory for this molecule
    MOL_DIR="$OUTPUT_DIR/$MOL_ID"
    mkdir -p "$MOL_DIR"
    cd "$MOL_DIR" || { echo "Failed to cd to $MOL_DIR"; continue; }
    
    echo "---- Processing: $MOL_ID ----"
    
    # Step 1: Generate torsions and XYZ from SMILES (if requested)
    if [[ "$STEP_MODE" == "torsions" || "$STEP_MODE" == "both" ]]; then
        CALC_CMD=$(eval echo "$CALC_CMD_TEMPLATE")
        if [[ $VERBOSE == true ]]; then
            CALC_CMD="$CALC_CMD --verbose"
        fi
        
        echo "Running calc_dihedrals.py (logs -> calc_dihedrals.log)"
        if ! eval $CALC_CMD > calc_dihedrals.log 2>&1; then
            ERROR_MSG=$(head -n 3 calc_dihedrals.log | tr '\n' ' ' | tr '"' "'" | head -c 200)
            echo "$MOL_ID,\"$SMILES\",failed,\"calc_dihedrals failed: $ERROR_MSG\",\"$MOL_DIR\",0" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "calc_dihedrals.py failed for $MOL_ID — see calc_dihedrals.log"
            continue
        fi
        
        # Check if torsions.json and output.xyz were created
        if [[ ! -f "torsions.json" ]]; then
            echo "$MOL_ID,\"$SMILES\",failed,\"No torsions.json generated\",\"$MOL_DIR\",0" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "No torsions.json generated for $MOL_ID"
            continue
        fi
        
        if [[ ! -f "output.xyz" ]]; then
            echo "$MOL_ID,\"$SMILES\",failed,\"No output.xyz generated\",\"$MOL_DIR\",0" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "No output.xyz generated for $MOL_ID"
            continue
        fi
        
        echo "Torsions generated successfully for $MOL_ID"
    fi
    
    # Extract number of torsions
    if [[ -f "torsions.json" ]]; then
        N_TORSIONS=$(python3 -c "import json; data=json.load(open('torsions.json')); print(len(data.get('torsions', [])))" 2>/dev/null || echo "0")
    else
        N_TORSIONS="0"
    fi
    
    # Step 2: Run PSO optimization (if requested)
    if [[ "$STEP_MODE" == "optimization" || "$STEP_MODE" == "both" ]]; then
        # Check prerequisites for optimization
        if [[ ! -f "torsions.json" ]]; then
            echo "$MOL_ID,\"$SMILES\",failed,\"No torsions.json found for optimization\",\"$MOL_DIR\",$N_TORSIONS" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "No torsions.json found for optimization of $MOL_ID"
            continue
        fi
        
        if [[ ! -f "output.xyz" ]]; then
            echo "$MOL_ID,\"$SMILES\",failed,\"No output.xyz found for optimization\",\"$MOL_DIR\",$N_TORSIONS" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "No output.xyz found for optimization of $MOL_ID"
            continue
        fi
        
        MAIN_CMD=$(eval echo "$MAIN_CMD_TEMPLATE")
        
        echo "Running main.py (logs -> main.log)"
        if ! eval $MAIN_CMD > main.log 2>&1; then
            ERROR_MSG=$(head -n 3 main.log | tr '\n' ' ' | tr '"' "'" | head -c 200)
            echo "$MOL_ID,\"$SMILES\",failed,\"main.py failed: $ERROR_MSG\",\"$MOL_DIR\",$N_TORSIONS" >> "$SUMMARY_FILE"
            ((FAILED_COUNT++))
            echo "main.py failed for $MOL_ID — see main.log"
            continue
        fi
        
        echo "Optimization completed for $MOL_ID"
    fi
    
    # Success
    STATUS_MSG="OK"
    if [[ "$STEP_MODE" == "torsions" ]]; then
        STATUS_MSG="Torsions generated"
    elif [[ "$STEP_MODE" == "optimization" ]]; then
        STATUS_MSG="Optimization completed"
    fi
    
    echo "$MOL_ID,\"$SMILES\",success,\"$STATUS_MSG\",\"$MOL_DIR\",$N_TORSIONS" >> "$SUMMARY_FILE"
    ((SUCCESS_COUNT++))
    
    if [[ $VERBOSE == true ]]; then
        echo "  ✓ ID $MOL_ID: $N_TORSIONS torsions, $STATUS_MSG"
    fi
    
    echo "Completed: $MOL_ID"
    echo
done

echo ""
echo "Processing complete!"
echo "Processed: $PROCESSED molecules"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Summary written to: $SUMMARY_FILE"
echo ""
echo "Output files in $OUTPUT_DIR:"
echo "  - {ID}/                  (individual molecule directories)"
echo "  - {ID}/torsions.json     (torsion data)"
echo "  - {ID}/output.xyz        (optimized geometry)"
echo "  - {ID}/calc_dihedrals.log"
echo "  - {ID}/main.log"
echo "  - summary.csv            (processing summary)"