
import csv
import subprocess
import os
from pathlib import Path

def main():
    input_csv = "example.csv"
    output_dir = Path("linear_output")
    output_dir.mkdir(exist_ok=True)
    
    script_path = "smiles_to_xyz_linearize.py"
    
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
                
            mol_id = row[0].strip()
            smiles = row[3].strip()
            
            if not smiles:
                continue
                
            print(f"Processing ID: {mol_id}")
            output_xyz = output_dir / f"{mol_id}.xyz"
            report_json = output_dir / f"{mol_id}.json"
            
            cmd = [
                ".venv/bin/python3", script_path,
                "--smiles", smiles,
                "--out", str(output_xyz),
                "--report-json", str(report_json),
                "--print-summary"
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                # print(result.stdout) # Optional: print individual summaries
            except subprocess.CalledProcessError as e:
                print(f"Error processing {mol_id}:")
                print(e.stderr)
                
    print(f"\nBatch processing complete. Outputs in {output_dir}")

if __name__ == "__main__":
    main()
