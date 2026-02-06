import subprocess
from pathlib import Path

from rdkit import Chem


def gnina_score(row, size=20.0, exhaustiveness=8, cnn_scoring='rescore', gnina_path=None):
    """
    Calculate gnina docking score for a molecule.
    
    Args:
        row: DataFrame row containing molecule and protein information
        size: Box size for docking (Angstroms)
        exhaustiveness: Search exhaustiveness parameter (default 8)
        cnn_scoring: CNN scoring mode ('none', 'rescore', 'refinement', 'all')
        gnina_path: Path to gnina binary (if None, assumes gnina is in PATH)
    
    Returns:
        float: Gnina docking score (CNNaffinity or CNNscore)
    """
    try:
        if row["molecule"] is None:
            return None
        
        protein_path = row["test_set_item"]["protein"].path
        ligand_path = row["molecule_path"]
        
        return _calculate_gnina_score(
            protein_path,
            ligand_path,
            row["molecule"],
            size=size,
            exhaustiveness=exhaustiveness,
            cnn_scoring=cnn_scoring,
            gnina_path=gnina_path,
        )
    except Exception as e:
        print(f"Error calculating gnina score: {e}")
        return None


def _calculate_gnina_score(
    protein_path, 
    ligand_path, 
    mol, 
    size=20.0, 
    exhaustiveness=8,
    cnn_scoring='rescore',
    gnina_path=None
) -> float:
    """
    Run gnina to calculate docking score.
    
    Uses --minimize flag to score the ligand in its current position.
    Accepts PDB for protein and SDF for ligand.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand SDF file
        mol: RDKit molecule object
        size: Box size for docking
        exhaustiveness: Search exhaustiveness
        cnn_scoring: CNN scoring mode
        gnina_path: Path to gnina binary
    
    Returns:
        float: Docking score
    """
    # Calculate binding site center from molecule coordinates
    center = mol.GetConformer().GetPositions().mean(axis=0)
    
    # Determine gnina executable path
    if gnina_path is None:
        # Try common locations
        possible_paths = [
            "/mnt/SSD3/cong_nguyen/Code/pharmacy_code/gnina/build/gnina",
            "gnina",  # Assumes it's in PATH
        ]
        gnina_cmd = None
        for path in possible_paths:
            if Path(path).exists() or path == "gnina":
                gnina_cmd = path
                break
        if gnina_cmd is None:
            raise RuntimeError("gnina binary not found. Please build gnina or add it to PATH.")
    else:
        gnina_cmd = gnina_path
    
    # Build gnina command
    command = (
        f"{gnina_cmd} "
        f"-r {protein_path.resolve()} "
        f"-l {ligand_path.resolve()} "
        f"--center_x {center[0]:.3f} "
        f"--center_y {center[1]:.3f} "
        f"--center_z {center[2]:.3f} "
        f"--size_x {size} "
        f"--size_y {size} "
        f"--size_z {size} "
        f"--exhaustiveness {exhaustiveness} "
        f"--cnn_scoring {cnn_scoring} "
        f"--minimize"  # Score in current position without full docking
    )
    
    # Execute gnina
    result = subprocess.run(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Gnina returned non-zero return code {result.returncode} when running '{command}'\n"
            f"stderr: {result.stderr}"
        )
    
    # Parse output to extract score
    # Gnina outputs scores in a table format similar to Vina
    # Look for lines like:
    #    1        -7.5      0.000      0.000
    # where the second column is the affinity score
    
    out_lines = result.stdout.splitlines()
    
    # Try to parse the results table (standard Vina/Gnina output)
    for i, line in enumerate(out_lines):
        if line.strip().startswith("-----"):
            # Next line should contain the first result
            if i + 1 < len(out_lines):
                result_line = out_lines[i + 1].strip()
                parts = result_line.split()
                if len(parts) >= 2 and parts[0] == "1":
                    # Second column is the affinity score
                    score = float(parts[1])
                    return score
            break
            
    # If table parsing failed, try to parse 'Affinity:' line (common with --minimize)
    # Example: Affinity: -5.72376  0.00000 (kcal/mol)
    for line in out_lines:
        if line.strip().startswith("Affinity:"):
            parts = line.split()
            if len(parts) >= 2:
                # The second part should be the score
                try:
                    score = float(parts[1])
                    return score
                except ValueError:
                    continue

    
    # Filter out Open Babel warnings from stderr to avoid cluttering error messages
    # These warnings are usually about CONECT records in PDB files and are often harmless for our purpose
    if result.stderr:
        clean_stderr_lines = []
        is_warning_block = False
        for line in result.stderr.splitlines():
            if "*** Open Babel Warning" in line:
                is_warning_block = True
                continue
            if is_warning_block and line.strip().startswith("="):
                is_warning_block = False
                continue
            if not is_warning_block and not "THIS CONECT RECORD WILL BE IGNORED" in line and not "Problems reading a" in line and not "According to the PDB" in line:
                 clean_stderr_lines.append(line)
        cleaned_stderr = "\n".join(clean_stderr_lines)
    else:
        cleaned_stderr = ""

    # If we couldn't parse the score, raise an error
    raise RuntimeError(
        f"Could not parse gnina output. Command: {command}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {cleaned_stderr}"
    )
