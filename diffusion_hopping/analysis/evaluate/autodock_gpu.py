import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem


def autodock_gpu_score(row, size=20.0, nrun=1, exhaustiveness=8, autodock_gpu_path=None):
    """
    Calculate AutoDock-GPU docking score for a molecule.
    
    Args:
        row: DataFrame row containing molecule and protein information
        size: Box size for docking (Angstroms)
        nrun: Number of docking runs (default 1 for speed)
        exhaustiveness: Search exhaustiveness parameter (mapped to nev)
        autodock_gpu_path: Path to autodock_gpu binary (if None, uses default)
    
    Returns:
        float: AutoDock-GPU docking score (binding energy in kcal/mol)
    """
    try:
        if row["molecule"] is None:
            return None
        
        protein_path = Path(row["test_set_item"]["protein"].path)
        
        # If path doesn't exist, try to resolve it relative to DATA_ROOT
        if not protein_path.exists():
            from config import DATA_ROOT
            # Extract the relative path from 'data/' onwards
            path_parts = protein_path.parts
            if 'data' in path_parts:
                data_idx = path_parts.index('data')
                relative_path = Path(*path_parts[data_idx + 1:])
                protein_path = DATA_ROOT / relative_path
                
                # If still doesn't exist, raise error with helpful message
                if not protein_path.exists():
                    raise FileNotFoundError(
                        f"Could not find protein file: {protein_path}\n"
                        f"Original path: {row['test_set_item']['protein'].path}\n"
                        f"DATA_ROOT: {DATA_ROOT}"
                    )
        
        ligand_path = row["molecule_path"]
        
        return _calculate_autodock_gpu_score(
            protein_path,
            ligand_path,
            row["molecule"],
            size=size,
            nrun=nrun,
            exhaustiveness=exhaustiveness,
            autodock_gpu_path=autodock_gpu_path,
        )
    except Exception as e:
        print(f"Error calculating AutoDock-GPU score: {e}")
        return None


def _prepare_protein_pdbqt(protein_path) -> Path:
    """
    Prepare protein PDBQT file using obabel.
    
    Args:
        protein_path: Path to protein PDB file
        
    Returns:
        Path to protein PDBQT file
    """
    protein_folder = protein_path.parent.resolve()
    protein_pdbqt = protein_folder / f"{protein_path.stem}.pdbqt"
    
    # Check if already exists
    if protein_pdbqt.exists():
        return protein_pdbqt

    # Use obabel to convert PDB to PDBQT
    command = [
        "obabel",
        str(protein_path),
        "-O",
        str(protein_pdbqt),
        "-xr",
        "--partialcharge", "gasteiger",
        "-p", "7.4"
    ]
    
    try:
        subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Obabel protein preparation failed: {e.stderr}")
        raise e
        
    return protein_pdbqt


def _prepare_ligand_pdbqt(ligand_path) -> Path:
    """
    Prepare ligand PDBQT file using obabel.
    
    Args:
        ligand_path: Path to ligand SDF/PDB file
        
    Returns:
        Path to ligand PDBQT file
    """
    ligand_folder = ligand_path.parent.resolve()
    ligand_pdbqt = ligand_folder / f"{ligand_path.stem}.pdbqt"
    
    # Check if already exists
    if ligand_pdbqt.exists():
        return ligand_pdbqt

    # Use obabel to convert to PDBQT
    command = [
        "obabel",
        str(ligand_path),
        "-O",
        str(ligand_pdbqt),
        "--partialcharge", "gasteiger",
        "-p", "7.4"
    ]
    
    try:
        subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Obabel ligand preparation failed: {e.stderr}")
        raise e
        
    return ligand_pdbqt


def _get_atom_types(pdbqt_path):
    """Extract unique atom types from a PDBQT file."""
    atom_types = set()
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Atom type is usually the last column (ignoring charge)
                # PDBQT format: ... charge atom_type
                parts = line.split()
                if len(parts) >= 2:
                     # Check if last part is a valid atom type (usually 1-2 chars)
                    candidate = parts[-1]
                    # Sometimes the last column is charge if type is missing, but PDBQT should have type.
                    # MGLTools/AutoDock usually puts type at the end.
                    # Let's assume the last whitespace-separated token is the type.
                    atom_types.add(candidate)
    return sorted(list(atom_types))

def _prepare_grid_parameter_file(protein_pdbqt, ligand_pdbqt, center, size, output_dir) -> Path:
    """
    Create AutoDock grid parameter file (.gpf).
    
    Args:
        protein_pdbqt: Path to protein PDBQT file
        ligand_pdbqt: Path to ligand PDBQT file
        center: Center coordinates [x, y, z]
        size: Box size
        output_dir: Directory to save GPF file
        
    Returns:
        Path to GPF file
    """
    gpf_path = output_dir / f"{protein_pdbqt.stem}.gpf"
    
    # Get atom types present in receptor and ligand
    receptor_types = _get_atom_types(protein_pdbqt)
    ligand_types = _get_atom_types(ligand_pdbqt)
    
    # Filter valid AutoDock atom types (sometimes non-standard ones appear)
    # Standard AD4 types: C A N O S P H NA MG MN FE ZN BR I CL F
    # HD is H donor, OA is O acceptor, SA is S acceptor
    # Maps needed: one for each ligand atom type + electrostatics + desolvation
    
    map_lines = []
    for atom_type in ligand_types:
        map_lines.append(f"map {protein_pdbqt.stem}.{atom_type}.map")

    # Format lists for GPF
    receptor_types_str = " ".join(receptor_types)
    ligand_types_str = " ".join(ligand_types)
    
    map_lines_str = ''.join(line + '\n' for line in map_lines)
    
    # AutoDock grid parameter file content
    gpf_content = f"""npts 40 40 40
gridfld {protein_pdbqt.stem}.maps.fld
spacing 0.375
receptor_types {receptor_types_str}
ligand_types {ligand_types_str}
receptor {protein_pdbqt.name}
gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
smooth 0.5
{map_lines_str}elecmap {protein_pdbqt.stem}.e.map
dsolvmap {protein_pdbqt.stem}.d.map
dielectric -0.1465
"""
    
    gpf_path.write_text(gpf_content)
    return gpf_path


def _run_autogrid(gpf_path, protein_pdbqt_dir):
    """
    Run autogrid to generate map files.
    
    Args:
        gpf_path: Path to grid parameter file
        protein_pdbqt_dir: Directory containing protein PDBQT
    """
    # Try to find autogrid4
    autogrid_cmd = None
    possible_paths = [
        "autogrid4",
        "/usr/local/bin/autogrid4",
        str(Path.home() / "MGLTools-1.5.7/bin/autogrid4"),
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run(
                [path, "-h"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0 or "AutoGrid" in result.stdout.decode() or "AutoGrid" in result.stderr.decode():
                autogrid_cmd = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if autogrid_cmd is None:
        raise RuntimeError(
            "autogrid4 not found. Please install AutoDock4 or MGLTools. "
            "AutoDock-GPU requires grid map files generated by autogrid4."
        )
    
    # Run autogrid
    command = [autogrid_cmd, "-p", str(gpf_path), "-l", str(protein_pdbqt_dir / f"{gpf_path.stem}.glg")]
    
    result = subprocess.run(
        command,
        cwd=str(protein_pdbqt_dir),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"autogrid4 failed: {result.stderr}")


def _calculate_autodock_gpu_score(
    protein_path, 
    ligand_path, 
    mol, 
    size=20.0, 
    nrun=1,
    exhaustiveness=8,
    autodock_gpu_path=None
) -> float | None:
    """
    Run AutoDock-GPU to calculate docking score.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand SDF file
        mol: RDKit molecule object
        size: Box size for docking
        nrun: Number of docking runs
        exhaustiveness: Search exhaustiveness (mapped to nev parameter)
        autodock_gpu_path: Path to autodock_gpu binary
    
    Returns:
        float: Docking score (binding energy)
    """
    # Calculate binding site center from molecule coordinates
    center = mol.GetConformer().GetPositions().mean(axis=0)
    
    # Prepare protein and ligand PDBQT files
    protein_pdbqt = _prepare_protein_pdbqt(protein_path)
    ligand_pdbqt = _prepare_ligand_pdbqt(ligand_path)
    
    # Create temporary directory for grid files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Copy protein PDBQT to temp dir (autogrid needs it in same dir)
        temp_protein_pdbqt = tmpdir_path / protein_pdbqt.name
        temp_protein_pdbqt.write_bytes(protein_pdbqt.read_bytes())
        
        # Create grid parameter file
        gpf_path = _prepare_grid_parameter_file(temp_protein_pdbqt, ligand_pdbqt, center, size, tmpdir_path)
        
        # Run autogrid to generate map files
        try:
            _run_autogrid(gpf_path, tmpdir_path)
        except RuntimeError as e:
            # If autogrid is not available, we cannot proceed
            print(f"Warning: {e}")
            return None
        
        # Determine autodock_gpu executable path
        if autodock_gpu_path is None:
            possible_paths = [
                "/mnt/SSD3/cong_nguyen/Code/pharmacy_code/AutoDock-GPU/bin/autodock_gpu_128wi",
                "autodock_gpu_128wi",
                "autodock_gpu_64wi",
            ]
            autodock_cmd = None
            for path in possible_paths:
                if Path(path).exists() or path.startswith("autodock_gpu"):
                    autodock_cmd = path
                    break
            if autodock_cmd is None:
                raise RuntimeError("AutoDock-GPU binary not found.")
        else:
            autodock_cmd = autodock_gpu_path
        
        # Build AutoDock-GPU command
        fld_file = tmpdir_path / f"{temp_protein_pdbqt.stem}.maps.fld"
        output_name = tmpdir_path / "docking_result"
        
        # Map exhaustiveness to nev (number of evaluations)
        # Higher exhaustiveness = more evaluations
        nev = exhaustiveness * 312500  # Scale factor similar to Vina
        
        command = [
            autodock_cmd,
            "--ffile", str(fld_file),
            "--lfile", str(ligand_pdbqt),
            "--nrun", str(nrun),
            "--nev", str(nev),
            "--resnam", str(output_name),
            "--dlgoutput", "1",
            "--xmloutput", "0",
        ]
        
        # Execute AutoDock-GPU
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(tmpdir_path)
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"AutoDock-GPU returned non-zero return code {result.returncode}\\n"
                f"stderr: {result.stderr}"
            )
        
        # Parse output DLG file to extract best binding energy
        dlg_file = tmpdir_path / f"{output_name.name}.dlg"
        
        if not dlg_file.exists():
            raise RuntimeError(f"AutoDock-GPU did not produce expected output file: {dlg_file}")
        
        # Parse DLG file for lowest binding energy
        best_energy = None
        with open(dlg_file, 'r') as f:
            for line in f:
                # Look for lines like: "RANKING    1      -7.52      0.00      0.00"
                if line.strip().startswith("RANKING"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "1":
                        # Second value after "1" is the binding energy
                        best_energy = float(parts[2])
                        break
                # Alternative: look for "Estimated Free Energy of Binding"
                elif "Estimated Free Energy of Binding" in line:
                    # Format: "Estimated Free Energy of Binding    =   -7.52 kcal/mol"
                    parts = line.split("=")
                    if len(parts) >= 2:
                        energy_str = parts[1].strip().split()[0]
                        best_energy = float(energy_str)
        
        if best_energy is None:
            raise RuntimeError(f"Could not parse binding energy from AutoDock-GPU output: {dlg_file}")
        
        return best_energy
