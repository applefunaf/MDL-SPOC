import os
import sys
import glob
import numpy as np
import pandas as pd

def parse_pqr(pqr_file):
    """
    Parses a PQR file to extract coordinates, charges, radii, atom names and residue info.
    
    Args:
        pqr_file (str): Path to the .pqr file.
        
    Returns:
        pd.DataFrame: DataFrame containing atom information.
    """
    atoms = []
    with open(pqr_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    parts = line.split()
                    
                    # Standard PQR parsing logic (from back for safety on variable chain columns)
                    # Field order usually: Record, Serial, AtomName, ResName, (Chain), ResSeq, X, Y, Z, Q, R
                    
                    radius = float(parts[-1])
                    charge = float(parts[-2])
                    z = float(parts[-3])
                    y = float(parts[-4])
                    x = float(parts[-5])
                    
                    # Extract specific columns for centroid identification
                    # 3rd column (index 2) -> Atom Name
                    # 6th column (index 5) -> Residue ID (assuming chain present) or possibly X coordinate if missing
                    # We store it as a string primarily to check against '167'
                    
                    atom_name = parts[2] if len(parts) > 2 else ""
                    
                    # The prompt specifies "6th column". In 0-indexed split list, this is index 5.
                    col6 = parts[5] if len(parts) > 5 else ""

                    atom_info = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'charge': charge,
                        'radius': radius,
                        'atom_name': atom_name,
                        'col6': col6,
                        'line': line.strip()
                    }
                    atoms.append(atom_info)
                except (ValueError, IndexError):
                    continue
                    
    return pd.DataFrame(atoms)

def calculate_esp(df):
    """
    Calculates the Electrostatic Potential (ESP) for each atom.
    ESP_i = Sum( q_j / d_ij ) for j != i
    
    Args:
        df (pd.DataFrame): DataFrame with x, y, z, charge.
        
    Returns:
        np.array: Array of ESP values.
    """
    coords = df[['x', 'y', 'z']].values
    charges = df['charge'].values
    n = len(df)
    esp = np.zeros(n)
    
    # Calculate pairwise distances efficiently
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # Euclidean distances
    dists = np.sqrt(np.sum(delta**2, axis=-1))
    
    # Avoid division by zero on diagonal (self-interaction)
    # We treat self-potential as 0 for this interaction summation context (or handle infinite)
    with np.errstate(divide='ignore'):
        inv_dists = 1.0 / dists
    np.fill_diagonal(inv_dists, 0.0)
    
    # ESP calculation: Matrix vector multiplication
    # (N, N) * (N,) -> (N,)
    esp = inv_dists @ charges
    
    return esp

def calculate_centroid_distance(df):
    """
    Calculates the distance of each atom to a centroid atom using coordinates.
    Centroid condition: 3rd column (atom_name) == 'NZ' and 6th column (col6) == '167'.
    
    Args:
        df (pd.DataFrame): DataFrame with x, y, z, atom_name, col6.
        
    Returns:
        np.array: Array of distances.
    """
    # Filter for the centroid atom
    # Identifying column 3 as 'NZ' and column 6 as '167'
    # Note: col6 is a string from parsing
    centroid_atom = df[(df['atom_name'] == 'NZ') & (df['col6'] == '167')]
    
    if centroid_atom.empty:
        print("    Warning: Centroid atom (NZ, 167) not found. Setting distances to NaN.")
        return np.full(len(df), np.nan)
    
    # Use the first one found
    c_x = centroid_atom.iloc[0]['x']
    c_y = centroid_atom.iloc[0]['y']
    c_z = centroid_atom.iloc[0]['z']
    centroid_coords = np.array([c_x, c_y, c_z])
    
    atom_coords = df[['x', 'y', 'z']].values
    
    # Euclidean distance
    # sqrt(sum((x-c)^2))
    diff = atom_coords - centroid_coords
    dists = np.sqrt(np.sum(diff**2, axis=1))
    
    return dists

def process_directory(root_dir):
    print(f"Scanning {root_dir}...")
    pqr_files = glob.glob(os.path.join(root_dir, '**', '*.pqr'), recursive=True)
    
    print(f"Found {len(pqr_files)} PQR files.")
    
    for pqr_path in pqr_files:
        print(f"Processing {os.path.basename(pqr_path)}...")
        try:
            df = parse_pqr(pqr_path)
            if df.empty:
                print(f"  Warning: No valid atoms found in {pqr_path}")
                continue
                
            # Calculate ESP
            esp = calculate_esp(df)
            
            # Calculate Distance to Centroid (NZ 167)
            dists = calculate_centroid_distance(df)

            # Add to dataframe
            df['esp'] = esp
            df['distance_to_centroid'] = dists
            
            # Construct descriptor matrix
            # Selecting: Radius, ESP, and Distance
            descriptor_matrix = df[['radius', 'esp', 'distance_to_centroid']]
            
            # Save the matrix
            # Structure: original_name + _descriptor.csv
            output_path = pqr_path.replace('.pqr', '_descriptor.csv')
            
            # Ensure unique filename if .pqr is not extension (rare edge case)
            if output_path == pqr_path:
                 output_path += "_descriptor.csv"
                 
            descriptor_matrix.to_csv(output_path, index=False)
            print(f"  Saved descriptor to {output_path}")
            
        except Exception as e:
            print(f"  Error processing {pqr_path}: {e}")

if __name__ == "__main__":
    # Use current directory as root for scanning
    current_dir = os.getcwd() # Or specify explicitly if needed
    print(f"Working directory: {current_dir}")
    process_directory(current_dir)
