import os
import glob
import numpy as np
import pandas as pd

def process_pqr_and_save(pqr_path):
    """
    Parses a PQR file, calculates ESP for each atom, and saves a descriptor matrix.
    Reference: Modeled after HSPOC structural descriptor generation, but adapted for PQR inputs.
    """
    # Define PQR columns (Standard PDB2PQR format)
    # Field 0: Record (ATOM/HETATM)
    # Field 1: Serial
    # Field 2: Atom Name
    # Field 3: Residue Name
    # Field 4: Chain ID
    # Field 5: Residue ID
    # Field 6-8: X, Y, Z Coordinates
    # Field 9: Charge (Q)
    # Field 10: Radius (R)
    
    data = []
    try:
        with open(pqr_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    parts = line.split()
                    # Check for minimal column count (handling optional chainID or merged columns slightly loosely)
                    # We expect at least 10 fields if chain is missing, or 11 if present.
                    # Standard format usually has clear whitespace separation.
                    
                    if len(parts) >= 10:
                        # Heuristic to detect if chain ID is present
                        # If column 4 is a single letter and column 5 is an integer, likely Chain is present.
                        # If column 4 is an int, Chain might be missing or merged.
                        
                        # In the observed file: ATOM 1 N LEU A 4 ... (11 parts)
                        if len(parts) >= 11:
                            chain = parts[4]
                            res_id = parts[5]
                            x_idx, y_idx, z_idx = 6, 7, 8
                            q_idx, r_idx = 9, 10
                        else:
                            # Guessing format without chain or non-standard
                            chain = 'X'
                            res_id = parts[4]
                            x_idx, y_idx, z_idx = 5, 6, 7
                            q_idx, r_idx = 8, 9

                        try:
                            row = {
                                'Record': parts[0],
                                'Atom_Index': int(parts[1]), # Serial
                                'Atom_Name': parts[2],
                                'Residue_Name': parts[3],
                                'Chain_ID': chain,
                                'Residue_ID': int(res_id),
                                'x': float(parts[x_idx]),
                                'y': float(parts[y_idx]),
                                'z': float(parts[z_idx]),
                                'Charge': float(parts[q_idx]),
                                'vdW_Radius': float(parts[r_idx])
                            }
                            data.append(row)
                        except ValueError as e:
                            # Skip lines that don't parse as expected (headers etc)
                            continue
                            
        if not data:
            print(f"Warning: No valid atom data found in {pqr_path}")
            return

        df = pd.DataFrame(data)

        # --- Calculation of Electrostatic Potential (ESP) ---
        # Method: Direct Coulomb summation: V_i = Sum_j(q_j / r_ij) for j != i
        
        coords = df[['x', 'y', 'z']].values
        charges = df['Charge'].values
        
        # Compute residues distance matrix (N x N)
        # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        
        # Euclidean distance
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Handle diagonal (self-interaction) to avoid division by zero
        # 1/0 -> inf. We can set diagonal to infinity so 1/d becomes 0
        np.fill_diagonal(dists, np.inf)
        
        inv_dists = 1.0 / dists
        
        # Calculate ESP
        # Matrix multiplication: (N x N) * (N x 1) -> (N x 1)
        # Result is sum(1/r_ij * q_j)
        esp = inv_dists @ charges
        
        df['ESP'] = esp
        
        # --- Calculation of Distance to Centroid ---
        # Centroid: 3rd column (Atom_Name) == 'NZ', 6th column (Residue_ID) == '167'
        # In our dataframe, Atom_Name is parts[2], Residue_ID is parts[5] (if chain exists) or parts[4]
        # Our previous parsing logic tried to normalize into 'Atom_Name' and 'Residue_ID'.
        # However, checking "6th column" strictly means index 5 in the split parts.
        
        # We need to rely on the columns we extracted.
        # But 'Residue_ID' might have come from column 5 or 4.
        
        # Let's find index of atom with Atom_Name == 'NZ' and Residue_ID == 167 
        # (Assuming 167 was indeed the 6th column, which maps to Residue ID usually)
        
        # Since 'Residue_ID' is int, compare with 167
        centroid_mask = (df['Atom_Name'] == 'NZ') & (df['Residue_ID'] == 167)
        if centroid_mask.any():
            centroid_row = df[centroid_mask].iloc[0]
            c_coords = centroid_row[['x', 'y', 'z']].values.astype(float)
            
            # Distance from all atoms to C
            atom_coords = df[['x', 'y', 'z']].values
            dist_to_centroid = np.sqrt(np.sum((atom_coords - c_coords)**2, axis=1))
            
            df['Dist_to_Centroid'] = dist_to_centroid
        else:
            print(f"Warning: Centroid (NZ 167) not found in {os.path.basename(pqr_path)}.")
            df['Dist_to_Centroid'] = np.nan

        # --- Integration into Descriptor Matrix ---
        # The DataFrame itself is the matrix. 
        # Selecting key descriptors: Index, Identity, Geometry (Coord), Properties (Q, R, ESP, Dist)
        
        output_cols = ['Atom_Index', 'Atom_Name', 'Residue_Name', 'Chain_ID', 'Residue_ID', 
                      'x', 'y', 'z', 'Charge', 'vdW_Radius', 'ESP', 'Dist_to_Centroid']
        
        result_df = df[output_cols]
        
        # Generate output filename
        # Structure: original_name + _descriptors.csv
        out_path = os.path.splitext(pqr_path)[0] + "_descriptors.csv"
        
        result_df.to_csv(out_path, index=False)
        print(f"Processed: {pqr_path} -> {os.path.basename(out_path)}")
        
    except Exception as e:
        print(f"Error processing {pqr_path}: {str(e)}")

def main():
    # Directory setup
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Scanning for .pqr files in {root_dir}...")
    pqr_files = glob.glob(os.path.join(root_dir, '**', '*.pqr'), recursive=True)
    
    print(f"Found {len(pqr_files)} .pqr files.")
    
    for pqr in pqr_files:
        process_pqr_and_save(pqr)
        
    print("All tasks completed.")

if __name__ == '__main__':
    main()
