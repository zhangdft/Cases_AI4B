
import os
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.io.cif import CifWriter

def prepare_jarvis_data():
    # Load dataset
    print("Loading jarvis_dft_3d dataset (this might take a while)...")
    df = load_dataset("jarvis_dft_3d")
    
    # Jarvis shear modulus column name is 'shear modulus'
    # 'jid' is the unique identifier
    data = df[['structure', 'shear modulus', 'jid']].dropna(subset=['shear modulus'])
    
    # For teaching purpose, we take the first 5000 entries to speed up
    print(f"Total entries with shear modulus: {len(data)}")
    data = data.head(5000)
    print(f"Taking first {len(data)} entries for teaching example.")
    
    output_dir = "/data/home/5240019/class_AI4Bat/Case_3/data/jarvis_dft_3d"
    os.makedirs(output_dir, exist_ok=True)
    
    id_prop_list = []
    
    print(f"Processing entries...")
    for i, row in data.iterrows():
        struct = row['structure']
        shear_modulus = row['shear modulus']
        jid = row['jid']
        
        # Save as CIF
        cif_filename = f"{jid}.cif"
        cif_path = os.path.join(output_dir, cif_filename)
        # Check if already exists to skip
        if not os.path.exists(cif_path):
            try:
                CifWriter(struct).write_file(cif_path)
            except Exception as e:
                print(f"Error writing {jid}: {e}")
                continue
        
        # Add to id_prop.csv list
        id_prop_list.append([jid, shear_modulus])
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(data)} entries")
            
    # Save id_prop.csv
    id_prop_df = pd.DataFrame(id_prop_list)
    id_prop_df.to_csv(os.path.join(output_dir, "id_prop.csv"), index=False, header=False)
    
    # Also need atom_init.json - copy from cgcnn data
    source_atom_init = "/data/home/5240019/class_AI4Bat/cgcnn/data/sample-regression/atom_init.json"
    target_atom_init = os.path.join(output_dir, "atom_init.json")
    if os.path.exists(source_atom_init):
        import shutil
        shutil.copyfile(source_atom_init, target_atom_init)
    
    print(f"Done! Data saved to {output_dir}")

if __name__ == "__main__":
    prepare_jarvis_data()
