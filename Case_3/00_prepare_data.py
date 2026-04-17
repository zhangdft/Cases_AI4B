
import os
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.io.cif import CifWriter

def prepare_data():
    # Load dataset
    print("Loading elastic_tensor_2015 dataset...")
    df = load_dataset("elastic_tensor_2015")
    
    # We want shear modulus: G_VRH
    # Filter only relevant columns
    data = df[['structure', 'G_VRH', 'material_id']]
    
    output_dir = "/data/home/5240019/class_AI4Bat/Case_3/data/elastic_tensor_2015"
    cif_dir = os.path.join(output_dir, "cifs")
    os.makedirs(cif_dir, exist_ok=True)
    
    id_prop_list = []
    
    print(f"Processing {len(data)} entries...")
    for i, row in data.iterrows():
        struct = row['structure']
        shear_modulus = row['G_VRH']
        material_id = row['material_id']
        
        # Save as CIF
        cif_filename = f"{material_id}.cif"
        cif_path = os.path.join(cif_dir, cif_filename)
        CifWriter(struct).write_file(cif_path)
        
        # Add to id_prop.csv list
        id_prop_list.append([material_id, shear_modulus])
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} entries")
            
    # Save id_prop.csv
    id_prop_df = pd.DataFrame(id_prop_list)
    id_prop_df.to_csv(os.path.join(output_dir, "id_prop.csv"), index=False, header=False)
    
    print(f"Done! Data saved to {output_dir}")

if __name__ == "__main__":
    prepare_data()
