import os
import nrrd
import numpy as np
import h5py
from pathlib import Path
import nrrd
from typing import List
from tqdm import tqdm
import tifffile as tif

SUFFIXES = [".h5", ".hdf5"]

def convert_h5_tiff(h5_file_path, inference_key, output_directory, select_channels:List=None, threshold:float=0.5):
    with h5py.File(h5_file_path, 'r') as f:
        assert inference_key in f.keys(), "Inference key not found in hdf5 file"
        output_file_path = output_directory / (h5_file_path.stem + ".tif")
        raw_outputs = f[inference_key] # DHW, in memory
        tif.imwrite(output_file_path, raw_outputs)

if __name__ == "__main__":
    
    # List of mask image files
    root_dir = Path("/home/nickb/data/CT/VW_Steel_Kalibration_Zylinder/")
    
    candidate_files = [root_dir / path for path in os.listdir(root_dir)]
    inference_h5_files = [fp for fp in candidate_files if fp.suffix in SUFFIXES]

    # Convert to NRRD
    for fp in tqdm(inference_h5_files, total=len(inference_h5_files), desc=f"Converting .h5 files to .tff files in directory: {root_dir.stem}"):
        convert_h5_tiff(fp, 'layers', root_dir)