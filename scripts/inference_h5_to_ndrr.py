import os
import nrrd
import numpy as np
import h5py
from pathlib import Path
import nrrd
from typing import List
from tqdm import tqdm

SUFFIXES = [".h5", ".hdf5"]

def convert_write_nrrd(h5_file_path, inference_key, output_directory, select_channels:List=None):
    with h5py.File(h5_file_path, 'r') as f:
        assert inference_key in f.keys(), "Inference key not found in hdf5 file"
        output_file_path = output_directory / (h5_file_path.stem + ".nrrd")
        if inference_key == 'predictions':       
            raw_outputs = np.transpose(f[inference_key], [3,2,1,0]) # HWDC, in memory
            segmentations = np.argmax(raw_outputs, 3)
        elif inference_key == 'labels':
            segmentations = np.transpose(f[inference_key], [2,1,0])
        if select_channels is not None:
            segmentations = segmentations[:,:,:,select_channels]
        # Save the volume to an NRRD file
        nrrd.write(str(output_file_path), segmentations, index_order='F') # XYZ

if __name__ == "__main__":
    
    # List of mask image files
    root_dir = Path("/home/nickb/tmp/mesh_testing")
    # root_dir = Path("/home/nickb/data/CT/xct_pores_dataset/B7021_inference/dataset")

    candidate_files = [root_dir / path for path in os.listdir(root_dir)]
    inference_h5_files = [fp for fp in candidate_files if fp.suffix in SUFFIXES]

    # Convert to NRRD
    for fp in tqdm(inference_h5_files, total=len(inference_h5_files), desc=f"Converting .h5 inference files to .nrrd segmentation files in directory: {root_dir.stem}"):
        # convert_write_nrrd(fp, 'predictions', root_dir)
        convert_write_nrrd(fp, 'predictions', root_dir)
