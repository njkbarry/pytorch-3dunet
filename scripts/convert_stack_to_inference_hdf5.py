import os
import nrrd
import numpy as np
import h5py
from pathlib import Path
import nrrd
from typing import List
from tqdm import tqdm
import cv2
import warnings

SUFFIXES = [".tiff", ".tif", ".png"]

def convert_stack_to_inference_hdf5(image_stack, inference_key, output_path, chunks=(256,256,256)):
    image_data_list = []
    # Load each image and append it to the list
    for file in sorted(image_stack):
        slice = cv2.imread(str(file))
        if slice is not None:
            if len(slice.shape) == 3:
                warnings.warn("3-channel image read")
                slice = slice[:,:,0]
            if len(slice.shape) > 3:
                raise ValueError(f"Alpha channel image read, expecting layer image: {file}")
            image_data_list.append(slice)

    # Stack the image data along a new dimension to create a 4D volume
    volume = np.stack(image_data_list, axis=0) 

    with h5py.File(output_path, "w") as f:
        dset_shape = (volume.shape[0], volume.shape[1], volume.shape[2])
        f.create_dataset(
            inference_key,
            shape=dset_shape,
            data=volume,
            dtype=np.uint8,
            chunks=chunks,
            compression="gzip",
            compression_opts=3,
        )


if __name__ == "__main__":
    
    # List of mask image files
    root_dir = Path("/home/nickb/data/CT/VW_Steel_Kalibration_Zylinder/Neu_A6_1")
    output_path = None

    if output_path is None:
        output_path = root_dir.parent / (root_dir.stem+".hdf5")

    candidate_files = [root_dir / path for path in os.listdir(root_dir)]
    inference_h5_files = [fp for fp in candidate_files if fp.suffix in SUFFIXES]

    # Convert to NRRD
    convert_stack_to_inference_hdf5(inference_h5_files, "layers", output_path, chunks=(128,128,128))