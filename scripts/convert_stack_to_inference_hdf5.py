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
        slice = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if slice is not None:
            # if slice.dtype == np.uint16:
            #     if np.max(slice) > 256:   # True 16 bit
            #         # slice = (slice/256).astype(np.uint8)
            #         slice = slice.astype(np.uint8)
            #     else:   # 8-bit stored as 16-bit
            #         slice = slice.astype(np.uint8)
            if len(slice.shape) == 3:
                warnings.warn("3-channel image read")
                assert(np.allclose(slice[:,:,0], slice[:,:,1], slice[:,:,2]))
                slice = slice[:,:,0]
            if len(slice.shape) > 3:
                raise ValueError(f"Alpha channel image read, expecting layer image: {file}")
            image_data_list.append(slice)

    # Stack the image data along a new dimension to create a 4D volume
    volume = np.stack(image_data_list, axis=0)
    if volume.dtype == np.uint16:
        volume = ((volume / np.max(volume)) * 255).astype(np.uint8) 

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
    # root_dir = Path("/home/nickb/data/CT/B7021/ct/layers_cleaned/batch_2_chemically_treated/model-32")

    output_dir = Path("/home/nickb/data/CT/xct_pores_dataset/B7021_inference/model_inputs/batch_1_as_built")

    for i in tqdm(range(80), total=80):
        root_dir = Path(f"/home/nickb/data/CT/B7021/alignment_v6/batch_1_as_built/ct/model-{i}")
        if not os.path.isdir(root_dir):
            continue

        if output_dir is None:
            output_path = root_dir.parent / (root_dir.stem+".hdf5")
        else:
            output_path = output_dir / (root_dir.stem+".hdf5")

        candidate_files = [root_dir / path for path in os.listdir(root_dir)]
        inference_h5_files = [fp for fp in candidate_files if fp.suffix in SUFFIXES]

        # Convert to hdf5
        convert_stack_to_inference_hdf5(inference_h5_files, "layers", output_path, chunks=(90,90,90))