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
CLASS_INDEX_DICT = {
    'pore': 2,
    'part': 1,
    'background': 0
}

def convert_write_nrrd(h5_file_path, inference_key, output_directory, select_channels:List=None):
    with h5py.File(h5_file_path, 'r') as f:
        assert inference_key in f.keys(), "Inference key not found in hdf5 file"
        output_file_path = output_directory / (h5_file_path.stem + ".nrrd")
        raw_outputs = np.transpose(f[inference_key], [3,2,1,0]) # HWDC, in memory
        segmentations = np.argmax(raw_outputs, 3)
        if select_channels is not None:
            segmentations = segmentations[:,:,:,select_channels]
        # Save the volume to an NRRD file
        nrrd.write(str(output_file_path), segmentations, index_order='F') # XYZ

def integrate_manual_labels(original_labels, updated_labels, debug=False, n=1500):
    """Need to account for missing part labels where pore labels have been removed from initial inference
    """
    original_labels_array, original_labels_dict = original_labels
    updated_labels_array, updated_labels_dict = updated_labels

    original_pore_labels = np.where(original_labels_array==CLASS_INDEX_DICT['pore'], 1, 0)
    updated_pore_labels = np.where(updated_labels_array==CLASS_INDEX_DICT['pore'], 1, 0)
    original_part_labels = np.where(original_labels_array==CLASS_INDEX_DICT['part'], 1, 0)
    updated_part_labels = np.where(updated_labels_array==CLASS_INDEX_DICT['part'], 1, 0)

    integrated_part_labels_array = np.bitwise_or(original_pore_labels, updated_part_labels)

    integrated_labels_array  = np.zeros_like(original_labels_array)
    integrated_labels_array = np.where(integrated_part_labels_array>0, CLASS_INDEX_DICT['part'], integrated_labels_array)
    integrated_labels_array = np.where(updated_pore_labels>0, CLASS_INDEX_DICT['pore'], integrated_labels_array)

    if debug:
        integrated_slice = (integrated_labels_array[:,:,n]*100).astype(np.uint8)
        original_pore_slice = (original_pore_labels[:,:,n]*100).astype(np.uint8)
        updated_part_slice = (updated_part_labels[:,:,n]*100).astype(np.uint8)
        integrated_parts_slice = (integrated_part_labels_array[:,:,n]*100).astype(np.uint8)


    return integrated_labels_array



if __name__ == "__main__":
    
    # List of mask image files
    original_labels_path = Path("/home/nickb/data/CT/xct_pores_dataset/B7021_inference/model_1_inference/model-1_predictions.nrrd")
    updated_labels_path = Path("/home/nickb/mlstore/04_CT_Data/3d_ct_dataset/B7021/updated_labels/model-1_predictions.nrrd")
    # inputs_path = Path("/home/nickb/data/CT/xct_pores_dataset/B7021_inference/model_inputs/model-1.hdf5")
    inputs_path = Path("/home/nickb/mlstore/04_CT_Data/3d_ct_dataset/B7021/initialisaiton/model_inputs/model-1.hdf5")

    output_file_path = Path("/home/nickb/data/CT/xct_pores_dataset/aa_pore_datasets/hdf5_store/B7021_model_1_start_00001_end_02200.hdf5")

    original_labels = nrrd.read(str(original_labels_path), index_order='F') # XYZ
    updated_labels = nrrd.read(str(updated_labels_path), index_order='F') # XYZ

    integrated_labels = integrate_manual_labels(original_labels, updated_labels)   # XYZ
    integrated_labels = np.swapaxes(integrated_labels, 0, 2)   # ZYX

    if inputs_path.suffix in [".tiff", ".tif"]:
        ct_volume = tif.imread(str(inputs_path))   # ZYX 
    elif inputs_path.suffix in [".h5", ".hdf5"]:
        with h5py.File(inputs_path, 'r') as f:
            ct_volume = np.array(f["layers"])

    assert (ct_volume.shape == integrated_labels.shape)
    # assert ct_volume.dtype == np.uint16

    chunks = (128, 128, 128)

    with h5py.File(output_file_path, "w") as f:
        dset_shape = (ct_volume.shape[0], ct_volume.shape[1], ct_volume.shape[2])
        f.create_dataset(
            'layers',
            shape=dset_shape,
            data=ct_volume,
            dtype=np.uint16,
            chunks=chunks,
            compression="gzip",
            compression_opts=3,
        )
        f.create_dataset(
            'labels',
            shape=dset_shape,
            data=integrated_labels,
            dtype=np.uint8,
            chunks=chunks,
            compression="gzip",
            compression_opts=3,
        )
    
    print("wait!")



