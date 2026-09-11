"""Test helpers and fixtures for InSpectro-Gadget tests."""

import os
import numpy as np
import nibabel as ni
import pandas as pd


STANDARD_AFFINE = np.array([
    [-2.0,  0.0,  0.0,   90.0],
    [ 0.0,  2.0,  0.0, -126.0],
    [ 0.0,  0.0,  2.0,  -72.0],
    [ 0.0,  0.0,  0.0,    1.0],
])


def create_dummy_mask(shape=(91, 109, 91), center=(45, 54, 45), radius=3):
    """Create a 3D binary mask array with a small sphere around center."""
    mask = np.zeros(shape, dtype=np.float32)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    mask[dist <= radius] = 1.0
    return mask


def save_dummy_nifti(filepath, data=None, shape=(91, 109, 91), pixdim=(2.0, 2.0, 2.0)):
    """Save a NIfTI image with specified pixdim and dimensions."""
    if data is None:
        data = create_dummy_mask(shape=shape)
    img = ni.Nifti1Image(data, STANDARD_AFFINE)
    # Ensure header pixdim matches expected
    header = img.header
    header['pixdim'][1:4] = pixdim
    ni.save(img, filepath)
    return filepath


def sample_receptor_list():
    """Return a minimal receptor list DataFrame for unit tests."""
    return pd.DataFrame({
        'subunit': ['GABRA1', 'GABRA2', 'GABRB1', 'GABRG1', 'GRIN1', 'GRIA1'],
        'grouping': ['GABAA_Alpha', 'GABAA_Alpha', 'GABAA_Beta', 'GABAA_Gamma', 'NMDA', 'AMPA'],
        'trans_radar': [True, True, True, True, True, True],
        'mod_radar': [False, False, False, False, False, False],
    })
