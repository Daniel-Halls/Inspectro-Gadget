#!/usr/bin/env python3

"""
Resources for handling data input, output, and organisation.
"""

import inspect
import os
import nibabel as ni
import numpy as np
import pandas as pd
from copy import deepcopy


def is_valid(var, var_type, list_type=None):
    """
    Check that the var is of a certain type.
    If type is list and list_type is specified, checks that the list contains list_type.

    Parameters
    ----------
    var: any type
        Variable to be checked.
    var_type: type
        Type the variable is assumed to be.
    list_type: type
        Like var_type, but applies to list elements.
    Returns
    -------
    var: any type
        Variable to be checked (same as input).
    Raises
    ------
    TypeError
        If var is not of var_type or elements not of list_type
    """
    if not isinstance(var, var_type):
        raise TypeError(f'The given variable is of type {type(var)}, expected {var_type}')

    if var_type is list and list_type is not None:
        for element in var:
            _ = is_valid(element, list_type)

    return var


def test_nifti_ext(fname):
    """
    Ensure that the input filename has the expected extension for a nifti file.

    Parameters
    ----------
    fname: string
        The filename

    """
    if not isinstance(fname, str):
        raise TypeError(f'Expected filename to be a string, got {type(fname)}')
    # Check it's a valid filename
    if fname.endswith('.nii') or fname.endswith('.nii.gz'):
        return
    else:
        raise ValueError(f'The input file {fname} does not appear to be for a nifti image (.nii or .nii.gz)')


def load_nifti(fname):
    """
    Load data from an input nifti image.

    Parameters
    ----------
    fname: string
        Input filename

    Returns
    -------
    Image data as a numpy array

    """
    # Check it's a valid filename
    test_nifti_ext(fname)

    # Load image
    img = ni.load(fname)

    # Ensure image is correct size
    if not img.shape == (91, 109, 91):
        raise ValueError(f'Input file {fname} does not have the correct dimensions: expected (91, 109, 91), got {img.shape}')
    if not tuple(img.header['pixdim'][1:4]) == (2, 2, 2):
        raise ValueError(f'Input file {fname} does not have 2mm isotropic voxels: pixdim is {tuple(img.header["pixdim"][1:4])}')

    return img.get_fdata()


def save_nifti(img, affine, out_dir):
    """
    Save an array as a nifti image. Specifically for saving subject overlap image.

    Parameters
    ----------
    img: array
        Subject overlap values
    affine: array
        MNI152 affine matrix
    out_dir: str
        Directory to output to

    Returns
    -------

    """
    img = ni.Nifti1Image(img, affine)
    img.to_filename(os.path.join(out_dir, 'subject-overlap.nii.gz'))


def save_exin(exin_data, labels, out_dir):
    """
    Save each subject's estimated excitation/inhibition ratio as a CSV file.

    Parameters
    ----------
    exin_data: dict
        Each subject's ex/in data
    labels: list
        List of subject labels
    out_dir: str
        Directory to save file to

    Returns
    -------

    """
    exin = np.zeros(len(exin_data))
    for ss, subject in enumerate(labels):
        exin[ss] = exin_data[subject]
    df = pd.DataFrame(data=exin, index=labels, columns=['Ex_In'])
    df.to_csv(os.path.join(out_dir, 'subject-excitation-inhibition-ratios.csv'))


def extract_mrna(subunit_path, region_mask):
    """
    Extract the mRNA expression values within a mask region and normalise relative to cortical expression through
    robust sigmoid method.

    Parameters
    ----------
    subunit_path: string
        Path to image file containing mRNA values
    region_mask: array
        Numpy array containing voxel mask

    Returns
    -------
    region_norm:
        Normalised expression values for all voxels within mask

    """
    # Load image file
    mrna = ni.load(subunit_path).get_fdata()

    # Remove any weird values
    mrna[mrna < 0] = 0
    mrna[mrna > 15] = 0
    mrna[np.isnan(mrna)] = 0

    # Get all non-zero values
    mrna_removed = np.array(mrna[(mrna != 0)])

    # interquartile range
    if len(mrna_removed) == 0:
        return np.zeros(np.sum(region_mask == 1))
    median = np.median(mrna_removed)
    Q1 = np.percentile(mrna_removed, 25)
    Q3 = np.percentile(mrna_removed, 75)
    IQRx = (Q3 - Q1) / 1.35
    if IQRx == 0:
        IQRx = 1.0

    # robust sigmoid
    image_norm = 1 / (1 + np.exp(-(mrna - median) / IQRx))
    region_norm = image_norm[region_mask == 1]

    return region_norm


def get_receptor_data(receptors, mask, data_dir):
    """
    Extract mRNA data for each gene from a given mask.

    mRNA expression is normalised to cortical values.

    Parameters
    ----------
    receptors: list
        List of gene names (corresponding to the names used in the image file names)
    mask: array
        Binary mask indicating the MRS voxel
    data_dir: string
        Directory where module data is located.

    Returns
    -------
    Numpy array with a column for each gene. Each row is a voxel.

    """
    mask_voxels = (mask == 1)
    n_voxels = int(np.sum(mask_voxels))
    if n_voxels == 0:
        raise ValueError('The provided mask contains no voxels with value 1.')

    out = np.zeros((n_voxels, len(receptors)))
    for rr, receptor in enumerate(receptors):
        out[:, rr] = extract_mrna(os.path.join(data_dir, 'mRNA_images', f'{receptor}_mirr_mRNA.nii'), mask)
    out_df = pd.DataFrame(data=out, columns=receptors)
    # Set voxels where there is no expression data to "NaN"
    out_df = out_df.mask(out_df < 0.1, np.nan)
    return out_df


class GadgetData:
    """
    Attributes
    ----------
        mask_fnames: list
            File names for voxel masks
        labels: list
            Labels for the masks
        multi_subject: bool
            Whether the masks are from multiple subjects or not
        no_subjects: int
            How many subjects are included (defaults to 1 if it's not a multi-subject analysis)
        receptor_list: dataframe
            Dataframe containing the list of genes to be analysed and the receptor type they are related to.
        img_affine: array
            Affine matrix for the MNI template
        bground_image: array
            Array containing the background image against which masks are to be plotted.
        mask_images: dict
            Dictionary containing arrays with the region masks. Keys are region labels.
        receptor_data: dict
            Dictionary containing the normalised mRNA expression values. Keys are region labels. Arrays within have a
            row per voxel and column per gene. Multi-subject analyses contain separate arrays per subject in a list.
        ex_in_ratio: dictionary
            Excitation/inhibition ratio for each region (and participant where relevant)

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, mask_fnames, labels, data_dir, multi_region=False, multi_subject=False, no_subjects=1):
        """
        Initialise data object.
        """
        #  Inititate user-defined parameters
        self.mask_fnames = deepcopy(mask_fnames)
        self.labels = deepcopy(labels)
        self.no_subjects = deepcopy(no_subjects)
        self.multi_subject = deepcopy(multi_subject)
        self.multi_region = deepcopy(multi_region)

        # Load built-in data
        self.receptor_list = pd.read_csv(os.path.join(data_dir, 'GroupedReceptors.tsv'), delimiter='\t', header=0)
        tmp_img = ni.load(os.path.join(data_dir, 'MNI152_T1_2mm.nii.gz'))
        self.img_affine = tmp_img.affine
        self.bground_image = tmp_img.get_fdata()

        # Load mask(s) and receptor data
        self.mask_images = {}
        self.receptor_data = {}
        if self.multi_subject:
            for ss in range(no_subjects):
                print(f'Loading data for subject {self.labels[ss]}')
                self.mask_images[self.labels[ss]] = load_nifti(self.mask_fnames[ss])
                self.receptor_data[labels[ss]] = get_receptor_data(self.receptor_list.iloc[:, 0].values,
                                                                       self.mask_images[labels[ss]], data_dir)
        elif self.multi_region:
            for ll, label in enumerate(labels):
                print(f'Loading data for region {label}')
                self.mask_images[label] = load_nifti(self.mask_fnames[ll][0])
                self.receptor_data[label] = get_receptor_data(self.receptor_list.iloc[:, 0].values,
                                                              self.mask_images[label], data_dir)
        else:
            print(f'Loading data for region {labels[0]}')
            self.mask_images[labels[0]] = load_nifti(self.mask_fnames[0])
            self.receptor_data[labels[0]] = get_receptor_data(self.receptor_list.iloc[:, 0].values,
                                                          self.mask_images[labels[0]], data_dir)

        # Initiate common stats variables
        self.ex_in_ratio = {}
        self.receptor_median = {}
        self.overlap_image = {}
