#!/usr/bin/env python3

"""
Tool to show the receptor expressions within an MRS region

MRS regions masks must be in MNI152 space (2 mm).

Arguments need to be given for region mask names when using command line
GABA and Glutamate regions must ine in a .tsv file "receptors.tsv"

"""

import os
import inspect
import time
from inspectro_gadget import plotting, stats, io
from matplotlib.backends.backend_pdf import PdfPages


def gadget(mask_fnames, mask_labels=None, out_root=None, bground_fname=None, multi_violin=True):
    """
    Main function that runs the analysis.

    User provides the location(s) of the masks to be used and the relevant output will be produced depending
    upon whether masks for one region, two regions, or multiple subjects are entered.

    Parameters
    ----------
    mask_fnames: list
        List containing mask filename(s). If one region then a list with a single string. If two regions then a list
        with two filenames, each in their own sub-list. If multiple subjects then a list with multiple filenames.
    mask_labels: list
        List containing the labels for each mask. Structure follows that used for filenames.
    out_root: string
        Directory to which output should be written. If none is given then a new directory will be made in the current
        working directory.
    bground_fname: string
        Filename of alternative background image for mask location plots. Must be in MNI152 2mm space.

    Returns
    -------
    Data object

    """

    print('Starting')

    # Check mask images are entered as a list
    mask_fnames = io.is_valid(mask_fnames, list)
    if len(mask_fnames) == 0:
        raise ValueError('mask_fnames cannot be empty.')

    # Count number of regions
    if sum(isinstance(i, list) for i in mask_fnames) == 0:
        multi_region = False
        no_regions = 1
        no_subjects = 1
    else:
        multi_region = True
        no_regions = sum(isinstance(i, list) for i in mask_fnames)
        no_subjects = 1

    # Test if there are multiple subjects
    multi_sub = False
    if not multi_region:
        if len(mask_fnames) > 1:
            multi_sub = True
            no_subjects = len(mask_fnames)

    # Create mask labels if not given
    if not mask_labels:
        mask_labels = []
        if multi_sub:
            for sub_no in range(1, no_subjects + 1):
                mask_labels.append(f'Subject {sub_no}')
        else:
            for mask_no in range(1, no_regions + 1):
                mask_labels.append(f'Region {mask_no}')
    else:
        mask_labels = io.is_valid(mask_labels, list)
        expected_len = no_subjects if multi_sub else no_regions
        if len(mask_labels) != expected_len:
            raise ValueError(f'Expected {expected_len} mask labels, but got {len(mask_labels)}')

    # Create output directory
    if out_root:
        if not os.path.isdir(out_root):
            raise NotADirectoryError(f'The directory to create the output folder in does not exist: {out_root}')
        out_dir = os.path.join(out_root, f'gadget-out_{time.strftime("%Y%m%d-%H%M%S")}')
    else:
        out_dir = os.path.join(os.getcwd(), f'gadget-out_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    # Set path where package data is stored
    data_dir = os.path.join(os.path.dirname(inspect.getfile(io)), 'data')

    # Collect all required data
    data = io.GadgetData(mask_fnames, mask_labels, data_dir, multi_region=multi_region, multi_subject=multi_sub,
                         no_subjects=no_subjects)
    data.out_dir = out_dir

    # Replace background image if the user provides one
    if bground_fname:
        data.bground_image = io.load_nifti(bground_fname)

    # Calculate statistics
    print('Estimating gene expression statistics')
    if data.multi_subject:
        data.receptor_median = stats.subject_median(data.receptor_data, data.receptor_list)
        # Slightly odd way of storing the overlap image so that it can be used directly with the "plot_masks" function.
        data.overlap_image['Subject overlap'] = stats.subject_overlap(data.mask_images)
        for subject in data.labels:
            data.ex_in_ratio[subject] = stats.ex_in(data.receptor_data[subject], data.receptor_list)
    else:
        for region in data.labels:
            data.receptor_median[region] = stats.region_median(data.receptor_data[region], data.receptor_list)
            data.ex_in_ratio[region] = stats.ex_in(data.receptor_data[region], data.receptor_list)
        if data.multi_region:
            data.subunit_d_vals, data.subunit_d_cis, data.subunit_pct_diff, data.subunit_ks_vals = stats.compare_regions(data.receptor_data,
                                                                                                                         data.receptor_list)
    # Create output PDF
    print('Plotting results')
    with PdfPages(os.path.join(out_dir, 'gadget-output.pdf')) as pdf:
        # Mask images
        if data.multi_subject:
            pdf = plotting.plot_overlap(data.overlap_image['Subject overlap'], ['Subject overlap'], data.bground_image, pdf)
            pdf = plotting.multisub_exin(data.ex_in_ratio, data.labels, pdf)
        else:
            pdf = plotting.plot_masks(data.mask_images, data.labels, data.bground_image, pdf, data.ex_in_ratio)

        # Radar plots
        if data.multi_subject:
            pdf = plotting.multisub_radar(data.receptor_median, data.receptor_list, pdf)
        else:
            pdf = plotting.region_radar(data.receptor_median, data.labels, data.receptor_list, pdf)

        # Violin plots
        if data.multi_subject:
            if multi_violin:
                pdf = plotting.multisub_violin(data.receptor_data, data.receptor_list, pdf, data.receptor_median)
        else:
            if data.multi_region:
                pdf = plotting.two_region_violins(data.receptor_data, data.receptor_list, pdf, data.subunit_pct_diff,
                                                  data.subunit_d_vals, data.subunit_d_cis, data.subunit_ks_vals)
            else:
                pdf = plotting.single_region_violins(data.receptor_data[data.labels[0]], data.receptor_list, pdf)

    # Save any other relevant files
    if data.multi_subject:
        io.save_nifti(data.overlap_image['Subject overlap'], data.img_affine, out_dir)
        io.save_exin(data.ex_in_ratio, data.labels, out_dir)
    print('Finished')
    return data
