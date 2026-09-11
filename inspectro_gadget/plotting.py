#!/usr/bin/env python3

"""
Resources for plotting MRS voxel locations and the various mRNA value plots.

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors
from matplotlib import table
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import center_of_mass

# Colours to use for plots
colours = ["#528fad", "#f7aa58", "#aadce0"]

def plot_masks(mask_imgs, labels, bground, pdf, ex_in=None):
    """
    Show the location of the MRS voxels on a background image.

    Parameters
    ----------
    mask_imgs: list
        List containing the MRS voxel mask images
    labels: list
        List contatining the region labels
    bground: array
        Numpy array with the background image masks should be superimposed upon
    pdf: PDFPages object
        PDF object to add figures to
    ex_in: dict
        Estimated excitation/inhibition ratio for each region. Only used for one or two region plots,
        not multi-subject.

    Returns
    -------
    Figure object

    """
    #colours = ["#0086a8", "#a00e00"]
    no_ax = len(labels)
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(no_ax, 3, width_ratios=[1, 1, 0.835])
    if no_ax == 1:
        mask_data = mask_imgs[labels[0]]
        if np.sum(mask_data != 0) == 0:
            raise ValueError(f"Mask for '{labels[0]}' contains no non-zero voxels to plot.")
        cmass = np.round(center_of_mass(mask_data), 0).astype(int)
        # Sagittal
        mask_roi = np.ma.masked_where(mask_data[cmass[0], :, :] == 0, mask_data[cmass[0], :, :])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(bground[cmass[0], :, :].T, cmap='Greys_r', origin='lower', interpolation='none')
        ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[0]]))
        ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
        ax.tick_params(left=False, bottom=False)
        # Coronal
        mask_roi = np.ma.masked_where(mask_data[:, cmass[1], :] == 0, mask_data[:, cmass[1], :])
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(bground[:, cmass[1], :].T, cmap='Greys_r', origin='lower', interpolation='none')
        ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[0]]))
        ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
        ax.tick_params(left=False, bottom=False)
        if ex_in:
            ax.set_title(f'{labels[0]}\nEstimated excitation/inhibition ratio = {ex_in[labels[0]]:0.2f}')
        else:
            ax.set_title(labels[0])
        # Axial
        mask_roi = np.ma.masked_where(mask_data[:, :, cmass[2]] == 0, mask_data[:, :, cmass[2]])
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(bground[:, :, cmass[2]].T, cmap='Greys_r', origin='lower', interpolation='none')
        ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[0]]))
        ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
        ax.tick_params(left=False, bottom=False)
        pdf.savefig(fig)
        plt.close(fig)
    else:
        for aa in range(no_ax):
            mask_data = mask_imgs[labels[aa]]
            if np.sum(mask_data != 0) == 0:
                raise ValueError(f"Mask for '{labels[aa]}' contains no non-zero voxels to plot.")
            cmass = np.round(center_of_mass(mask_data), 0).astype(int)
            # Sagittal
            mask_roi = np.ma.masked_where(mask_data[cmass[0], :, :] == 0, mask_data[cmass[0], :, :])
            ax = fig.add_subplot(gs[aa, 0])
            ax.imshow(bground[cmass[0], :, :].T, cmap='Greys_r', origin='lower', interpolation='none')
            ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[aa]]))
            ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
            ax.tick_params(left=False, bottom=False)
            # Coronal
            mask_roi = np.ma.masked_where(mask_data[:, cmass[1], :] == 0, mask_data[:, cmass[1], :])
            ax = fig.add_subplot(gs[aa, 1])
            ax.imshow(bground[:, cmass[1], :].T, cmap='Greys_r', origin='lower', interpolation='none')
            ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[aa]]))
            ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
            ax.tick_params(left=False, bottom=False)
            if ex_in:
                ax.set_title(f'{labels[aa]}\nEstimated excitation/inhibition ratio = {ex_in[labels[aa]]:0.2f}')
            else:
                ax.set_title(labels[aa])
            # Axial
            mask_roi = np.ma.masked_where(mask_data[:, :, cmass[2]] == 0, mask_data[:, :, cmass[2]])
            ax = fig.add_subplot(gs[aa, 2])
            ax.imshow(bground[:, :, cmass[2]].T, cmap='Greys_r', origin='lower', interpolation='none')
            ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap=colors.ListedColormap([colours[aa]]))
            ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
            ax.tick_params(left=False, bottom=False)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf


def plot_overlap(overlap, labels, bground, pdf):
    """
    Show the overlap between subject MRS regions on a background image.

    Parameters
    ----------
    overlap: array
        Numpy array containing overlap image
    labels: list
        List containing region name
    bground: array
        Numpy array containing background image
    pdf: PDFPages object
        PDF object to add figures to

    Returns
    -------
    Figure object

    """
    if np.sum(overlap != 0) == 0:
        raise ValueError("Overlap image contains no non-zero voxels to plot.")
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 3)
    cmass = np.round(center_of_mass(overlap), 0).astype(int)
    # Sagittal
    mask_roi = np.ma.masked_where(overlap[cmass[0], :, :] == 0, overlap[cmass[0], :, :])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(bground[cmass[0], :, :].T, cmap='Greys_r', origin='lower', interpolation='none')
    ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap='plasma')
    ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False, bottom=False)
    # Coronal
    mask_roi = np.ma.masked_where(overlap[:, cmass[1], :] == 0, overlap[:, cmass[1], :])
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(bground[:, cmass[1], :].T, cmap='Greys_r', origin='lower', interpolation='none')
    ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap='plasma')
    ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False, bottom=False)
    ax.set_title(labels[0])
    # Axial
    mask_roi = np.ma.masked_where(overlap[:, :, cmass[2]] == 0, overlap[:, :, cmass[2]])
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(bground[:, :, cmass[2]].T, cmap='Greys_r', origin='lower', interpolation='none')
    ax.imshow(mask_roi.T, origin='lower', interpolation='none', alpha=0.8, cmap='plasma')
    ax.set(yticklabels=[], xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False, bottom=False)
    pdf.savefig(fig)
    plt.close(fig)
    return pdf


def make_single_violin(ax, receptors, group):
    """
    Create a subplot with violin plots showing receptor expression for a group of subunits.

    Parameters
    ----------
    ax: matplotlib axis
        The axis to add the plot to
    receptors: dataframe
        Dataframe with receptor expression. Column names are subunit names. Includes only subunits for the receptor to
        be plotted.
    group: string
        The subunit grouping name for use in the title

    Returns
    -------
    ax: matplotlib axis
        The axis with the plots added
    """
    sb.violinplot(data=receptors, inner="box", ax=ax, linewidth=0.1, saturation=0.6, color=colours[0])
    ax.set_ylim(0, 1.05)
    ax.set_title(group, fontsize=6)
    ax.tick_params(axis='x', which='major', labelsize=6, rotation=45)
    ax.tick_params(axis='y', which='major', labelsize=6)
    return ax


def single_region_violins(subunit_data, receptor_list, pdf):
    """
    Create violin plots for groups of receptors.

    Parameters
    ----------
    subunit_data: dataframe
        Dataframe with receptor expression. Column names are subunit names
    receptor_list: dataframe
        Dataframe containing the list of receptor subunits and the receptors they compose. Plots are grouped by
        receptor type
    pdf: PDFPages object
        PDF object to add figures to

    Returns
    -------
    PDFPages object

    """
    # First PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['GABAA_Alpha', 'GABAA_Beta', 'GABAA_Gamma']):
        axs[0, rr] = make_single_violin(axs[0, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    for rr, receptor in enumerate(['GABAB', 'NMDA', 'AMPA']):
        axs[1, rr] = make_single_violin(axs[1, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    pdf.savefig(fig)
    plt.close(fig)
    # Second PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['mGlu(I)', 'mGlu(II)', 'mGlu(III)']):
        axs[0, rr] = make_single_violin(axs[0, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    for rr, receptor in enumerate(['Kainate', 'Dopamine', '5-HT1']):
        axs[1, rr] = make_single_violin(axs[1, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    pdf.savefig(fig)
    plt.close(fig)
    # Third PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['5-HT2', '5-HT3+', 'NA_alpha1']):
        axs[0, rr] = make_single_violin(axs[0, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    for rr, receptor in enumerate(['NA_alpha2', 'NA_beta', 'ACh_muscarinic']):
        axs[1, rr] = make_single_violin(axs[1, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    pdf.savefig(fig)
    plt.close(fig)
    # Fourth PDF page
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['ACh_nicotinic']):
        axs[0, rr] = make_single_violin(axs[0, rr],
                                        subunit_data[receptor_list.subunit[receptor_list.grouping == receptor].values],
                                        receptor)
    fig.delaxes(axs[0][1])
    fig.delaxes(axs[1][0])
    fig.delaxes(axs[1][1])
    pdf.savefig(fig)
    plt.close(fig)
    return pdf


def make_two_violins(ax, receptors, group, pcts, ds, ds_ci, kss):
    """
    Create a subplot with violin plots comparing receptor expression between two regions for a group of subunits.

    Plots have table beneath them with comparison statistics.

    Parameters
    ----------
    ax: matplotlib axis
        The axis to add the plot to
    receptors: dataframe
        Dataframe with receptor expression. Column names are subunit names. Includes only subunits for the receptor to
        be plotted.
    group: string
        The subunit grouping name for use in the title
    stats:

    Returns
    -------
    ax: matplotlib axis
        The axis with the plots added
    """
    # Table contents
    rows = ['% diff.', "Cohen's d", 'KS']
    cell_text = [[f'{pct:0.2f}' for pct in pcts],
                 [f'{d:0.2f} ({ds_ci[ii][0]:0.2f} {ds_ci[ii][1]:0.2f})' for ii, d in enumerate(ds)],
                 [f'{ks:0.2f}' for ks in kss]]
    columns = np.unique(receptors.subunit)
    # Plot data
    n_hues = len(np.unique(receptors['region'])) if 'region' in receptors else 1
    palette = colours[:max(n_hues, 1)]
    sb.violinplot(data=receptors, x='subunit', y='values', hue='region', inner="box",
                  ax=ax, linewidth=0.1, palette=palette, saturation=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(group, fontsize=6)
    ax.tick_params(axis='y', which='both', labelsize=4, width=0.5)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc=0, prop={'size': 6})
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which='minor', linestyle='-', linewidth='0.01')
    # Add table
    the_table = table.table(ax, cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(2.9)
    the_table.scale(1, 0.7)
    return ax


def two_region_prep(subunit_data, receptor_list, receptor, subunit_pct_diff, subunit_d_vals, subunit_d_cis, subunit_ks_vals):
    # Arrange subunit data into dataframe for plotting
    target_subunits = receptor_list.subunit[receptor_list.grouping == receptor].values
    dfs = []
    for region in list(subunit_data.keys()):
        for subunit in target_subunits:
            vals = subunit_data[region][subunit].dropna().values
            if len(vals) > 0:
                df = pd.DataFrame({'values': vals, 'subunit': subunit, 'region': region})
                dfs.append(df)
    if dfs:
        subunit_exp = pd.concat(dfs, ignore_index=True)
    else:
        subunit_exp = pd.DataFrame(columns=['values', 'subunit', 'region'])
    # Arrange stats
    pcts = [subunit_pct_diff.get(subunit, 0.0) for subunit in target_subunits]
    ds = [subunit_d_vals.get(subunit, 0.0) for subunit in target_subunits]
    ds_ci = [subunit_d_cis.get(subunit, [0.0, 0.0]) for subunit in target_subunits]
    kss = [subunit_ks_vals.get(subunit, 0.0) for subunit in target_subunits]
    return subunit_exp, receptor, pcts, ds, ds_ci, kss


def two_region_violins(subunit_data, receptor_list, pdf, subunit_pct_diff, subunit_d_vals, subunit_d_cis, subunit_ks_vals):
    # First PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['GABAA_Alpha', 'GABAA_Beta', 'GABAA_Gamma']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[0, rr] = make_two_violins(axs[0, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    for rr, receptor in enumerate(['GABAB', 'NMDA', 'AMPA']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[1, rr] = make_two_violins(axs[1, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()
    # Second PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['mGlu(I)', 'mGlu(II)', 'mGlu(III)']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[0, rr] = make_two_violins(axs[0, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    for rr, receptor in enumerate(['Kainate', 'Dopamine', '5-HT1']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[1, rr] = make_two_violins(axs[1, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()
    # Third PDF page
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['5-HT2', '5-HT3+', 'NA_alpha1']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[0, rr] = make_two_violins(axs[0, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    for rr, receptor in enumerate(['NA_alpha2', 'NA_beta', 'ACh_muscarinic']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[1, rr] = make_two_violins(axs[1, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()
    # Fourth PDF page
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.tight_layout(pad=4.0)
    for rr, receptor in enumerate(['ACh_nicotinic']):
        # Prepare data
        subunit_exp, receptor, pcts, ds, ds_ci, kss = two_region_prep(subunit_data, receptor_list, receptor,
                                                                      subunit_pct_diff, subunit_d_vals,
                                                                      subunit_d_cis, subunit_ks_vals)
        # Make plot
        axs[0, rr] = make_two_violins(axs[0, rr], subunit_exp, receptor, pcts, ds, ds_ci, kss)
    fig.delaxes(axs[0][1])
    fig.delaxes(axs[1][0])
    fig.delaxes(axs[1][1])
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close()
    return pdf


def region_radar(receptor_median, labels, receptor_list, pdf):
    """
    Create radar plots showing median gene expresion for (a) GABA and Glu subunits, and (b) neuromodulator receptor
    subunits. Shows two regions on each plot.

    Parameters
    ----------
    receptor_median: dictionary
        Dictionary with dataframes holding subunit receptor medians for each region.
    labels: list
        List with the names of each region.
    receptor_list: list
        List with names of genes
    pdf: PDFPages object
        PDF object to add figures to

    Returns
    -------
    PDFPages object

    """
    #colours = ["#0086a8", "#a00e00"]
    #fig = plt.figure()
    #gs = gridspec.GridSpec(1, 2)
    # One radar for transmitters (GABA+Glu) and one for neuromodulators
    for tt, radar in enumerate(['trans_radar', 'mod_radar']):
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0,", "0.2", "0.4", "0.6", "0.8", "1"], color="black", size=5)
        ax.set_ylim(0, 1)
        for rr, region in enumerate(labels):
            # Get subunit names and values
            df = receptor_median[region].loc[:, receptor_list[radar].values]
            subunits = df.columns.values
            n_subunits = len(subunits)
            values = df.iloc[0, :].values.flatten().tolist()
            values += values[:1]
            # Calculate spacing
            angles = [n / float(n_subunits) * 2 * np.pi for n in range(n_subunits)]
            angles += angles[:1]
            # Add to plot
            ax.set_xticks(angles[:-1], subunits, size=7)
            ax.set_rlabel_position(0)
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=region, c=colours[rr])
            ax.tick_params(axis='x', labelsize=4)
            # Add legend
            ax.legend(loc='upper left', bbox_to_anchor=(-0.6, 1), fontsize=10)
    # Add to pdf
        fig.tight_layout(pad=3.0)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf


def split_subunit_list(subunits, length):
    """
    Split a list of subunits into sublists of set length. Used to get lists of subunits that will fit on a single
    PDF page.

    Parameters
    ----------
    subunits: list
        List of subunits to split
    length: int
        How long the sublists should be

    Returns
    -------

    """
    # Loop to required length
    for i in range(0, len(subunits), length):
        # Yield means the function starts where it finished the last time
        yield subunits[i:i + length]


def multisub_prep(subunit_data, subunit, receptor_median):
    """
    Prepare subunit expression values for a single target subunit for each subject for plotting.

    Removes NaN values and creates list of arrays, one per subject.

    Also calculates the percentage distance from the group median for each subject's median expression value.

    Parameters
    ----------
    subunit_data: dict
        Dictionary with each subject's expression data
    subunit: str
        Subunit name
    receptor_median: dataframe
        Dataframe with median expression values for each subject.

    Returns
    -------
    subunit_exp: list
        List of dataframes with expression for each subject
    receptor_median: array
        Array of median expression values for the target subunit
    median_dist: array
        Distance from group median for each subject

    """
    subunit_exp = []
    for subject in receptor_median.index:
        subunit_exp.append(subunit_data[subject].loc[:, subunit].dropna())
    median_dist = (receptor_median[subunit].values/np.median(receptor_median[subunit].values)*100)-100
    return subunit_exp, receptor_median[subunit], median_dist


def make_multisub_violin(ax, subunit_exp, subunit, medians, median_dist):
    """

    Create a subplot with violin plots for each subject for a subunit.

    Also plots subject medians.

    Parameters
    ----------
    ax: matplotlib axis
        The axis to add the plot to
    subunit_exp: dict
        Dictionary with subunit expression for each subject
    subunit: str
        Subunit name
    medians: dataframe
        Dataframe with subjects' median subunit expression values
    median_dist: list
        Distance from group median for each subject

    Returns
    -------
    ax: matplotlib axis
        The axis with the plots added

    """

    subjects = medians.index.values
    n_subs = len(subjects)
    # Create a colour map
    colour_map = colors.LinearSegmentedColormap.from_list('cmap', ["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7", "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"], N=n_subs)
    # Table contents
    rows = ['% from\nmedian']
    cell_text = [[f'{dist:0.2f}' for dist in median_dist]]
    cell_text[0].append('')  # Blank for median column
    columns = list(medians.index.values)
    columns.append('Median')
    # Plot data
    for ss in range(n_subs):
        parts = ax.violinplot([subunit_exp[ss]], positions=[ss], showextrema=False)
        ax.scatter(n_subs, medians.iloc[ss], s=18, color=colour_map(ss), alpha=0.75)
        for pc in parts['bodies']:
            pc.set_facecolor(colour_map(ss))
            #pc.set_edgecolor('black')
            pc.set_alpha(0.75)
    # Add overall median as dot
    ax.scatter(n_subs, np.median(medians), edgecolor='black', facecolor=None, s=24)
    ax.set_ylim(0, 1.05)
    ax.set_title(subunit, fontsize=6)
    ax.tick_params(axis='y', which='both', labelsize=4, width=0.5)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which='minor', linestyle='-', linewidth='0.01')
    # Add table
    the_table = table.table(ax, cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')
    the_table.set_fontsize(6)
    the_table.scale(1, 1.1)
    return ax


def multisub_violin(subunit_data, receptor_list, pdf, receptor_median):
    """
    Create violin plots for all genes entered in the case of multiple subject input.

    Parameters
    ----------
    subunit_data: dict
        Dictionary with each subject's expression data
    receptor_list: list
        List with the names of each subject
    pdf: PDFPages object
        PDF object to add figures to
    receptor_median: dictionary
        Dictionary with dataframes holding subunit receptor medians for each region.

    Returns
    -------

    """
    # Split the receptor list into sublists of six to fit page
    #subunit_lists = list(split_subunit_list(receptor_list.loc[receptor_list.multisub_violin, 'subunit'].values, 6))
    subunit_lists = list(split_subunit_list(receptor_list.subunit.values, 6))
    # Create plots
    for subunits in subunit_lists:
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(10, 7), linewidth=0.01)
        plt.tick_params(bottom=False, top=False, left=False, right=False)
        fig.text(0.015, 0.5, 'Normalised mRNA Expression Value', va='center', ha='center', rotation='vertical',
                 fontsize=12)
        fig.tight_layout(pad=4.0)
        for rr, subunit in enumerate(subunits[0:3]):
            # Prepare data
            subunit_exp, medians, median_dist = multisub_prep(subunit_data, subunit, receptor_median)
            # Make plot
            axs[0, rr] = make_multisub_violin(axs[0, rr], subunit_exp, subunit, medians, median_dist)
        for rr, subunit in enumerate(subunits[3:]):
            # Prepare data
            subunit_exp, medians, median_dist = multisub_prep(subunit_data, subunit, receptor_median)
            # Make plot
            axs[1, rr] = make_multisub_violin(axs[1, rr], subunit_exp, subunit, medians, median_dist)
        # Remove any empty plots
        if len(subunits) == 3:
            fig.delaxes(axs[1][0])
            fig.delaxes(axs[1][1])
            fig.delaxes(axs[1][2])
        if len(subunits) == 4:
            fig.delaxes(axs[1][1])
            fig.delaxes(axs[1][2])
        if len(subunits) == 5:
            fig.delaxes(axs[1][2])
        fig.tight_layout(pad=3.0)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf


def multisub_exin(exin_data, labels, pdf):
    """
    Plot each subject's estimated E:I ratio, along with the group median

    Parameters
    ----------
    exin_data: dict
        Dictionary with E:I values for each subject
    labels: list
        List of subject names
    pdf: PDFPages object
        PDF object to add figures to

    Returns
    -------

    """
    # Define colourmap
    colour_map = colors.LinearSegmentedColormap.from_list('cmap',
                                                          ["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7",
                                                           "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"],
                                                          N=len(labels))

    exin = np.zeros(len(exin_data))
    xs = np.random.uniform(-0.01, 0.01, len(exin_data))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 5), linewidth=0.01)
    for ss, subject in enumerate(labels):
        ax.scatter(xs[ss], exin_data[subject], label=labels[ss], s=14, color=colour_map(ss))
        exin[ss] = exin_data[subject]
    ax.scatter([0], [np.median(exin)], c='black', s=24)
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0.5, 1)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.tick_params(axis='y', which='both', labelsize=8, width=0.5)
    ax.set_title('Estimated excitation/inhibition ratio', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)
    return pdf


def multisub_radar(receptor_median, receptor_list, pdf):
    """
    Create radar plot showing median gene expresion for neuromodulator receptor subunits. Shows all subjects on one
    plot.

    Parameters
    ----------
    receptor_median: dictionary
        Dictionary with dataframes holding subunit receptor medians for each region.
    receptor_list: list
        List with the names of each region.
    pdf: PDFPages object
        PDF object to add figures to

    Returns
    -------
    PDFPages object

    """
    # GABA+Glu
    df = receptor_median.loc[:, receptor_list.trans_radar.values]
    subjects = df.index.values
    receptors = df.columns.values
    n_receptors = len(receptors)

    # Define colourmap
    colour_map = colors.LinearSegmentedColormap.from_list('cmap',
                                                          ["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7",
                                                           "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"],
                                                          N=len(subjects))

    # Space points around circle
    angles = [n / float(n_receptors) * 2 * np.pi for n in range(n_receptors)]
    angles += angles[:1]  # Connect back to start point

    # Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], receptors, size=4)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0,", "0.2", "0.4", "0.6", "0.8", "1"], color="black", size=5)
    plt.ylim(0, 1)

    for ss, subject in enumerate(subjects):
        values = df.loc[subject, :].values.flatten().tolist()
        values += values[:1]  # Connect back to start point
        ax.plot(angles, values, color=colour_map(ss), linewidth=1, linestyle='solid', label=subject)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(-0.6, 1), fontsize=10)
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close(fig)

    # Neuromodulators
    df = receptor_median.loc[:, receptor_list.mod_radar.values]
    subjects = df.index.values
    receptors = df.columns.values
    n_receptors = len(receptors)

    # Space points around circle
    angles = [n / float(n_receptors) * 2 * np.pi for n in range(n_receptors)]
    angles += angles[:1]  # Connect back to start point

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], receptors, size=4)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0,", "0.2", "0.4", "0.6", "0.8", "1"], color="black", size=5)
    plt.ylim(0, 1)

    for ss, subject in enumerate(subjects):
        values = df.loc[subject, :].values.flatten().tolist()
        values += values[:1]  # Connect back to start point
        ax.plot(angles, values, color=colour_map(ss), linewidth=1, linestyle='solid', label=subject)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(-0.6, 1), fontsize=10)
    fig.tight_layout(pad=3.0)
    pdf.savefig(fig)
    plt.close(fig)
    return pdf
