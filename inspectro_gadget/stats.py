#!/usr/bin/env python3

"""
Resources for handling statistical operations and similar.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, kstest


def compare_regions(subunit_data, receptor_list, n_samples=5000):
    """
    Compare receptor subunit expression between two regions.

    Parameters
    ----------
    subunit_data: dict
        Dictionary with the receptor expression values for the two regions
    receptor_list: dataframe
        Dataframe containing the list of receptor subunits and the receptors they compose. Analysis is grouped by
        receptor type
    n_samples: int
        Number of permutations for calculating confidence intervals

    Returns
    -------
    tuple of (dict, dict, dict, dict)
        subunit_d_vals, subunit_d_cis, subunit_pct_diff, subunit_ks_vals
    """
    region_names = list(subunit_data.keys())
    subunit_d_vals = {}
    subunit_d_cis = {}
    subunit_ks_vals = {}
    subunit_pct_diff = {}
    # Loop through receptor types
    for receptor in np.unique(receptor_list['grouping']):
        # Modify alpha for number of comparisons
        alpha = 0.05 / receptor_list[receptor_list.grouping == receptor].shape[0]
        for subunit in receptor_list[receptor_list.grouping == receptor]['subunit']:
            # Get data and remove NaNs
            region_one = subunit_data[region_names[0]][subunit].values
            region_two = subunit_data[region_names[1]][subunit].values
            if np.isnan(region_one).all() or np.isnan(region_two).all():
                subunit_pct_diff[subunit], subunit_d_vals[subunit], subunit_d_cis[subunit] = [0.0, 0.0, [0.0, 0.0]]
                subunit_ks_vals[subunit] = 0.0
                if np.isnan(region_one).all():
                    print(f'No {subunit} expression values in {region_names[0]}')
                else:
                    print(f'No {subunit} expression values in {region_names[1]}')
            else:
                region_one = region_one[~np.isnan(region_one)]
                region_two = region_two[~np.isnan(region_two)]
                # Compare subunit values
                subunit_pct_diff[subunit], subunit_d_vals[subunit], subunit_d_cis[subunit] = bootstrap_diff(region_one,
                                                                                                            region_two,
                                                                                                            n_samples=n_samples,
                                                                                                            alpha=alpha)
                # Mean centre for KS test
                r1_centered = region_one - np.mean(region_one)
                r2_centered = region_two - np.mean(region_two)
                # Apply KS test
                subunit_ks_vals[subunit] = kstest(r1_centered, r2_centered, alternative='two-sided', mode='auto').statistic
    return subunit_d_vals, subunit_d_cis, subunit_pct_diff, subunit_ks_vals


def ex_in(subunit_data, receptor_list):
    """
    Calculate the excitation/inhibition ratio for a region.

    Divides the sum of AMPA+NMDA expression by the sum of GABAA(alpha, beta, gamma) expression.

    Definition taken from Deco et al, Dynamical consequences of regional heterogeneity in the brain’s transcriptional
    landscape. Science Advances, 7(29):eabf4752, 2021.

    Parameters
    ----------
    subunit_data: dict or dataframe
        Dictionary or dataframe with the expression data per subunit
    receptor_list: dataframe
        Dataframe containing the list of receptor subunits and the receptors they compose. Analysis is grouped by
        receptor type

    Returns
    -------
    ex_in: float
        Excitation inhibition ratio
    """
    gabaa_a = np.nansum(subunit_data[receptor_list.subunit[receptor_list.grouping == 'GABAA_Alpha'].values])
    gabaa_b = np.nansum(subunit_data[receptor_list.subunit[receptor_list.grouping == 'GABAA_Beta'].values])
    gabaa_g = np.nansum(subunit_data[receptor_list.subunit[receptor_list.grouping == 'GABAA_Gamma'].values])
    nmda = np.nansum(subunit_data[receptor_list.subunit[receptor_list.grouping == 'NMDA'].values])
    ampa = np.nansum(subunit_data[receptor_list.subunit[receptor_list.grouping == 'AMPA'].values])
    denom = gabaa_a + gabaa_b + gabaa_g
    if denom == 0:
        return 0.0
    return float((nmda + ampa) / denom)


def region_median(subunit_data, receptor_list):
    """
    Calculate the median gene expression for each subunit for a region.

    Returns a dataframe with subunit name as columns and a single row of median expression values.

    Parameters
    ----------
    subunit_data: array or dataframe
        Array containing the region's expression data
    receptor_list: dataframe
        Dataframe with all subunit names

    Returns
    -------
    Dataframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(subunit_data, axis=0).reshape((1, subunit_data.shape[1]))
    medians[np.isnan(medians)] = 0
    df = pd.DataFrame(columns=receptor_list.subunit.values, data=medians)
    return df


def subject_median(subunit_data, receptor_list):
    """
    Calculate the median gene expression for each subunit per subject.

    Returns a dataframe with subject as rows and subunit name as columns.

    Parameters
    ----------
    subunit_data: dict
        Dictionary containing all subject's expression data
    receptor_list: dataframe
        Dataframe with all subunit names

    Returns
    -------
    Dataframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rows = {subject: np.nanmedian(subunit_data[subject], axis=0) for subject in subunit_data.keys()}
    return pd.DataFrame.from_dict(rows, orient='index', columns=receptor_list.subunit.values)


def calc_cohend(g1, g2):
    """
    Calculate Cohen's d for a comparison of two groups.

    Parameters
    ----------
    g1: array-like
        First group data
    g2: array-like
        Second group data

    Returns
    -------
    float
        Cohen's d effect size
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0
    if n1 < 2 or n2 < 2:
        sd = (np.std(g1) + np.std(g2)) / 2.0
    else:
        s1 = np.var(g1, ddof=1)
        s2 = np.var(g2, ddof=1)
        pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        sd = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if sd == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / sd)


def get_boot_d(g1, g2):
    """
    Calculate Cohen's d for a single pair of resampled data.

    g1: First group data
    g2: Second group data
    """
    idx1 = np.random.randint(0, g1.shape[0], g1.shape[0])
    idx2 = np.random.randint(0, g2.shape[0], g2.shape[0])
    return calc_cohend(g1[idx1], g2[idx2])


def get_jack_d(g1, g2, ii):
    """
    Calculate Cohen's d for a single pair of jackknife data.

    g1: First group data
    g2: Second group data
    ii: iteration number
    """
    j1 = np.delete(g1, ii, axis=0)
    j2 = np.delete(g2, ii, axis=0)
    return calc_cohend(j1, j2)


def cohens_d(g1, g2, n_samples=10000, alpha=0.05):
    """
    Calculate Cohen's d plus adjusted confidence interval for a comparison
    of two groups using BCa (bias-corrected and accelerated) bootstrap.

    Parameters
    ----------
    g1: array-like
        First group data
    g2: array-like
        Second group data
    n_samples: number of permutations
    alpha: confidence interval width

    Returns
    -------
    orig: float
        Observed Cohen's d
    ci: list of [float, float]
        Lower and upper confidence interval bounds
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0, [0.0, 0.0]

    orig = calc_cohend(g1, g2)
    alphas = np.array([alpha / 2.0, 1.0 - alpha / 2.0])

    # Vectorized bootstrap calculation
    idx1 = np.random.randint(0, n1, size=(n_samples, n1))
    idx2 = np.random.randint(0, n2, size=(n_samples, n2))
    boot_g1 = g1[idx1]
    boot_g2 = g2[idx2]
    m1 = np.mean(boot_g1, axis=1)
    m2 = np.mean(boot_g2, axis=1)
    if n1 >= 2 and n2 >= 2:
        v1 = np.var(boot_g1, axis=1, ddof=1)
        v2 = np.var(boot_g2, axis=1, ddof=1)
        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        sd = np.sqrt(np.maximum(pooled_var, 0.0))
    else:
        sd = (np.std(boot_g1, axis=1) + np.std(boot_g2, axis=1)) / 2.0
    sd[sd == 0] = 1e-12
    boot = np.sort((m1 - m2) / sd)

    # Bias correction z0
    prop = np.mean(boot < orig)
    prop = np.clip(prop, 1.0 / (2.0 * n_samples), 1.0 - 1.0 / (2.0 * n_samples))
    z0 = norm.ppf(prop)

    # Jackknife acceleration parameter a
    n_jack = min(n1, n2)
    jstat = np.empty(n_jack)
    for ii in range(n_jack):
        j1 = np.delete(g1, ii, axis=0)
        j2 = np.delete(g2, ii, axis=0)
        jstat[ii] = calc_cohend(j1, j2)

    jmean = np.mean(jstat)
    denom = 6.0 * (np.sum((jmean - jstat) ** 2) ** 1.5)
    if denom > 0:
        a = np.sum((jmean - jstat) ** 3) / denom
    else:
        a = 0.0

    zs = z0 + norm.ppf(alphas)
    denom_a = 1.0 - a * zs
    denom_a[denom_a == 0] = 1e-12
    avals = norm.cdf(z0 + zs / denom_a)
    nvals = np.clip(np.around((n_samples - 1) * avals).astype(int), 0, n_samples - 1)
    return float(orig), [float(boot[nvals[0]]), float(boot[nvals[1]])]


def mad_median(x):
    """
    Remove outliers based upon median absolute deviation. Threshold is set
    by chi squared distribution with two degrees of freedom.

    x: data to remove outliers from
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return x
    mm = (x - med) / (mad / 0.6745)
    filtered = x[np.abs(mm) < 2.24]
    return filtered if len(filtered) > 0 else x


def perm_median(g1, g2):
    """
    Calculate median difference for a single pair of shuffled data.

    g1: First group data
    g2: Second group data
    """
    len1 = len(g1)
    comb_gs = np.hstack((g1, g2))
    np.random.shuffle(comb_gs)
    return np.median(comb_gs[:len1]) - np.median(comb_gs[len1:])


def bootstrap_diff(g1, g2, n_samples=10000, alpha=0.05):
    """
    Run robust comparison of two independent samples.

    Removes outliers based on median absolute deviation. Tests difference in medians.
    Calculates Cohen's d plus confidence interval of this. Returns these plus
    percentage difference between samples.

    g1: First group data
    g2: Second group data
    n_samples: number of permutations
    alpha: significance level

    return: percentage difference, Cohen's d, Cohen's d confidence interval
    """
    D, Dci = cohens_d(g1, g2, alpha=alpha, n_samples=n_samples)
    g1_mad = mad_median(g1)
    g2_mad = mad_median(g2)
    med2 = np.median(g2_mad)
    mDif = np.median(g1_mad) - med2
    if med2 != 0:
        pctDif = (mDif / med2) * 100.0
    else:
        pctDif = 0.0
    return pctDif, D, Dci


def subject_overlap(mask_images):
    """
    Calculate the overlap between subject masks. Returns as % overlap.

    Parameters
    ----------
    mask_images: dict
        Dictionary containing all subject mask images

    Returns
    -------
    Array with percentage overlaps values
    """
    subjects = list(mask_images.keys())
    dims = np.append(mask_images[subjects[0]].shape, len(subjects))
    overlap = np.zeros(dims)
    for ss, subject in enumerate(subjects):
        overlap[:, :, :, ss] = mask_images[subject]
    return (np.sum(overlap, axis=-1) / len(subjects)) * 100.0

