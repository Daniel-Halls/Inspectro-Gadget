"""Unit tests for inspectro_gadget.stats module."""

import unittest
import numpy as np
import pandas as pd

from inspectro_gadget import stats
from tests.helpers import sample_receptor_list, create_dummy_mask


class TestStatsCohenD(unittest.TestCase):
    """Tests for Cohen's d effect size calculation and bootstrapping."""

    def test_calc_cohend_known_values(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        d = stats.calc_cohend(g1, g2)
        # m1=3, m2=4, var1=2.5, var2=2.5, pooled_var=2.5, sd=sqrt(2.5)
        expected_d = -1.0 / np.sqrt(2.5)
        self.assertAlmostEqual(d, expected_d, places=6)

    def test_calc_cohend_identical_groups(self):
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = stats.calc_cohend(g, g)
        self.assertAlmostEqual(d, 0.0)

    def test_calc_cohend_zero_variance(self):
        g1 = np.array([3.0, 3.0, 3.0])
        g2 = np.array([3.0, 3.0, 3.0])
        d = stats.calc_cohend(g1, g2)
        self.assertEqual(d, 0.0)

    def test_calc_cohend_small_samples(self):
        # Single element inputs
        d = stats.calc_cohend(np.array([1.0]), np.array([2.0]))
        self.assertEqual(d, 0.0)

    def test_cohens_d_bootstrap_ci(self):
        np.random.seed(42)
        g1 = np.random.normal(loc=1.0, scale=1.0, size=50)
        g2 = np.random.normal(loc=1.5, scale=1.0, size=50)
        orig_d, ci = stats.cohens_d(g1, g2, n_samples=500, alpha=0.05)

        self.assertIsInstance(orig_d, float)
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])
        # The observed effect size should generally lie within the CI
        self.assertLessEqual(ci[0], orig_d)
        self.assertGreaterEqual(ci[1], orig_d)

    def test_cohens_d_empty_input(self):
        orig_d, ci = stats.cohens_d(np.array([]), np.array([1.0]))
        self.assertEqual(orig_d, 0.0)
        self.assertEqual(ci, [0.0, 0.0])


class TestOutlierAndBootstrap(unittest.TestCase):
    """Tests for MAD outlier filtering and bootstrap difference."""

    def test_mad_median_filters_outliers(self):
        # Regular values with one extreme outlier
        x = np.array([1.0, 1.1, 1.05, 0.95, 1.02, 100.0])
        filtered = stats.mad_median(x)
        self.assertNotIn(100.0, filtered)
        self.assertEqual(len(filtered), 5)

    def test_mad_median_zero_mad(self):
        x = np.array([2.0, 2.0, 2.0, 2.0])
        filtered = stats.mad_median(x)
        np.testing.assert_array_equal(filtered, x)

    def test_mad_median_empty(self):
        x = np.array([])
        filtered = stats.mad_median(x)
        self.assertEqual(len(filtered), 0)

    def test_perm_median(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        diff = stats.perm_median(g1, g2)
        self.assertIsInstance(diff, float)

    def test_bootstrap_diff(self):
        np.random.seed(42)
        g1 = np.array([2.0, 2.1, 2.2, 1.9, 2.05])
        g2 = np.array([1.0, 1.1, 1.05, 0.95, 1.02])
        pct_diff, d, ci = stats.bootstrap_diff(g1, g2, n_samples=200)

        self.assertGreater(pct_diff, 0)
        self.assertGreater(d, 0)
        self.assertEqual(len(ci), 2)


class TestReceptorStatistics(unittest.TestCase):
    """Tests for excitation/inhibition ratio and median summaries."""

    def setUp(self):
        self.receptor_list = sample_receptor_list()

    def test_ex_in_ratio(self):
        # E = NMDA(1.0) + AMPA(1.0) = 2.0
        # I = GABAA_Alpha(1.0) + GABAA_Beta(1.0) + GABAA_Gamma(1.0) = 3.0
        sub_data = pd.DataFrame({
            "GABRA1": [0.5, 0.5],
            "GABRA2": [0.5, 0.5],
            "GABRB1": [1.0, 1.0],
            "GABRG1": [1.0, 1.0],
            "GRIN1": [1.0, 1.0],
            "GRIA1": [1.0, 1.0],
        })
        ei = stats.ex_in(sub_data, self.receptor_list)
        # Sums across all voxels in mask:
        # gabaa_a sum = 1.0 + 1.0 = 2.0
        # gabaa_b sum = 1.0 + 1.0 = 2.0
        # gabaa_g sum = 1.0 + 1.0 = 2.0
        # nmda sum = 1.0 + 1.0 = 2.0
        # ampa sum = 1.0 + 1.0 = 2.0
        # (2 + 2) / (2 + 2 + 2) = 4 / 6 = 2/3
        self.assertAlmostEqual(ei, 4.0 / 6.0, places=6)

    def test_ex_in_ratio_zero_denominator(self):
        sub_data = pd.DataFrame({
            "GABRA1": [0.0],
            "GABRA2": [0.0],
            "GABRB1": [0.0],
            "GABRG1": [0.0],
            "GRIN1": [1.0],
            "GRIA1": [1.0],
        })
        ei = stats.ex_in(sub_data, self.receptor_list)
        self.assertEqual(ei, 0.0)

    def test_region_median(self):
        data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ])
        df = stats.region_median(data, self.receptor_list)
        self.assertEqual(df.shape, (1, 6))
        self.assertAlmostEqual(df.loc[0, "GABRA1"], 2.0)
        self.assertAlmostEqual(df.loc[0, "GRIA1"], 7.0)

    def test_subject_median(self):
        sub_data = {
            "sub-01": pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], columns=self.receptor_list.subunit),
            "sub-02": pd.DataFrame([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], columns=self.receptor_list.subunit),
        }
        df = stats.subject_median(sub_data, self.receptor_list)
        self.assertEqual(df.shape, (2, 6))
        self.assertEqual(list(df.index), ["sub-01", "sub-02"])
        self.assertAlmostEqual(df.loc["sub-01", "GABRA1"], 1.0)
        self.assertAlmostEqual(df.loc["sub-02", "GABRA1"], 3.0)

    def test_compare_regions(self):
        sub_data = {
            "R1": pd.DataFrame({
                "GABRA1": [0.2, 0.3, 0.4], "GABRA2": [0.1, 0.2, 0.3],
                "GABRB1": [0.4, 0.5, 0.6], "GABRG1": [0.3, 0.4, 0.5],
                "GRIN1": [0.5, 0.6, 0.7], "GRIA1": [0.6, 0.7, 0.8]
            }),
            "R2": pd.DataFrame({
                "GABRA1": [0.3, 0.4, 0.5], "GABRA2": [0.2, 0.3, 0.4],
                "GABRB1": [0.5, 0.6, 0.7], "GABRG1": [0.4, 0.5, 0.6],
                "GRIN1": [0.6, 0.7, 0.8], "GRIA1": [0.7, 0.8, 0.9]
            })
        }
        d_vals, d_cis, pct_diff, ks_vals = stats.compare_regions(sub_data, self.receptor_list, n_samples=100)
        for subunit in self.receptor_list.subunit:
            self.assertIn(subunit, d_vals)
            self.assertIn(subunit, d_cis)
            self.assertIn(subunit, pct_diff)
            self.assertIn(subunit, ks_vals)

    def test_subject_overlap(self):
        mask1 = create_dummy_mask(center=(45, 54, 45), radius=3)
        mask2 = create_dummy_mask(center=(45, 54, 45), radius=3)
        overlap = stats.subject_overlap({"s1": mask1, "s2": mask2})
        self.assertEqual(overlap.shape, (91, 109, 91))
        # Complete overlap region should equal 100%
        self.assertEqual(np.max(overlap), 100.0)


if __name__ == "__main__":
    unittest.main()
