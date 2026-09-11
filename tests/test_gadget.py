"""End-to-end integration and pipeline tests for inspectro_gadget.gadget module."""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from inspectro_gadget.gadget import gadget
from tests.helpers import create_dummy_mask, save_dummy_nifti


class TestGadgetPipeline(unittest.TestCase):
    """End-to-end tests for the gadget() analysis function across modes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mask1_path = os.path.join(self.temp_dir.name, "mask1.nii.gz")
        self.mask2_path = os.path.join(self.temp_dir.name, "mask2.nii.gz")
        # Save two masks with slight spatial offset
        m1 = create_dummy_mask(center=(45, 54, 45), radius=3)
        m2 = create_dummy_mask(center=(50, 60, 50), radius=3)
        save_dummy_nifti(self.mask1_path, data=m1)
        save_dummy_nifti(self.mask2_path, data=m2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_input_validation(self):
        # Non-list input
        with self.assertRaises(TypeError):
            gadget("not_a_list.nii.gz")
        # Empty list
        with self.assertRaises(ValueError):
            gadget([])
        # Non-existent out_root
        with self.assertRaises(NotADirectoryError):
            gadget([self.mask1_path], out_root="/non/existent/path/here")
        # Mismatched labels
        with self.assertRaises(ValueError):
            gadget([self.mask1_path], mask_labels=["Label 1", "Label 2"])

    def test_single_region_without_labels_no_unbound_error(self):
        # Tests the bug fix where omitting mask_labels previously raised UnboundLocalError
        out = gadget([self.mask1_path], out_root=self.temp_dir.name)
        self.assertEqual(out.labels, ["Region 1"])
        self.assertFalse(out.multi_region)
        self.assertFalse(out.multi_subject)
        self.assertTrue(os.path.isdir(out.out_dir))
        pdf_file = os.path.join(out.out_dir, "gadget-output.pdf")
        self.assertTrue(os.path.isfile(pdf_file))
        self.assertGreater(os.path.getsize(pdf_file), 0)
        self.assertIn("Region 1", out.ex_in_ratio)
        self.assertGreater(out.ex_in_ratio["Region 1"], 0)
        # Verify CSV outputs for single-region
        exin_csv = os.path.join(out.out_dir, "excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(exin_csv))
        df_exin = pd.read_csv(exin_csv)
        self.assertEqual(list(df_exin["label"]), ["Region 1"])
        med_csv = os.path.join(out.out_dir, "receptor-medians.csv")
        self.assertTrue(os.path.isfile(med_csv))
        df_med = pd.read_csv(med_csv)
        self.assertIn("Region 1", df_med.columns)
        self.assertIn("subunit", df_med.columns)
        vox_csv = os.path.join(out.out_dir, "Region_1_voxel_expression.csv")
        self.assertTrue(os.path.isfile(vox_csv))
        df_vox = pd.read_csv(vox_csv)
        self.assertIn("voxel_id", df_vox.columns)
        self.assertGreater(len(df_vox), 0)

    def test_two_region_pipeline(self):
        out = gadget(
            [[self.mask1_path], [self.mask2_path]],
            mask_labels=["ROI_A", "ROI_B"],
            out_root=self.temp_dir.name,
        )
        self.assertTrue(out.multi_region)
        self.assertFalse(out.multi_subject)
        self.assertEqual(out.labels, ["ROI_A", "ROI_B"])
        pdf_file = os.path.join(out.out_dir, "gadget-output.pdf")
        self.assertTrue(os.path.isfile(pdf_file))
        # Verify stats dictionaries populated
        self.assertGreater(len(out.subunit_d_vals), 0)
        self.assertGreater(len(out.subunit_d_cis), 0)
        self.assertGreater(len(out.subunit_pct_diff), 0)
        self.assertGreater(len(out.subunit_ks_vals), 0)
        # Verify CSV outputs for two-region
        med_csv = os.path.join(out.out_dir, "receptor-medians.csv")
        self.assertTrue(os.path.isfile(med_csv))
        df_med = pd.read_csv(med_csv)
        self.assertIn("ROI_A", df_med.columns)
        self.assertIn("ROI_B", df_med.columns)
        comp_csv = os.path.join(out.out_dir, "two-region-comparison-statistics.csv")
        self.assertTrue(os.path.isfile(comp_csv))
        df_comp = pd.read_csv(comp_csv)
        for col in ["subunit", "grouping", "pct_difference", "cohens_d", "cohens_d_ci_lower", "cohens_d_ci_upper", "ks_statistic"]:
            self.assertIn(col, df_comp.columns)
        self.assertTrue(os.path.isfile(os.path.join(out.out_dir, "ROI_A_voxel_expression.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out.out_dir, "ROI_B_voxel_expression.csv")))

    def test_multi_subject_pipeline(self):
        out = gadget(
            [self.mask1_path, self.mask2_path],
            mask_labels=["Sub_01", "Sub_02"],
            out_root=self.temp_dir.name,
            multi_violin=False,
        )
        self.assertFalse(out.multi_region)
        self.assertTrue(out.multi_subject)
        self.assertEqual(out.no_subjects, 2)
        pdf_file = os.path.join(out.out_dir, "gadget-output.pdf")
        self.assertTrue(os.path.isfile(pdf_file))
        # Verify exported overlap image and CSV tables
        overlap_nii = os.path.join(out.out_dir, "subject-overlap.nii.gz")
        csv_file = os.path.join(out.out_dir, "subject-excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(overlap_nii))
        self.assertTrue(os.path.isfile(csv_file))
        exin_csv = os.path.join(out.out_dir, "excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(exin_csv))
        sub_med_csv = os.path.join(out.out_dir, "subject-receptor-medians.csv")
        self.assertTrue(os.path.isfile(sub_med_csv))
        df_sub_med = pd.read_csv(sub_med_csv)
        self.assertEqual(len(df_sub_med), 2)
        grp_med_csv = os.path.join(out.out_dir, "group-receptor-medians.csv")
        self.assertTrue(os.path.isfile(grp_med_csv))
        df_grp_med = pd.read_csv(grp_med_csv)
        self.assertIn("group_median", df_grp_med.columns)
        self.assertTrue(os.path.isfile(os.path.join(out.out_dir, "Sub_01_voxel_expression.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out.out_dir, "Sub_02_voxel_expression.csv")))


if __name__ == "__main__":
    unittest.main()
