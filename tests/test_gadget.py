"""End-to-end integration and pipeline tests for inspectro_gadget.gadget module."""

import os
import tempfile
import unittest
import numpy as np

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
        # Verify exported overlap image and CSV table
        overlap_nii = os.path.join(out.out_dir, "subject-overlap.nii.gz")
        csv_file = os.path.join(out.out_dir, "subject-excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(overlap_nii))
        self.assertTrue(os.path.isfile(csv_file))


if __name__ == "__main__":
    unittest.main()
