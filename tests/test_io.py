"""Unit tests for inspectro_gadget.io module."""

import os
import inspect
import tempfile
import unittest
import numpy as np
import pandas as pd
import nibabel as ni

from inspectro_gadget import io
from tests.helpers import create_dummy_mask, save_dummy_nifti, STANDARD_AFFINE


class TestIOValidation(unittest.TestCase):
    """Tests for input validation functions."""

    def test_is_valid_type(self):
        self.assertEqual(io.is_valid(42, int), 42)
        self.assertEqual(io.is_valid("test", str), "test")
        with self.assertRaises(TypeError):
            io.is_valid("not an int", int)

    def test_is_valid_list_elements(self):
        self.assertEqual(io.is_valid(["a", "b", "c"], list, str), ["a", "b", "c"])
        with self.assertRaises(TypeError):
            io.is_valid(["a", 123, "c"], list, str)

    def test_test_nifti_ext(self):
        self.assertIsNone(io.test_nifti_ext("mask.nii"))
        self.assertIsNone(io.test_nifti_ext("mask.nii.gz"))
        self.assertIsNone(io.test_nifti_ext("/path/to/my_mask.nii.gz"))
        with self.assertRaises(ValueError):
            io.test_nifti_ext("mask.txt")
        with self.assertRaises(ValueError):
            io.test_nifti_ext("mask.dcm")
        with self.assertRaises(TypeError):
            io.test_nifti_ext(12345)


class TestNiftiIO(unittest.TestCase):
    """Tests for NIfTI file loading and saving."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_nifti_valid(self):
        mask_path = os.path.join(self.temp_dir.name, "valid_mask.nii.gz")
        save_dummy_nifti(mask_path)
        data = io.load_nifti(mask_path)
        self.assertEqual(data.shape, (91, 109, 91))
        self.assertGreater(np.sum(data == 1), 0)

    def test_load_nifti_invalid_shape(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_shape.nii.gz")
        save_dummy_nifti(bad_path, shape=(50, 50, 50))
        with self.assertRaises(ValueError):
            io.load_nifti(bad_path)

    def test_load_nifti_invalid_pixdim(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_pixdim.nii.gz")
        save_dummy_nifti(bad_path, pixdim=(1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            io.load_nifti(bad_path)

    def test_save_nifti(self):
        overlap = create_dummy_mask()
        io.save_nifti(overlap, STANDARD_AFFINE, self.temp_dir.name)
        out_file = os.path.join(self.temp_dir.name, "subject-overlap.nii.gz")
        self.assertTrue(os.path.isfile(out_file))
        loaded = ni.load(out_file)
        self.assertEqual(loaded.shape, (91, 109, 91))

    def test_save_exin(self):
        exin_data = {"sub-01": 1.25, "sub-02": 0.95}
        labels = ["sub-01", "sub-02"]
        io.save_exin(exin_data, labels, self.temp_dir.name)
        out_file = os.path.join(self.temp_dir.name, "excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(out_file))
        df = pd.read_csv(out_file)
        self.assertEqual(list(df["label"]), labels)
        self.assertAlmostEqual(df.loc[0, "excitation_inhibition_ratio"], 1.25)
        self.assertAlmostEqual(df.loc[1, "excitation_inhibition_ratio"], 0.95)
        # Legacy file
        legacy_file = os.path.join(self.temp_dir.name, "subject-excitation-inhibition-ratios.csv")
        self.assertTrue(os.path.isfile(legacy_file))

    def test_save_receptor_medians_single_region(self):
        rec_list = pd.DataFrame({
            "subunit": ["GABRA1", "GRIN1"],
            "grouping": ["GABAA_Alpha", "NMDA"]
        })
        rec_median = {
            "Region 1": pd.DataFrame({"GABRA1": [0.45], "GRIN1": [0.65]})
        }
        io.save_receptor_medians(rec_median, rec_list, ["Region 1"], self.temp_dir.name)
        csv_file = os.path.join(self.temp_dir.name, "receptor-medians.csv")
        self.assertTrue(os.path.isfile(csv_file))
        df = pd.read_csv(csv_file)
        self.assertIn("Region 1", df.columns)
        self.assertEqual(list(df["subunit"]), ["GABRA1", "GRIN1"])
        self.assertAlmostEqual(df.loc[0, "Region 1"], 0.45)

    def test_save_receptor_medians_multi_subject(self):
        rec_list = pd.DataFrame({
            "subunit": ["GABRA1", "GRIN1"],
            "grouping": ["GABAA_Alpha", "NMDA"]
        })
        rec_median = pd.DataFrame({
            "GABRA1": [0.4, 0.5],
            "GRIN1": [0.6, 0.7],
        }, index=["s1", "s2"])
        io.save_receptor_medians(rec_median, rec_list, ["s1", "s2"], self.temp_dir.name, multi_subject=True)
        sub_file = os.path.join(self.temp_dir.name, "subject-receptor-medians.csv")
        grp_file = os.path.join(self.temp_dir.name, "group-receptor-medians.csv")
        self.assertTrue(os.path.isfile(sub_file))
        self.assertTrue(os.path.isfile(grp_file))
        df_grp = pd.read_csv(grp_file)
        self.assertAlmostEqual(df_grp.loc[0, "group_median"], 0.45)

    def test_save_comparison_stats(self):
        rec_list = pd.DataFrame({
            "subunit": ["GABRA1"],
            "grouping": ["GABAA_Alpha"]
        })
        d_vals = {"GABRA1": 0.55}
        d_cis = {"GABRA1": [0.2, 0.9]}
        pct_diff = {"GABRA1": 12.5}
        ks_vals = {"GABRA1": 0.3}
        io.save_comparison_stats(d_vals, d_cis, pct_diff, ks_vals, rec_list, self.temp_dir.name)
        csv_file = os.path.join(self.temp_dir.name, "two-region-comparison-statistics.csv")
        self.assertTrue(os.path.isfile(csv_file))
        df = pd.read_csv(csv_file)
        self.assertAlmostEqual(df.loc[0, "cohens_d"], 0.55)
        self.assertAlmostEqual(df.loc[0, "cohens_d_ci_lower"], 0.2)
        self.assertAlmostEqual(df.loc[0, "pct_difference"], 12.5)

    def test_save_voxel_data(self):
        v_df = pd.DataFrame({"GABRA1": [0.1, 0.2], "GRIN1": [0.3, 0.4]})
        io.save_voxel_data(v_df, "Region 1", self.temp_dir.name)
        csv_file = os.path.join(self.temp_dir.name, "Region_1_voxel_expression.csv")
        self.assertTrue(os.path.isfile(csv_file))
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 2)
        self.assertIn("GABRA1", df.columns)


class TestmRNAExtraction(unittest.TestCase):
    """Tests for mRNA extraction and robust sigmoid normalization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(os.path.dirname(inspect.getfile(io)), "data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_mrna_with_builtin_data(self):
        subunit_path = os.path.join(self.data_dir, "mRNA_images", "GABRA1_mirr_mRNA.nii")
        self.assertTrue(os.path.isfile(subunit_path), f"Built-in file {subunit_path} should exist")
        mask = create_dummy_mask(radius=2)
        norm_values = io.extract_mrna(subunit_path, mask)
        self.assertEqual(len(norm_values), int(np.sum(mask == 1)))
        # Values should be normalized between 0 and 1 via sigmoid
        self.assertTrue(np.all(norm_values >= 0))
        self.assertTrue(np.all(norm_values <= 1))

    def test_get_receptor_data(self):
        mask = create_dummy_mask(radius=2)
        receptors = ["GABRA1", "GRIN1"]
        df = io.get_receptor_data(receptors, mask, self.data_dir)
        self.assertEqual(df.shape[0], int(np.sum(mask == 1)))
        self.assertEqual(list(df.columns), receptors)

    def test_get_receptor_data_empty_mask_raises(self):
        empty_mask = np.zeros((91, 109, 91))
        receptors = ["GABRA1"]
        with self.assertRaises(ValueError):
            io.get_receptor_data(receptors, empty_mask, self.data_dir)


class TestGadgetDataClass(unittest.TestCase):
    """Tests for GadgetData class initialization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(os.path.dirname(inspect.getfile(io)), "data")
        self.mask1_path = os.path.join(self.temp_dir.name, "mask1.nii.gz")
        self.mask2_path = os.path.join(self.temp_dir.name, "mask2.nii.gz")
        save_dummy_nifti(self.mask1_path)
        save_dummy_nifti(self.mask2_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gadget_data_single_region(self):
        data = io.GadgetData([self.mask1_path], ["Region 1"], self.data_dir)
        self.assertFalse(data.multi_region)
        self.assertFalse(data.multi_subject)
        self.assertIn("Region 1", data.mask_images)
        self.assertIn("Region 1", data.receptor_data)
        self.assertGreater(data.receptor_data["Region 1"].shape[0], 0)
        self.assertEqual(data.bground_image.shape, (91, 109, 91))


if __name__ == "__main__":
    unittest.main()
