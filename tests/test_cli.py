"""Unit and integration tests for inspectro_gadget CLI subcommands."""

import os
import tempfile
import unittest
from unittest.mock import patch

from inspectro_gadget.cli import create_parser, run_cli
from inspectro_gadget.version import __version__
from tests.helpers import create_dummy_mask, save_dummy_nifti


class TestCLISubcommands(unittest.TestCase):
    """Test CLI parsing, subcommands, validation, and execution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mask1_path = os.path.join(self.temp_dir.name, "mask1.nii.gz")
        self.mask2_path = os.path.join(self.temp_dir.name, "mask2.nii.gz")
        self.mask3_path = os.path.join(self.temp_dir.name, "mask3.nii.gz")
        m1 = create_dummy_mask(center=(45, 54, 45), radius=3)
        m2 = create_dummy_mask(center=(50, 60, 50), radius=3)
        m3 = create_dummy_mask(center=(48, 55, 48), radius=3)
        save_dummy_nifti(self.mask1_path, data=m1)
        save_dummy_nifti(self.mask2_path, data=m2)
        save_dummy_nifti(self.mask3_path, data=m3)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parser_help_and_version(self):
        parser = create_parser()
        self.assertEqual(parser.prog, "inspectro-gadget")
        # Test version flag
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_no_arguments_returns_error(self):
        ret = run_cli([])
        self.assertNotEqual(ret, 0)

    @patch("inspectro_gadget.cli.gadget")
    def test_region_subcommand_success(self, mock_gadget):
        ret = run_cli([
            "region",
            "-m", self.mask1_path,
            "-l", "SingleROI",
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[self.mask1_path],
            mask_labels=["SingleROI"],
            out_root=self.temp_dir.name,
            bground_fname=None,
        )

    @patch("inspectro_gadget.cli.gadget")
    def test_region_subcommand_default_label(self, mock_gadget):
        ret = run_cli([
            "region",
            "-m", self.mask1_path,
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[self.mask1_path],
            mask_labels=None,
            out_root=self.temp_dir.name,
            bground_fname=None,
        )

    def test_region_subcommand_nonexistent_file(self):
        ret = run_cli([
            "region",
            "-m", "/non/existent/mask.nii.gz",
        ])
        self.assertNotEqual(ret, 0)

    def test_region_subcommand_too_many_masks(self):
        ret = run_cli([
            "region",
            "-m", self.mask1_path, self.mask2_path,
        ])
        self.assertNotEqual(ret, 0)

    def test_region_subcommand_too_many_labels(self):
        ret = run_cli([
            "region",
            "-m", self.mask1_path,
            "-l", "Label1", "Label2",
        ])
        self.assertNotEqual(ret, 0)

    @patch("inspectro_gadget.cli.gadget")
    def test_compare_subcommand_success(self, mock_gadget):
        ret = run_cli([
            "compare",
            "-m", self.mask1_path, self.mask2_path,
            "-l", "ROI_A", "ROI_B",
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[[self.mask1_path], [self.mask2_path]],
            mask_labels=["ROI_A", "ROI_B"],
            out_root=self.temp_dir.name,
            bground_fname=None,
        )

    @patch("inspectro_gadget.cli.gadget")
    def test_compare_subcommand_default_labels(self, mock_gadget):
        ret = run_cli([
            "compare",
            "-m", self.mask1_path, self.mask2_path,
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[[self.mask1_path], [self.mask2_path]],
            mask_labels=None,
            out_root=self.temp_dir.name,
            bground_fname=None,
        )

    def test_compare_subcommand_wrong_mask_count(self):
        ret = run_cli([
            "compare",
            "-m", self.mask1_path,
        ])
        self.assertNotEqual(ret, 0)

    def test_compare_subcommand_wrong_label_count(self):
        ret = run_cli([
            "compare",
            "-m", self.mask1_path, self.mask2_path,
            "-l", "ROI_A",
        ])
        self.assertNotEqual(ret, 0)

    @patch("inspectro_gadget.cli.gadget")
    def test_multiple_subcommand_success(self, mock_gadget):
        ret = run_cli([
            "multiple",
            "-m", self.mask1_path, self.mask2_path, self.mask3_path,
            "-l", "Sub1", "Sub2", "Sub3",
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[self.mask1_path, self.mask2_path, self.mask3_path],
            mask_labels=["Sub1", "Sub2", "Sub3"],
            out_root=self.temp_dir.name,
            bground_fname=None,
            multi_violin=True,
        )

    @patch("inspectro_gadget.cli.gadget")
    def test_multiple_subcommand_no_multi_violin(self, mock_gadget):
        ret = run_cli([
            "multiple",
            "-m", self.mask1_path, self.mask2_path,
            "--no-multi-violin",
            "-o", self.temp_dir.name,
        ])
        self.assertEqual(ret, 0)
        mock_gadget.assert_called_once_with(
            mask_fnames=[self.mask1_path, self.mask2_path],
            mask_labels=None,
            out_root=self.temp_dir.name,
            bground_fname=None,
            multi_violin=False,
        )

    def test_multiple_subcommand_insufficient_masks(self):
        ret = run_cli([
            "multiple",
            "-m", self.mask1_path,
        ])
        self.assertNotEqual(ret, 0)

    def test_multiple_subcommand_label_count_mismatch(self):
        ret = run_cli([
            "multiple",
            "-m", self.mask1_path, self.mask2_path,
            "-l", "Sub1", "Sub2", "Sub3",
        ])
        self.assertNotEqual(ret, 0)

    def test_background_file_not_found(self):
        ret = run_cli([
            "region",
            "-m", self.mask1_path,
            "-b", "/non/existent/bg.nii.gz",
        ])
        self.assertNotEqual(ret, 0)

    def test_end_to_end_region_cli(self):
        # Actual pipeline execution through run_cli without mocks
        out_dir = os.path.join(self.temp_dir.name, "cli_region_out")
        ret = run_cli([
            "region",
            "-m", self.mask1_path,
            "-l", "TestCLIROI",
            "-o", out_dir,
        ])
        self.assertEqual(ret, 0)
        # Verify no subdirectories are created inside out_dir
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        self.assertEqual(len(subdirs), 0)
        # Verify files are saved directly in out_dir
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "gadget-output.pdf")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "receptor-medians.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "excitation-inhibition-ratios.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "TestCLIROI_voxel_expression.csv")))


if __name__ == "__main__":
    unittest.main()
