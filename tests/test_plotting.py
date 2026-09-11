"""Unit tests for inspectro_gadget.plotting module."""

import os
import inspect
import tempfile
import unittest
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from inspectro_gadget import plotting, io
from tests.helpers import create_dummy_mask, sample_receptor_list


class TestPlotting(unittest.TestCase):
    """Tests for report figure generation and PDF compilation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.temp_dir.name, "test_output.pdf")
        self.bground = np.zeros((91, 109, 91))
        self.mask1 = create_dummy_mask(center=(45, 54, 45), radius=3)
        self.mask2 = create_dummy_mask(center=(50, 60, 50), radius=3)
        self.data_dir = os.path.join(os.path.dirname(inspect.getfile(io)), "data")
        self.real_receptor_list = pd.read_csv(
            os.path.join(self.data_dir, "GroupedReceptors.tsv"), sep="\t"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_masks_single_region(self):
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.plot_masks(
                {"Region 1": self.mask1}, ["Region 1"], self.bground, pdf, {"Region 1": 1.15}
            )
        self.assertTrue(os.path.isfile(self.pdf_path))
        self.assertGreater(os.path.getsize(self.pdf_path), 0)

    def test_plot_masks_two_regions(self):
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.plot_masks(
                {"R1": self.mask1, "R2": self.mask2},
                ["R1", "R2"],
                self.bground,
                pdf,
                {"R1": 1.1, "R2": 0.9},
            )
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_plot_masks_empty_mask_raises(self):
        empty_mask = np.zeros((91, 109, 91))
        with PdfPages(self.pdf_path) as pdf:
            with self.assertRaises(ValueError):
                plotting.plot_masks({"Empty": empty_mask}, ["Empty"], self.bground, pdf)

    def test_plot_overlap(self):
        overlap = self.mask1 * 100.0
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.plot_overlap(overlap, ["Subject overlap"], self.bground, pdf)
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_plot_overlap_empty_raises(self):
        empty_overlap = np.zeros((91, 109, 91))
        with PdfPages(self.pdf_path) as pdf:
            with self.assertRaises(ValueError):
                plotting.plot_overlap(empty_overlap, ["Subject overlap"], self.bground, pdf)

    def test_region_radar_no_hang(self):
        subunits = self.real_receptor_list["subunit"].values
        med_df = pd.DataFrame(
            [np.random.uniform(0.2, 0.8, size=len(subunits))], columns=subunits
        )
        rec_medians = {"Region 1": med_df}
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.region_radar(rec_medians, ["Region 1"], self.real_receptor_list, pdf)
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_multisub_radar(self):
        subunits = self.real_receptor_list["subunit"].values
        med_df = pd.DataFrame(
            np.random.uniform(0.2, 0.8, size=(2, len(subunits))),
            index=["sub-01", "sub-02"],
            columns=subunits,
        )
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.multisub_radar(med_df, self.real_receptor_list, pdf)
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_single_region_violins(self):
        subunits = self.real_receptor_list["subunit"].values
        data = pd.DataFrame(
            np.random.uniform(0.2, 0.8, size=(20, len(subunits))), columns=subunits
        )
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.single_region_violins(data, self.real_receptor_list, pdf)
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_two_region_violins(self):
        subunits = self.real_receptor_list["subunit"].values
        d1 = pd.DataFrame(np.random.uniform(0.2, 0.8, size=(20, len(subunits))), columns=subunits)
        d2 = pd.DataFrame(np.random.uniform(0.2, 0.8, size=(20, len(subunits))), columns=subunits)
        sub_data = {"R1": d1, "R2": d2}
        pcts = {s: 2.5 for s in subunits}
        ds = {s: 0.15 for s in subunits}
        ds_ci = {s: [-0.1, 0.4] for s in subunits}
        kss = {s: 0.1 for s in subunits}

        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.two_region_violins(
                sub_data, self.real_receptor_list, pdf, pcts, ds, ds_ci, kss
            )
        self.assertTrue(os.path.isfile(self.pdf_path))

    def test_multisub_exin(self):
        exin_data = {"sub-01": 0.85, "sub-02": 0.75, "sub-03": 0.90}
        labels = ["sub-01", "sub-02", "sub-03"]
        with PdfPages(self.pdf_path) as pdf:
            pdf = plotting.multisub_exin(exin_data, labels, pdf)
        self.assertTrue(os.path.isfile(self.pdf_path))


if __name__ == "__main__":
    unittest.main()
