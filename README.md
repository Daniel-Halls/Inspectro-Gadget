# InSpectro-Gadget

InSpectro-Gadget is a tool to obtain reference gene expression values from brain regions of interest for a set of neurotransmitter receptors and receptor subunits. Values are estimates for each voxel within the specified region and then presented graphically in a PDF report. Values are also returned as a Python dictionary for additional analysis.

The receptors and receptor subunits included are:

| Receptor           | Genes                                  |
|--------------------|----------------------------------------|
| GABA<sub>A</sub>   | GABRA1, GABRA2, GABRA3, GABRA4, GABRA5 |
| GABA<sub>B</sub>   | GABRB1, GABRB2, GABRB3                 |
|                    | GABRG1, GABRG2, GABRG3                 |
|                    | GABBR1, GABBR2                         |
| NMDA               | GRIN1                                  |
|                    | GRIN2A, GRIN2B, GRIN2C                 |
| AMPA               | GRIA1, GRIA2, GRIA3, GRIA4             |
| Kainate            | GRIK1, GRIK2, GRIK3, GRIK4, GRIK5      |
| mGlu               | GRM1, GRM5                             |
|                    | GRM2, GRM3, GRM4                       |
|                    | GRM5, GRM6, GRM7, GRM8                 |
| mACh               | CHRM1, CHRM2, CHRM3, CHRM4, CHRM5      |
| nACh               | CHRNA2, CHRNA3, CHRNA4, CHRNA5, CHRNA6, CHRNA7, CHRNA9, CHRNA10 |
|                    | CNRNB2, CHRNB3 |
| &alpha;-adrenergic | ADRA1D, ADRA1B, ADRA1A |
|                    | ADRA2A, ADRA2B, ADRA2C |
| &beta;-adrenergic  | ADRB1, ADRB2, ADRB3 |
| Dopamine | D1, D5 |
| | D2, D3, D4 |
| 5-HT | 5HT1A, 5HT1B, 5HT1D, 5HT1E, 5HT1F |
| | 5HT2A, 5HT2B, 5HT2C |
| | 5HT3A, 5HT4, 5HT5A, 5HT6, 5HT7 |

Gene expression data is obtained from the Allen Human Brain Atlas (https://human.brain-map.org/).

## Installation

The package can be installed via pip as follows:

```
pip install https://github.com/lizmcmanus/Inspectro-Gadget/archive/refs/heads/main.zip
```


## Usage
InSpectro-Gadget can be used either as a command-line pipeline tool or imported as a Python package.

Mask regions of interest should be binary images in 2 mm isotropic MNI152 space (`.nii` or `.nii.gz`).

### Command-Line Interface (Pipeline)

InSpectro-Gadget provides a Git-style CLI with three subcommands: `region`, `compare`, and `multiple`.

Each subcommand accepts:
- `-m` / `--mask-inputs`: Paths to mask NIfTI images
- `-l` / `--labels`: Labels for each mask (optional)
- `-o` / `--outdir`: Output directory to save the PDF report and CSV tables (optional)
- `-b` / `--background`: Custom background NIfTI image in MNI152 2mm space (optional)

```bash
# 1. Single region
inspectro-gadget region -m /path/to/mask.nii.gz -l "ACC" -o /path/to/output

# 2. Two region comparison
inspectro-gadget compare -m /path/to/mask1.nii.gz /path/to/mask2.nii.gz -l "ACC" "DLPFC" -o /path/to/output

# 3. Multiple participant comparison
inspectro-gadget multiple -m sub-01.nii.gz sub-02.nii.gz sub-03.nii.gz -l "Sub1" "Sub2" "Sub3" -o /path/to/output
```

You can also run via `gadget <subcommand>` or `python -m inspectro_gadget <subcommand>`.

---

### Python API

#### Single region
To obtain receptor estimates for a single region enter that region's mask:

```python
from inspectro_gadget import gadget

gadget_out = gadget(['/home/data/region1_mask.nii.gz'],
                    mask_labels=['Region 1'],
                    out_root='/home/data/results')
```

The report will include a variety of information:

- An image showing the region being analysed, along with the estimated excitation/inhibition ratio for it

![Brain slices showing the region being analysed. The excitation/inhibition ratio for the region is also shown.](images/single-eiratio.png?raw=true "Single region mask image and excitation/inhibition ratio")

- Radarplots showing the receptor fingerprint of the region of interest. Separate ones are plotted for GABA+glutamate genes and for neuromodulators

![Radarplots showing receptor fingerprints for the region of interest.](images/single-radar.png?raw=true "Region of interest receptor fingerprints")

- Violinplots showing the range of gene expression within the region's mask for each of the included genes. GABAA alpha subunits and alpha2 noradrenaline receptor genes are shown here for illustration

![Example violinplots showing expression levels for the region of interest. GABAA alpha subunits and alpha2 noradrenaline receptor genes are shown for illustration.](images/single-violin.png?raw=true "Expression estimates for GABAA alpha subunits and alpha2 NA receptors")


### Two region comparison
To compare receptor estimates between two regions, enter masks for each. A PDF report will be produced in the current working directory or in the output directory specified.

```
gadget_out = gadget([['/home/data/region1_mask.nii.gz'],
                    ['/home/data/region2_mask.nii.gz']],
                    mask_labels=['Region 1', 'Region 2'],
                    out_root='/home/data/results')
```

The report will include a variety of information:

- An image showing the two regions being analysed, along with the estimated excitation/inhibition ratio for each

![Brain slices showing the two regions being analysed. The excitation/inhibition ratios for each region are also shown.](images/two-eiratio.png?raw=true "Two region mask images and excitation/inhibition ratios")

- Radarplots showing the receptor fingerprints for each of the regions. Separate ones are plotted for GABA+glutamate genes and for neuromodulators

![Radarplots showing receptor fingerprints for the two regions.](images/two-radars.png?raw=true "Two region receptor fingerprints")

- Violinplots showing the range of gene expression within each region's mask for each of the included genes. A comparison of the mean and shape of the distributions are also provided. GABAA alpha subunits and alpha2 noradrenaline receptor genes are shown here for illustration

![Example violinplots showing expression levels for each of the two regions. GABAA alpha subunits and alpha2 noradrenaline receptor genes are shown for illustration.](images/two-violins.png?raw=true "Regional expression estimates for GABAA alpha subunits and alpha2 NA receptors")


### Multiple participant comparison
To analyse the similarity of receptor estimates across a group of participants, enter masks for each one. A participant overlap nifti image will be created in the current working directory or in the output directory specified, along with a PDF report. 

```
gadget_out = gadget(['/home/data/sub-01_region1_mask.nii.gz',
                     '/home/data/sub-02_region1_mask.nii.gz',
                     '/home/data/sub-03_region1_mask.nii.gz',
                     '/home/data/sub-04_region1_mask.nii.gz'],
                    mask_labels=['sub-01', 'sub-02', 'sub-03', 'sub-04'],
                    out_root='/home/data/results')
```

The report will include a variety of information:

- An image showing the overlap between participant masks

![Brain slices showing overlap between the participant masks.](images/multi-overlap.png?raw=true "Multiple participant mask overlap image")

- Estimates excitation/inhibition ratios for each participant's mask (and a group average value)

- ![Excitation/inhibition ratio estimates for each participant.](images/multi-eiratio.png?raw=true "Multiple participant excitation/inhibition ratios")

- Radarplots showing the receptor fingerprints for each participant's mask. Separate ones are plotted for GABA+glutamate genes and for neuromodulators

![Radarplots showing receptor fingerprints for participants.](images/multi-radars.png?raw=true "Multiple participant receptor fingerprints")

- Violinplots showing the range of gene expression within each participant's mask for each of the included genes. How far each participant mask's distribution median is from the group average is also provided. A set of AMPA receptor genes are shown here for illustration

![Example violinplots showing expression levels for each of the included genes. A set of AMPA receptor genes are shown for illustration.](images/multi-violins.png?raw=true "Individual expression estimates for a set of AMPA receptor genes")


### Important limitations
The Allen Human Brain Atlas data comes from six people, five of whom were male. Their ages ranged from 24 to 57 years. As brain morphology and gene expression can differ between the sexes and across the lifespan, the dataset may not be fully representative of the human population. As such, the results produced by this tool should be interpreted with these limitations in mind. 


## Citing
When reporting results obtained with the tool please cite the following work:

_Inspectro-Gadget_

McManus, E., Muhlert, N., Duncan, N.W., 2024. InSpectro-Gadget: A Tool for Estimating Neurotransmitter and Neuromodulator Receptor Distributions for MRS Voxels. _Neuroinform_. https://doi.org/10.1007/s12021-024-09654-w

_Allen Human Brain Atlas_

Hawrylycz, M.J., Lein, E.S., Guillozet-Bongaarts, A.L., Shen, E.H., Ng, L., Miller, J.A., van de Lagemaat, L.N., Smith, K.A., Ebbert, A., Riley, Z.L., Abajian, C., Beckmann, C.F., Bernard, A., Bertagnolli, D., Boe, A.F., Cartagena, P.M., Chakravarty, M.M., Chapin, M., Chong, J., Dalley, R.A. et al. An anatomically comprehensive atlas of the adult human brain transcriptome. _Nature_, 489(7416):391–399, 2012. doi: 10.1038/nature11405

_Whole-brain gene expression images_

Gryglewski, G., Seiger, R., James, G.M., Godbersen, G.M., Komorowski, A., Unterholzner, J., Michenthaler, P., Hahn, A., Wadsak, W., Mitterhauser, M., Kasper, S. and Lanzenberger, R. Spatial analysis and high resolution mapping of the human whole-brain transcriptome for integrative analysis in neuroimaging. _NeuroImage_, 176:259–267, 2018. doi: 10.1016/j.neuroimage.2018.04.068.
