# STELES Pipeline

**Python data reduction pipeline for the STELES high-resolution echelle spectrograph**

> Experimental public testing release

---

## Overview

The **STELES Data Reduction Pipeline** is the official Python-based data reduction pipeline for the **SOAR TELescope Echelle Spectrograph (STELES)**, developed for automated and reproducible reduction of STELES spectroscopic data.

The pipeline processes raw multi-extension CCD FITS data into fully calibrated one-dimensional spectra suitable for scientific analysis.

Current processing includes:

- Detector-level CCD reduction
- Overscan correction and amplifier merging
- Cosmic ray removal
- Master flat / arc / science frame generation
- Spectral order identification, tracing and refinement
- Order extraction and geometric rectification
- Wavelength calibration
- Spatial distortion correction
- Blaze function modeling and correction
- Automated quality control diagnostics
- JSON and HTML reduction reports

---

## Instrument Overview

STELES is a high-resolution optical echelle spectrograph installed at the **4.1-m SOAR Telescope**, located at Cerro Pachón, Chile.

Instrument characteristics:

- Dual-channel operation:
  - Blue channel
  - Red channel
- High spectral resolution (up to ~50k depending on the slit width)
- Multi-extension CCD detector format
- Cross-dispersed echelle format
- Designed for precision stellar spectroscopy

---

## Development Status

⚠️ **This software is under active development.**

This repository currently represents a **testing release** intended for early users.

The pipeline is functional, but interfaces, configuration parameters, output products, and internal implementations may change.

Current status:

| Feature                        | Status     |
|--------------------------------|------------|
| CCD reduction                  | ✅ Stable  |
| Order tracing                  | ✅ Stable  |
| Spectral extraction            | ✅ Stable  |
| Wavelength calibration         | ✅ Stable  |
| Spatial distortion correction  | ✅ Stable  |
| Blaze correction               | ✅ Stable  |
| Quality control diagnostics    | ✅ Stable  |
| Flux calibration               | 🚧 Planned |
| Sky subtraction                | 🚧 Planned |
| Optimal extraction             | 🚧 Planned |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/navarete/STELES.git
cd STELES/pipeline/
```

Create a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(MUST create requirements.txt)

pip-compile --cert=None --client-cert=None --index-url=None --output-file=requirements.txt --pip-args=None requirements.in

---

## Dependencies

Main dependencies include:

- Python >= 3.10
- astropy
- ccdproc
- datetime
- glob
- json
- lacosmic
- logging
- matplotlib
- numpy
- os
- pandas
- scipy
- time

Additional packages may be required depending on future extensions.

---

## Quick Start

Minimal example:

```python
config = {
    "run_stages": ["all"],
    "overwrite": True,
    "raw_directory": "/home/user/STELES/YYYY-MM-DD/red_channel/",

    "flat_list": [flat_0001, flat_0002, flat_0003, flat_0004, flat_0005],
    "arc_list": [arc_1s_0006, arc_60s_0007],
    "sci_list": [sci_0008],
    
    "fileout_flat": "master_flat",
    "fileout_sci":  "master_science",
    "fileout_arc":  "master_ThAr",

    "average_sigma_clip": 5,

    "saturation_threshold_for_arc": 40000,
    
    "trace_source":               "sci",
    "refit_trace_skip_orders":   [None],
    "fail_on_trace_intersection": False,
    
    "aperture_orders": [0, 9, 19, 29, 39],
    "aperture_vmax": {"flat": 4.0, "arc": 3.5, "sci": 4.0},
    
    "x_extract": [-12,+4],

    "window_fit_peak":             11,
    "wavesolution_max_fwhm":     10.0,
    "wavelength_pixel_offset":     -2,
    "wavesolution_ndeg":            3,
    "wavesolution_niter":           3,
    "wavesolution_rms_threshold": 2.0,
    
    "spatdist_ndeg":         2,
    "spatdist_npix_step":    1,
    "spatdist_saturation_threshold": 1000000,
    "spatdist_niter":        3,
    "spatdist_sigma_init": 3.0,
    "spatdist_sigma_min":  1.0,
    "spatdist_sigma_step": 0.5,

    "blaze_mode":   "simple",
    "blaze_min_deg":      15,
    "blaze_max_deg":      55,
    "blaze_sigma":         5,
    "qc_blaze_ncols":      7,
    "qc_blaze_n_panels":  10,
    "qc_snr_n_panels":     4,
    "qc_final1d_n_panels": 4    
}

pipeline_partI = STELES_Pipeline_PartI(config)
pipeline_partI.run()

pipeline_partII = STELES_Pipeline_PartII(config)
pipeline_partII.run()
```

---

## Pipeline Architecture

Processing is divided into two stages:

### Part I — Basic CCD Reduction & Spectral Extraction

Includes:

- Overscan correction
- Amplifier gain normalization
- Extension merging
- Cosmic ray cleaning
- Arc scaling
- Master frame generation
- Spectral order tracing
- Aperture validation
- Rectified order extraction

Outputs:

```text
master_flat_linear.fits
master_arc_linear.fits
master_science_linear.fits
```

---

### Part II — Calibration & Final Extraction

Includes:

- Wavelength calibration
- Spatial distortion modeling
- Application of calibration solutions
- Blaze modeling
- Final one-dimensional extraction
- QC diagnostics

Outputs:

```text
*_wavelength_solution.fits
*_distortion.csv
*_blaze.fits
*_final_1d.fits
```

---

## Input Data Requirements

Required:

- Quartz flat-field exposures
- ThAr arc lamp exposures
- Science observations

Recommended:

- High S/N spectrophotometric standard star for tracing

Optional (not yet implemented):

- Bias frames
- Dark frames
- Sky frames

---

## Configuration

Pipeline behavior is controlled through a Python configuration dictionary.

Main parameter groups:

- Execution control
- Detector parameters
- Input data selection
- Trace configuration
- Extraction parameters
- Wavelength calibration
- Spatial distortion
- Blaze correction
- Quality control settings

See the documentation for full configuration details.

---

## Quality Control

Automated QC products include:

### Part I

- Master frame diagnostics
- Trace validation plots
- Aperture validation

### Part II

- RMS wavelength solution histograms
- Residual distributions
- Dispersion stability
- Blaze validation panels
- Final spectral diagnostic products

All plots are automatically saved in:

```text
RED/plots/
```

---

## Output Products

Typical directory structure:

```text
raw_directory/
└── RED/
    ├── plots/
    ├── db_tracing_refit/
    ├── db_wavelength/
    ├── db_distortion/
    ├── pipeline_report_partI.html
    ├── pipeline_report_partII.html
    └── calibrated products
```

---

## Documentation

Detailed documentation is available in:

- Instrument User Manual (coming soon)
- Example notebooks (planned)

---

## Testing & Feedback

This release is currently being validated by early testers.

If you encounter issues, please report:

- operating system
- Python version
- input dataset description
- configuration used
- full error traceback
- diagnostic plots (if relevant)

Feedback on:

- installation
- usability
- documentation clarity
- scientific validity
- output quality

is highly appreciated.

---

## Known Limitations

Current limitations include:

- No flux calibration
- No sky subtraction
- No optimal extraction
- Limited automatic handling of pathological tracing cases
- Configuration currently requires manual setup

---

## Citation / Acknowledgment

If you use STELES Pipeline in scientific work, please acknowledge:

> STELES data were reduced using the STELES Python data reduction pipeline developed by the STELES instrument team.


---

## Contributing

Contributions, bug reports, and feature suggestions are welcome.

Please open an issue or contact the maintainers.

---

## License

[GPL / proprietary testing release]

---

## Contact

Pipeline maintainer:

**Felipe Navarete**  
Laboratório Nacional de Astrofísica (LNA)  
Brazil

GitHub Issues preferred for bug reports.
