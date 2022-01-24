# eBOSC: extended Better OSCillation Detection

[![DOI](https://zenodo.org/badge/342610969.svg)](https://zenodo.org/badge/latestdoi/342610969)

![Rhythm](https://jkosciessa.github.io/images/rhythm.png)

## Overview
--------

**eBOSC** (extended Better OSCillation detection) is a toolbox (or a set of scripts) that can be used to detect the occurrence of rhythms in continuous signals (i.e., at the single trial level). It uses a static aperiodic ‘background’ spectrum as the basis to define a ‘power threshold’ that continuous signals have to exceed in order to qualify as ‘rhythmic’. As such, it leverages the observation that stochastic components of the frequency spectrum of neural data are aharacterized by a '1/f'-like power spectrum. An additional ‘duration threshold’ can be set up in advance, or rhythmic episodes can be filtered by duration following detection to ensure that detected rhythmic episodes have a rather sustained vs. transient appearance.

## Documentation
-------------

A project wiki for eBOSC is available [here](https://github.com/jkosciessa/eBOSC/wiki).

* [Motivation](https://github.com/jkosciessa/eBOSC/wiki/Pitfalls)
* [Tutorial](https://github.com/jkosciessa/eBOSC/wiki/Tutorial)
* [Version update/Legacy information](https://github.com/jkosciessa/eBOSC/wiki/Legacy)

Simulation scripts and data files regarding the 2020 NeuroImage paper can be found at https://github.com/jkosciessa/eBOSC_resources_NI2020.

A MATLAB implementation can be found [here](https://github.com/jkosciessa/eBOSC).

## Installation
-------------

Get the latest development version using git:
`git clone https://github.com/jkosciessa/eBOSC_py`

To install this cloned copy, move into the cloned directory and run:
`pip install .`

The example files in the `/data` directory are stored in [Git Large File Storage](https://git-lfs.github.com/). To retrieve them install the lfs (`git lfs install`), and then get the files with `git lfs pull`.

To get started, see the example provided in `examples/eBOSC_example_empirical.ipynb`. Data is based off a FieldTrip structure and conversion scripts to MNE-style format are provided in the `examples` directory.

## Problems?
-------------

If you want to use the tool but encounter issues, or would like to suggest a new feature, feel free to open an [issue](https://github.com/jkosciessa/eBOSC_py/issues).

## Credits
-------------

If you find the method useful, please cite the following papers:

Kosciessa, J. Q., Grandy, T. H., Garrett, D. D., & Werkle-Bergner, M. (2020). Single-trial characterization of neural rhythms: Potential and challenges. NeuroImage, 206, 116331. http://doi.org/10.1016/j.neuroimage.2019.116331

Whitten, T. A., Hughes, A. M., Dickson, C. T., & Caplan, J. B. (2011). A better oscillation detection method robustly extracts EEG rhythms across brain state changes: The human alpha rhythm as a test case. NeuroImage, 54(2), 860–874. http://doi.org/10.1016/j.neuroimage.2010.08.064

## License
-------------

eBOSC is an extension of the BOSC library and partially uses scripts from the original toolbox. These functions are included in the 'external' folder of the current package.

The eBOSC library (and the original BOSC library) are free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The eBOSC library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.