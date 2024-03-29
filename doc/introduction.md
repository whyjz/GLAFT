# Introduction

GLAcier Feature Tracking testkit (GLAFT) is a Python package for assessing and benchmarking feature-tracked glacier velocity maps derived from satellite imagery. To be compatible with as many feature-tracking tools as possible, GLAFT analyzes velocity maps (and optional reliability files used as weight) and calculates two metrics based on statistics and ice flow dynamics. Along with GLAFT’s visualization tools, users can intercompare the quality of velocity maps processed by different methods and parameter sets. In the [GLAFT publication](https://doi.org/10.5194/tc-17-4063-2023), we further provide a guideline for optimizing glacier velocity maps by comparing the calculated metrics to an ideal threshold value.

GLAFT is an open sourced project and is hosted on Github (https://github.com/whyjz/GLAFT). All documentation and cloud-executable demos are deployed as Jupyter Book pages (https://whyjz.github.io/GLAFT/). 

## Installation

**Try GLAFT without installing**: We recommend running our [Quick Start notebook on MyBinder.org](https://mybinder.org/v2/gh/whyjz/glacier-ft-test/master?urlpath=tree/jupyter-book/doc/quickstart.ipynb).

**For cloud access**: We recommend using the [Ghub portal to launch GLAFT](https://theghub.org/tools/glaft/status) (registration required).

**For local installation**: GLAFT is available on PyPI and can be installed via `pip`. 

```
pip install glaft
```

## License

GLAFT uses the MIT License. More information is available [here](https://github.com/whyjz/GLAFT/blob/master/LICENSE).

## Citing GLAFT

Please consider citing the following references when using GLAFT:

#### Tool, metrics, and relevant discussion

- Zheng, W., Bhushan, S., Van Wyk De Vries, M., Kochtitzky, W., Shean, D., Copland, L., Dow, C., Jones-Ivey, R., and Pérez, F.: GLAcier Feature Tracking testkit (GLAFT): a statistically and physically based framework for evaluating glacier velocity products derived from optical satellite image feature tracking, The Cryosphere, 17, 4063–4078, https://doi.org/10.5194/tc-17-4063-2023, 2023.

#### Tool only

- Zheng, W., Bhushan, S., & Sundell, E. (2023). whyjz/GLAFT: GLAFT [version #]. Zenodo. https://doi.org/10.5281/zenodo.7527956