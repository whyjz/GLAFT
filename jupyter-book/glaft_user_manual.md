# Introduction

GLAcier Feature Tracking testkit (GLAFT) is based on the scientific Python ecosystem and focuses on calculating the metrics and benchmarking glacier velocity products derived using the feature tracking technique. To be compatible with most feature tracking tools, GLAFT aims to process only the most common and essential product, which are the velocity maps (and optional input of reliability file used as weight). GLAFT also provides visualization tools for the derived metrics, making the scientific communication much easier. 

GLAFT is an open source project with all source code hosted on Github (https://github.com/whyjz/GLAFT). Users can find relevant documentation and cloud-executable demos in the same repository and on its Jupyter Book-based Github pages (https://whyjz.github.io/GLAFT/). 

## Installation

**For cloud access**: users are recommended to use the [Ghub portal and launch GLAFT](https://theghub.org/tools/glaft/status) (registration required).

**For local installation**: GLAFT will be available on PyPI soon. Before that happens, one can download the [Github repository](https://github.com/whyjz/GLAFT), navigate to the repository folder on a terminal, and enter the following command  to install:

```
pip install .
```

Alternatively, for a development installation you can type

```
pip install -e .
```