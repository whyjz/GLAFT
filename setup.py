# A minimal setup.py file to make a Python project installable.

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f]


setuptools.setup(
    name             = "glaft",
    version          = "0.2.0",
    author           = "The GLAFT team",
    author_email     = "whyjz@berkeley.edu",
    maintainer       = "Whyjay Zheng",
    maintainer_email = "whyjz@berkeley.edu",
    description      = "GLAFT evaluates the quality of glacier velocity maps using statistics and physics based metrics.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/whyjz/GLAFT.git",
    license          = "MIT",
    packages         = setuptools.find_packages(),
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires  = '>= 3.7',
    install_requires = requirements,
)