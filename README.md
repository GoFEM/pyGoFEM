# Motivation

Python front-end for the **GoFEM** package. 

See jupyter notebooks for examples on how to setup GoFEM with python and process the results.

# Installation

### Linux

It is easiest to setup the pyGoFEM via [conda](https://docs.conda.io/en/latest/). Just deploy the environment using the environment file *pyGoFEM.yml* as:

```
conda env create -f pyGoFEM.yml
conda activate pygofem
```

That is it! You should now be able to run the tutorial notebooks.

### Windows

On Windows, you can use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl). Note that WSL version 2 or newer is needed. 

### macOS

The process is identical to **Linux**. However, you may need to replace linux compiler packages in the *pyGoFEM.yml* files. Specifically, these

- gcc_linux-64
- gxx_linux-64
- gfortran_linux-64

should be replaced by

- clang_osx-64
- clangxx_osx-64
- gfortran_osx-64

For more details check [this](https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html).

# Referencing

When used, please aknowledge GoFEM by citing one or more of the following references

> Grayver A.V., 2015, *Parallel 3D magnetotelluric inversion using adaptive finite-element method. Part I: theory and synthetic study*, Geophysical Journal International, 202(1), pp. 584-603, doi: 10.1093/gji/ggv165

> Grayver A.V., and Kolev, T. V., 2015, *Large-scale 3D geo-electromagnetic modeling using parallel adaptive high-order finite element method*, Geophysics, 80(6), pp. 277-291, doi: 10.1190/GEO2015-0013.1
