# Motivation

Python front-end for the **GoFEM** package. 

See jupyter notebooks for examples on how to setup GoFEM with python and process the results.

# Installation

**Note**: instructions below should work for linux and OSX systems. On Windows, you can use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl).

It is easiest to setup the pyGoFEM via [conda](https://docs.conda.io/en/latest/). Just deploy the environment using the environment file *pyGoFEM.yml* as:

```
conda env create -f pyGoFEM.yml
conda activate pygofem
```

That is it! You should now be able to run the tutorial notebooks.

# Referencing

When used, please aknowledge GoFEM by citing one or more of the following references

> Grayver A.V., 2015, *Parallel 3D magnetotelluric inversion using adaptive finite-element method. Part I: theory and synthetic study*, Geophysical Journal International, 202(1), pp. 584-603, doi: 10.1093/gji/ggv165

> Grayver A.V., and Kolev, T. V., 2015, *Large-scale 3D geo-electromagnetic modeling using parallel adaptive high-order finite element method*, Geophysics, 80(6), pp. 277-291, doi: 10.1190/GEO2015-0013.1
