# Motivation

Python front-end for the **GoFEM** package. 

See jupyter notebooks for examples on how to setup GoFEM with python and process the results.

# Installation

*Note: instructions below should work for most linux/OSX systems. I did not test this on Windows.*

It is easiest to setup the pyGoFEM via [conda](https://docs.conda.io/en/latest/). First, deploy the environment 

```
conda env create -f gofem_environment.yml
conda activate gofem
```

Then, get the latest [deal.II](https://github.com/dealii/dealii/) library. The library needs to be configured with the python bindings:

```
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/deal.II -DCMAKE_BUILD_TYPE=Release -DDEAL_II_WITH_THREADS=OFF -DDEAL_II_WITH_UMFPACK=OFF -DDEAL_II_WITH_ZLIB=ON -DDEAL_II_COMPONENT_PYTHON_BINDINGS=ON ../
make install
```

After this, you should be able to run the tutorial notebooks.

# Referencing

When used, please aknowledge GoFEM by citing one or more of the following references

> Grayver A.V., 2015, *Parallel 3D magnetotelluric inversion using adaptive finite-element method. Part I: theory and synthetic study*, Geophysical Journal International, 202(1), pp. 584-603, doi: 10.1093/gji/ggv165

> Grayver A.V., and Kolev, T. V., 2015, *Large-scale 3D geo-electromagnetic modeling using parallel adaptive high-order finite element method*, Geophysics, 80(6), pp. 277-291, doi: 10.1190/GEO2015-0013.1
