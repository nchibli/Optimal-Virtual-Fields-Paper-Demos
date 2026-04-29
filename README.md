# Welcome to [Chibli, Genet & Imperiale. A class of optimal virtual fields for inverse problems in elasticity. In revision.]'s demos!

Static and interactive demos can be found at [https://nchibli.github.io/Optimal-Virtual-Fields-Paper-Demos](https://nchibli.github.io/Optimal-Virtual-Fields-Paper-Demos/), or directly on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nchibli/Optimal-Virtual-Fields-Paper-Demos/master?urlpath=lab/tree/./demos).


 ## Installation

A working installation of **FEniCS** (version 2019.1.0, including the Dolfin Python interface) and **VTK** (including the Python interface) is required to run the code.  
The simplest way to set up your system is using **Conda**:

```bash
conda create -y -c conda-forge -n fenics fenics=2019.1.0 matplotlib=3.5 mpi4py=3.1.3 numpy=1.24 scipy=1.10 pandas=1.3 pip python=3.10  
conda activate fenics
