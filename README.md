# collab
Collaborative filtering with the GP-LVM

This repository contains the code for the paper:

["Non-linear Matrix Factorization with Gaussian Processes"](http://www.machinelearning.org/archive/icml2009/papers/384.pdf) by Neil D. Lawrence and Raquel Urtasun. It was published at ICML 2009.

The main code used in the paper is in the matlab subdirectory.

We also worked to do some experiments on netflix with a C++/Python variant of the code. These weren't done in time, but the code is included here although it is not well documented. The code makes use of the swig wrappers around [GPc](https://github.com/SheffieldML/GPc) for creating python objects. These are included as a submodule through [ndlml](https://github.com/SheffieldML/ndlml). 

