# ApproxCopula.jl 

*A Flexible Approximate-Copula Distribution for Statistical Modeling*

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://OpenMendel.github.io/ApproxCopula.jl/dev) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://OpenMendel.github.io/ApproxCopula.jl/stable) | [![build Actions Status](https://github.com/OpenMendel/ApproxCopula.jl/workflows/CI/badge.svg)](https://github.com/OpenMendel/ApproxCopula.jl/actions) [![CI (Julia nightly)](https://github.com/openmendel/ApproxCopula.jl/workflows/JuliaNightly/badge.svg)](https://github.com/OpenMendel/ApproxCopula.jl/actions/workflows/JuliaNightly.yml)| [![codecov](https://codecov.io/gh/OpenMendel/ApproxCopula.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/OpenMendel/ApproxCopula.jl) |

ApproxCopula.jl is a Julia package for the analysis of correlated data with specified margins. Currently the package supports multivariate simulation and analysis utilities for the Poisson, Negative Binomial, Bernoulli, Gaussian, Bivariate Poisson/Bernoulli Mixed distributions. ApproxCopula.jl supports covariance matrices structured under the variance component model (VCM) framework, autoregressive AR(1) covariance structure, and the compound symmetric (CS) covariance structure. ApproxCopula.jl supports Julia v1.6 or later. See the [documentation](https://openmendel.github.io/ApproxCopula.jl/dev/) for usage under the different models.

ApproxCopula.jl is a registered package, and it will require running the following code within Julia to install. 

```{julia}
using Pkg
pkg"add ApproxCopula"
```

## Citation

The manuscript for `ApproxCopula.jl` is available on [arXiv](https://arxiv.org/abs/2205.03505).

*A Flexible Quasi-Copula Distribution for Statistical Modeling. Ji SS, Chu BB, Sinsheimer JS, Zhou H, Lange K. arXiv preprint arXiv:2205.03505. 2022 May 6.*

If you use other [OpenMendel](https://openmendel.github.io) analysis packages in your research, please cite the following reference in the resulting publications:

*OPENMENDEL: a cooperative programming project for statistical genetics. Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. Hum Genet. 139, 61–71 (2020). [doi:10.1007/s00439-019-02001-z](10.1007/s00439-019-02001-z). PMID: 30915546*
