module QuasiCopula

using LinearAlgebra
using Reexport
using GLM, Distributions
using StatsFuns
using Statistics: cov
using ToeplitzMatrices
using DataFrames
using LinearAlgebra: BlasReal, copytri!
using SpecialFunctions
using FFTW
using SnpArrays
using Enzyme
using ProgressMeter
using Random
@reexport using Ipopt
import Base: show, fill!
import MathOptInterface as MOI

export fit!, update_θ_jensen!, init_β!, initialize_model!, loglikelihood!, standardize_res!, std_res_differential!
export update_res!, update_θ!
export update_∇θ!, update_Hθ! # update gradient and hessian of variance components
export glm_regress_jl, glm_regress_model, glm_score_statistic!  # these are to initialize our model
export component_loglikelihood, glm_gradient, hessian_glm
export GLMCopulaVCObs, GLMCopulaVCModel
export Poisson_Bernoulli_VCObs, Poisson_Bernoulli_VCModel
# for GWAS
export MixedCopulaVCObs, MixedCopulaVCModel
export MultivariateCopulaVCModel
export GWASCopulaVCModel_autodiff
export GWASCopulaVCModel_autodiff_fast
export multivariateGWAS_autodiff
# for GWAS simulations
export simulate_random_snparray
export simulate_multivariate_traits
export simulate_longitudinal_traits

include("parameter_estimation/gaussian_CS.jl")
include("parameter_estimation/NBCopulaCS.jl")
include("parameter_estimation/GLM_CS.jl")
include("parameter_estimation/GLM_VC.jl")
include("parameter_estimation/gaussian_VC.jl")
include("parameter_estimation/gaussian_AR.jl")
include("parameter_estimation/NBCopulaAR.jl")
include("parameter_estimation/NBCopulaVC.jl")
include("parameter_estimation/GLM_AR.jl")
include("generate_random_deviates/discrete_rand.jl")
include("generate_random_deviates/continuous_rand.jl")
include("generate_random_deviates/multivariate_rand.jl")
include("parameter_estimation/update_sigma_and_residuals.jl")
include("parameter_estimation/initialize_model.jl")
include("parameter_estimation/component_loglikelihood.jl")
include("parameter_estimation/gradient_hessian.jl")
include("parameter_estimation/fit_glm_ar_cs.jl")
include("parameter_estimation/fit_gaussian_ar_cs.jl")
include("parameter_estimation/fit_glm_vc.jl")
include("parameter_estimation/fit_nb.jl")
include("parameter_estimation/fit_gaussian_vc.jl")
include("parameter_estimation/inference_ci.jl")
include("model_interface/AR_interface.jl")
include("model_interface/CS_interface.jl")
include("model_interface/VC_interface.jl")
include("model_interface/show_io.jl")
include("gwas/longitudinal.jl")
include("gwas/longitudinal_autodiff.jl")
include("gwas/longitudinal_autodiff_fast.jl")
include("gwas/longitudinal_enzyme.jl")
include("gwas/multivariate.jl")
# include("gwas/multivariate_VC.jl")
# include("gwas/multivariate_gwas.jl")
# include("gwas/multivariate_gwas_autodiff.jl")
include("gwas/utilities.jl")

function config_solver(solver::MOI.AbstractOptimizer, solver_config::Dict)
    for (key, val) in solver_config
        MOI.set(solver, MOI.RawOptimizerAttribute(key), val)
    end
end

end # module
