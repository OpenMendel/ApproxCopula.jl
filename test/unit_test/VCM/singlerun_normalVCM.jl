using QuasiCopula, LinearAlgebra, Random, GLM
using DataFrames, Statistics
using BenchmarkTools, Test

BLAS.set_num_threads(1)
Threads.nthreads()

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
θtrue = [0.5]
τtrue = 100.0
σ2 = inv(τtrue)
σ = sqrt(σ2)

#simulation parameters
samplesize = 10000
ni = 25

# true Gamma
Γ = θtrue[1] * ones(ni, ni)

T = Float64
gcs = Vector{GaussianCopulaVCObs{T}}(undef, samplesize)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    μ = X * βtrue
    vecd = Vector{ContinuousUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Normal(μ[i], σ)
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V)
end

# form VarLmmModel
gcm = GaussianCopulaVCModel(gcs)
# precompile
println("precompiling Gaussian VCM fit")
gcm2 = deepcopy(gcm);
QuasiCopula.fit!(gcm2);

fittime = @elapsed QuasiCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.θ
@show gcm.τ
@show gcm.∇β
@show gcm.∇θ
@show gcm.∇τ

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show QuasiCopula.confint(gcm)

mseβ, mseτ, mseθ = MSE(gcm, βtrue, τtrue[1], θtrue)
@show mseβ
@show mseτ
@show mseθ

@test mseβ < 0.01
@test mseθ < 0.01
@test mseτ < 0.01
