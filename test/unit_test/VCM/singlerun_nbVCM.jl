using ApproxCopula, LinearAlgebra, Random, GLM
using DataFrames, Statistics
using BenchmarkTools, Test

BLAS.set_num_threads(1)
Threads.nthreads()

p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
# try next
βtrue = rand(Uniform(-2, 2), p_fixed)
θtrue = [0.5]
rtrue = 10.0

d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

samplesize = 10000
ni = 25

gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = θtrue[1] * ones(ni, ni)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η)
    p = rtrue ./ (μ .+ rtrue)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
end

# form VarLmmModel
gcm = NBCopulaVCModel(gcs)
# precompile
println("precompiling NB VCM fit")
gcm2 = deepcopy(gcm);
ApproxCopula.fit!(gcm2, maxBlockIter = 1);

fittime = @elapsed ApproxCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.θ
@show gcm.r
@show gcm.∇β
@show gcm.∇θ
@show gcm.∇r

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show ApproxCopula.confint(gcm)
# mse and time under our model
mseβ, mser, mseθ = MSE(gcm, βtrue, rtrue, θtrue)
@show mseβ
@show mser
@show mseθ

@test mseβ < 0.01
@test mseθ < 0.01
@test mser < 0.1
