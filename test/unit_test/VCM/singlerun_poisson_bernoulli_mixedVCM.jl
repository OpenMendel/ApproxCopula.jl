using QuasiCopula, LinearAlgebra, Random, GLM
using DataFrames, Statistics
using BenchmarkTools, Test

BLAS.set_num_threads(1)
Threads.nthreads()

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
# βtrue = [0.2; 0.1; -0.05; 0.2; 0.1; -0.1]
Random.seed!(123345)
βtrue = rand(Uniform(-0.2, 0.2), 2 * p)
θtrue = [0.1]

# generate data
intervals = zeros(2 * p + m, 2) #hold intervals
curcoverage = zeros(2 * p + m) #hold current coverage resutls
trueparams = [βtrue; θtrue] #hold true parameters

#simulation parameters
samplesize = 10000
ni = 2

currentind = 1
d1 = Poisson()
d2 = Bernoulli()
vecdist = [d1, d2]

link1 = LogLink()
link2 = LogitLink()
veclink = [link1, link2]

T = Float64
VD = typeof(vecdist)
VL = typeof(veclink)

Random.seed!(12345)
Xstack = [ones(samplesize) randn(samplesize, p - 1)]
gcs = Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}(undef, samplesize)
Γ = θtrue[1] * ones(ni, ni)
vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)

for i in 1:samplesize
    xi = Xstack[i, :]
    X = [transpose(xi) zeros(size(transpose(xi))); zeros(size(transpose(xi))) transpose(xi)]
    η = X * βtrue
    μ = zeros(ni)
    for j in 1:ni
        μ[j] = GLM.linkinv(veclink[j], η[j])
        vecd[j] = typeof(vecdist[j])(μ[j])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = Poisson_Bernoulli_VCObs(y, xi, V, vecdist, veclink)
end

gcm = Poisson_Bernoulli_VCModel(gcs)
# precompile
println("precompiling Bivariate Mixed VCM fit")
gcm2 = deepcopy(gcm);
QuasiCopula.fit!(gcm2);

fittime = @elapsed QuasiCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.θ
@show gcm.∇β
@show gcm.∇θ

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show QuasiCopula.confint(gcm)
# mse under our model
mseβ, mseθ = MSE(gcm, βtrue, θtrue)
@show mseβ
@show mseθ

@test mseβ < 0.01
@test mseθ < 0.01
