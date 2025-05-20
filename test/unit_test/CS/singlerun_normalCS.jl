using QuasiCopula, LinearAlgebra, GLM
using Random, Distributions, DataFrames, ToeplitzMatrices
using Test, BenchmarkTools

BLAS.set_num_threads(1)
Threads.nthreads()

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
σ2true = [0.5]
ρtrue = [0.5]
τtrue = 100.0
σ2 = inv(τtrue)
σ = sqrt(σ2)

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end

#simulation parameters
samplesize = 10000
ni = 25

V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

T = Float64
gcs = Vector{GaussianCopulaCSObs{T}}(undef, samplesize)

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
    gcs[i] = GaussianCopulaCSObs(y, X)
end

gcm = GaussianCopulaCSModel(gcs)
# precompile
println("precompiling Gaussian CS fit")
gcm2 = deepcopy(gcm);
QuasiCopula.fit!(gcm2);

loglikelihood!(gcm, true, true)
fittime = @elapsed QuasiCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.σ2
@show gcm.ρ
@show gcm.∇β
@show gcm.∇σ2
@show gcm.∇ρ

@show gcm.τ
@show gcm.∇τ

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show QuasiCopula.confint(gcm)

mseβ, mseρ, mseσ2, mseτ = MSE(gcm, βtrue, τtrue[1], ρtrue, σ2true)
@show mseβ
@show mseτ
@show mseσ2
@show mseρ

@test mseβ < 0.01
@test mseσ2 < 1
@test mseρ < 0.01
@test mseτ < 0.01
