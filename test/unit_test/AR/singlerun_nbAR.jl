using ApproxCopula, LinearAlgebra, GLM
using Random, Distributions, DataFrames, ToeplitzMatrices
using Test, BenchmarkTools

BLAS.set_num_threads(1)
Threads.nthreads()

p_fixed = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p_fixed)
rtrue = 10.0
σ2true = [0.5]
ρtrue = [0.5]

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = vec[i - 1] * ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    Matrix(V)
end

#simulation parameters
samplesize = 10000

st = time()
currentind = 1
d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 25 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

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
    # push!(Ystack, y)
    V = [ones(ni, ni)]
    # V = [ones(ni, ni)]
    gcs[i] = NBCopulaARObs(y, X, d, link)
end

# form model
gcm = NBCopulaARModel(gcs)
# precompile
println("precompiling NB AR fit")
gcm2 = deepcopy(gcm);
ApproxCopula.fit!(gcm2, maxBlockIter = 1);

fittime = @elapsed ApproxCopula.fit!(gcm, maxBlockIter = 5, tol=1e-6)
@show fittime
@show gcm.β
@show gcm.σ2
@show gcm.ρ
@show gcm.∇β
@show gcm.∇σ2
@show gcm.∇ρ

@show gcm.r
@show gcm.∇r

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show ApproxCopula.confint(gcm)

# mse and time under our model
mseβ, mseρ, mseσ2, mser = MSE(gcm, βtrue, ρtrue, σ2true, rtrue)
@show mseβ
@show mser
@show mseσ2
@show mseρ

@test mseβ < 0.01
@test mseσ2 < 1
@test mseρ < 0.01
@test mser < 0.1
