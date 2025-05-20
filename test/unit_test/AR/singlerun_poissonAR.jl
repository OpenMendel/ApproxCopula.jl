using ApproxCopula, LinearAlgebra, GLM
using Random, Distributions, DataFrames, ToeplitzMatrices
using Test, BenchmarkTools

BLAS.set_num_threads(1)
Threads.nthreads()

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
# βtrue = [log(5.0)]
σ2true = [0.5]
ρtrue = [0.5]
trueparams = [βtrue; ρtrue; σ2true]

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
samplesize = 10_000

st = time()
currentind = 1
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 25# number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(1234)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Poisson(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

# form model
gcm = GLMCopulaARModel(gcs)
# precompile
println("precompiling Poisson AR fit")
gcm2 = deepcopy(gcm);
ApproxCopula.fit!(gcm2);

fittime = @elapsed ApproxCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.σ2
@show gcm.ρ
@show gcm.∇β
@show gcm.∇σ2
@show gcm.∇ρ

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show ApproxCopula.confint(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ

@test mseβ < 0.01
@test mseσ2 < 1
@test mseρ < 0.01

println("checking memory allocation for Poisson AR")
# logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false) # this will allocate from threads
# logl_gradient_memory = @benchmark loglikelihood!($(gcm.data[1]), $(gcm.β), $(gcm.ρ[1]), $(gcm.σ2[1]), true, false)
# @test logl_gradient_memory.memory == 0.0
