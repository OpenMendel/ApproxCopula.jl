@reexport using Distributions
import Distributions: rand
export MultivariateMix, NonMixedMultivariateDistribution, cov, cor, simulate_nobs_independent_vectors, covariance_matrix, correlation_matrix
# ## create new type not in Distributions for multivariate mixture

"""
    MultivariateMix(vecd::AbstractVector{<:UnivariateDistribution}, Γ)
An N dimensional `MultivariateMix` constructed from a vector of N `UnivariateDistribution`s, and the user-specified covariance matrix, Γ.
Since Distributions.jl does not currently allow for multivariate vectors with mixed continuous and discrete components, this type is not of super type Distributions.MultivariateDistribution.
We will pre-allocate a Vector of `ContinuousUnivariateCopula`s and `DiscreteUnivariateCopula`s to then fill in the appropriate constants c0, c1, c2, recursively.
"""
struct MultivariateMix{
    V<:AbstractVector{<:UnivariateDistribution},
    T<: BlasReal
}
    vecd::V
    Γ::Matrix{T}
    trΓ::Float64
    gc_obs::Vector{Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}}
    function MultivariateMix(vecd::V, Γ::Matrix{T}) where {T <: BlasReal,
        V<:AbstractVector{<:UnivariateDistribution}}
        n = length(vecd)
        trΓ = tr(Γ)
        gc_obs = Vector{Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}}(undef, n)
        return new{V, T}(vecd, Γ, trΓ, gc_obs)
    end
end

"""
    NonMixedMultivariateDistribution(vecd::AbstractVector{<:UnivariateDistribution}, Γ)
An N dimensional `MultivariateDistribution` constructed from a vector of N `UnivariateDistribution`s, and the user-specified covariance matrix, Γ.
We will pre-allocate a Vector of `ContinuousUnivariateCopula`s or `DiscreteUnivariateCopula`s to then fill in the appropriate constants c0, c1, c2, recursively.
"""
struct NonMixedMultivariateDistribution{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    V<:AbstractVector{T},
}
    vecd::V
    Γ::Matrix{Float64}
    trΓ::Float64
    gc_obs::Vector{Union{DiscreteUnivariateCopula, ContinuousUnivariateCopula}}
    function NonMixedMultivariateDistribution(vecd::V, Γ::Matrix{Float64}) where
        V<:AbstractVector{T} where
        T<:UnivariateDistribution{S} where
        S<:ValueSupport
        n = length(vecd)
        trΓ = tr(Γ)
        gc_obs = Vector{Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}}(undef, n)
        return new{S, T, V}(vecd, Γ, trΓ, gc_obs)
    end
end

####
function rand(gc_vec::NonMixedMultivariateDistribution{S, T, V},
    Y::Vector{Float64},
    res::Vector{Float64}) where {S<: ValueSupport, T <: UnivariateDistribution{S}, V<:AbstractVector{T}}
    for i in 1:length(gc_vec.vecd)
        # form constants for conditional density of i given 1, ..., i-1
        gc_vec.gc_obs[i] = pdf_constants(gc_vec.Γ, res, i, gc_vec.vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = rand(gc_vec.gc_obs[i])
        # update residuals 1,..., i-1
        res[i] = update_res!(Y[i], res[i], gc_vec.gc_obs[i])
    end
    res[end] = update_res!(Y[end], res[end], gc_vec.gc_obs[end])
    Y
end

# ### we need a function that is like rand, but cannot use the distributions framework.
function rand(gc_vec::MultivariateMix{V, T},
    Y::Vector{T},
    res::Vector{T}) where {T <: BlasReal, V<:AbstractVector{<:UnivariateDistribution}}
    for i in 1:length(gc_vec.vecd)
        # form constants for conditional density of i given 1, ..., i-1
        gc_vec.gc_obs[i] = pdf_constants(gc_vec.Γ, res, i, gc_vec.vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = rand(gc_vec.gc_obs[i])
        # update residuals 1,..., i-1
        res[i] = update_res!(Y[i], res[i], gc_vec.gc_obs[i])
    end
    res[end] = update_res!(Y[end], res[end], gc_vec.gc_obs[end])
    Y
end

"""
    simulate_nobs_independent_vectors(multivariate_distribution::Union{NonMixedMultivariateDistribution, MultivariateMix}, n_obs::Integer)
Simulate n_obs independent realizations from multivariate copula-like distribution.
"""
function simulate_nobs_independent_vectors(
    multivariate_distribution::Union{NonMixedMultivariateDistribution, MultivariateMix},
    n_obs::Integer)
    dimension = length(multivariate_distribution.vecd)
    Y = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    res = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    for i in 1:n_obs
        rand(multivariate_distribution, Y[i], res[i])
    end
    Y
end

"""
    logpdf(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}, x::Real)
Theoretical log pdf under the copula model.
"""
function logpdf(multivar::Union{MultivariateMix, NonMixedMultivariateDistribution}, Y::Vector{T}) where T <: BlasReal
    logl = zero(T)
    for i in 1:length(multivar.vecd)
        logl += logpdf(multivar.gc_obs[i], Y[i])
    end
    logl
end

"""
    cov(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix}, k::Int64, l::Int64)
Theoretical covariance of the kth and lth element in the random vector.
"""
function Distributions.cov(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix}, k::Int64, l::Int64)
    cov = sqrt(var(gc_vec.vecd[k])) * sqrt(var(gc_vec.vecd[l])) * gc_vec.Γ[k, l] * inv(1 + 0.5 * gc_vec.trΓ)
    return cov
end

"""
    cor(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix}, k::Int64, l::Int64)
Theoretical correlation of the kth and lth element in the random vector.
"""
function Distributions.cor(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix}, k::Int64, l::Int64)
    cor = gc_vec.Γ[k, l] / (sqrt(1 + 0.5 * gc_vec.trΓ + gc_vec.Γ[k, k]) * sqrt(1 + 0.5 * gc_vec.trΓ + gc_vec.Γ[l, l]) )
    return cor
end

"""
    covariance_matrix(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix})
Theoretical covariance matrix of a random vector.
"""
function covariance_matrix(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix})
    n = length(gc_vec.gc_obs)
    Covariance = zeros(n, n)
    for i in 1:n
        Covariance[i, i] = ApproxCopula.var(gc_vec.gc_obs[i])
        for j = i+1:n
            Covariance[j, i] = ApproxCopula.cov(gc_vec, j, i)
            Covariance[i, j] = Covariance[j, i]
        end
    end
    Covariance
end

"""
    correlation_matrix(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix})
Theoretical covariance matrix of a random vector.
"""
function correlation_matrix(gc_vec::Union{NonMixedMultivariateDistribution, MultivariateMix})
    n = length(gc_vec.gc_obs)
    Corr = zeros(n, n)
    for i in 1:n
        Corr[i, i] = 1.0
        for j = i+1:n
            Corr[j, i] = ApproxCopula.cor(gc_vec, j, i)
            Corr[i, j] = Corr[j, i]
        end
    end
    Corr
end
