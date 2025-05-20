@reexport using Distributions
import Distributions: mean, var, logpdf, pdf, cdf, maximum, minimum, insupport, quantile, rand, rand!, params
export DiscreteUnivariateCopula, pdf_constants
export pmf_copula!, reorder_pmf

struct DiscreteUnivariateCopula{
    DistT <: DiscreteUnivariateDistribution,
    T     <: Real
    } <: DiscreteUnivariateDistribution
    d  :: DistT
    μ :: T
    σ2 :: T
    c0 :: T
    c1 :: T
    c2 :: T
    c  :: T # normalizing constant
    maxvalue :: T
    y_sample :: Vector{T}
    pmf_vec :: Vector{T}
end

"""
    DiscreteUnivariateCopula(d, c0, c1, c2)
The distribution with density `c * P(x = x) * (c0 + c1 * x + c2 * x^2)`, where `f`
is the density of the base distribution `d` and `c` is the normalizing constant.
"""
function DiscreteUnivariateCopula(
    d  :: DistT,
    c0 :: T,
    c1 :: T,
    c2 :: T) where {DistT <: DiscreteUnivariateDistribution, T <: Real}
    μ = mean(d)
    σ2 = var(d)
    c  = inv(c0 + c1 * μ + c2 * (σ2 + abs2(μ)))
    Tc = typeof(c)
    maxvalue = quantile(Base.typename(typeof(d)).wrapper(Distributions.params(d)...), 0.999999999999)
    y_sample = collect(0:maxvalue)
    pmf_vec = zeros(length(y_sample)) # marginal pmf
    DiscreteUnivariateCopula(d, Tc(μ), Tc(σ2), Tc(c0), Tc(c1), Tc(c2), c, Tc(maxvalue), Tc.(y_sample), Tc.(pmf_vec))
end

"""
    pdf_constants(Γ::Matrix{<:Real}, res::Vector{<:Real}, i::Int64, dist::DiscreteUnivariateDistribution)
This function will fill out the appropriate constants, c0, c1, c2 for each conditional distribution to form the `DiscreteUnivariateCopula` structure.
"""
function pdf_constants(Γ::Matrix{T}, res::Vector{T}, i::Int64, dist::DiscreteUnivariateDistribution) where T <: Real
    μ = Distributions.mean(dist)
    σ2 = Distributions.var(dist)
    c_0 = μ^2 * inv(σ2)
    c__0 = μ * inv(sqrt(σ2)) * sum(crossterm_res(res, i, Γ))
    c_1 = -2μ * inv(σ2)
    c__1 = inv(sqrt(σ2)) * sum(crossterm_res(res, i, Γ))
    c_2 = inv(σ2)
    storage = zeros(i-1)
    # first multiply Γ[1:i-1, 1:i-1] *  res[1:i-1]
    mul!(storage, Γ[1:i-1, 1:i-1], res[1:i-1])
    # then multiply to get transpose(res[1:i-1]) *  Γ[1:i-1, 1:i-1] *  res[1:i-1]
    c0 = 1 + 0.5 * dot(storage, res[1:i-1]) +  0.5 * tr(Γ[i+1:end, i+1:end]) + 0.5 * Γ[i, i] * c_0  - c__0
    c1 = 0.5 * Γ[i, i] * c_1  + c__1
    c2 = 0.5 * Γ[i, i] * c_2
    DiscreteUnivariateCopula(dist, c0, c1, c2)
end

"""
    pdf_constants(γ::T, dist::DiscreteUnivariateDistribution)
This function will fill out the appropriate constants, c0, c1, c2 for the univariate marginal distribution to form the `DiscreteUnivariateCopula` structure.
When the number of observations in the cluster is 1
"""
function pdf_constants(γ::T, dist::DiscreteUnivariateDistribution) where T <: Real
    μ = Distributions.mean(dist)
    σ2 = Distributions.var(dist)
    c_0 = μ^2 * inv(σ2)
    c_1 = -2μ * inv(σ2)
    c_2 = inv(σ2)
    c0 = 1 + 0.5 * γ * c_0
    c1 = 0.5 * γ * c_1
    c2 = 0.5 * γ * c_2
    DiscreteUnivariateCopula(dist, c0, c1, c2)
end

"""
    pmf_copula!(dist::DiscreteUnivariateCopula)
This function will get the appropriate probability mass function, using our copula density.
For discrete distributions with countably infinite values in the range, we want to find some approximate maximum value that is large enough so that the probability mass vector sums to about 1.
"""
function pmf_copula!(dist::DiscreteUnivariateCopula)
    for k in 1:length(dist.y_sample)
        dist.pmf_vec[k] = pdf(dist, dist.y_sample[k]) # get the pmf given maximum and our pdf implementation
    end
    nothing
end

"""
    reorder_pmf(pmf::Vector{<:Real}, μ)
This function will re-order the probability mass function, by sorting the vector of probabilities in decreasing order (starting with mean μ).
"""
function reorder_pmf(pmf::Vector{T}, μ) where T <: Real
    listofj = zeros(Int64, length(pmf))
    k = Integer(floor.(μ))
    reordered_pmf = zeros(length(pmf))
    i = 1
    j = k[1]
    while(i < length(pmf) && j > 0 && j < length(pmf))
        listofj[i] = j
        reordered_pmf[i] = pmf[j + 1]
        if i%2 == 1
            j = j + i
            elseif i%2 == 0
            j = j - i
        end
        i = i + 1
    end
    if j == 0
        listofj[i] = 0
        reordered_pmf[i] = pmf[1]
        for s in i+1:length(pmf)
            listofj[s] = s - 1
            reordered_pmf[s] = pmf[s]
            end
        end
    listofj, reordered_pmf
end

"""
    rand(dist::DiscreteUnivariateCopula)
This function will simulate the discrete random variable under our copula model.
"""
function rand(dist::DiscreteUnivariateCopula)
    pmf_copula!(dist) # get pmf under our copula density
    listofj, reordered_pmf = reorder_pmf(dist.pmf_vec, dist.μ) # re-order the pmf
    # reorder_pmf!(dist)
    sample = rand() # generate x from uniform(0, 1)
    (random_deviate, s) = listofj[1], reordered_pmf[1] # if the cumulative probability mass is less than the P(X = listofj[1]) then leave it as the mean
    for i in 2:length(reordered_pmf)
        if sample < s
            random_deviate = listofj[i - 1]
            break
        else
            s += reordered_pmf[i]
        end
    end
    random_deviate
end

"""
    rand(dist::DiscreteUnivariateCopula, n_reps::Int64)
This function will simulate the discrete random variable under our copula model, n_reps times.
"""
function rand(dist::DiscreteUnivariateCopula, n_reps::Int64)
    random_deviate = zeros(Int64, n_reps)
    for l in 1:n_reps
        random_deviate[l] = rand(dist)
    end
    random_deviate
end

"""
    rand!(dist::DiscreteUnivariateCopula, sample::Vector{T})
This function will write over each entry in the specified sample vector, the simulated univariate discrete values under our copula model.
"""
function rand!(dist::DiscreteUnivariateCopula, sample::Vector{T}) where T <: Real
    for i in 1:length(sample)
        sample[i] = rand(dist)
    end
    sample
end

"""
    rand(dist::DiscreteUnivariateCopula)
This function will simulate the discrete random variable under our copula model.
"""
function rand(dist::DiscreteUnivariateCopula{Bernoulli{T}, T}) where T <: Real
    random_deviate = Float64(rand(Bernoulli(ApproxCopula.mean(dist))))
    random_deviate
end

"""
    rand(dist::DiscreteUnivariateCopula, n_reps::Int64)
This function will simulate the discrete random variable under our copula model, n_reps times.
"""
function rand(dist::DiscreteUnivariateCopula{Bernoulli{T}, T}, n_reps::Int64) where T <: Real
    random_deviate = zeros(Int64, n_reps)
    for l in 1:n_reps
        random_deviate[l] = rand(dist)
    end
    random_deviate
end

"""
    rand!(dist::DiscreteUnivariateCopula, sample::Vector{T})
This function will write over each entry in the specified sample vector, the simulated univariate discrete values under our copula model.
"""
function rand!(dist::DiscreteUnivariateCopula{Bernoulli{T}, T}, sample::Vector{T}) where T <: Real
    for i in 1:length(sample)
        sample[i] = rand(dist)
    end
    sample
end
