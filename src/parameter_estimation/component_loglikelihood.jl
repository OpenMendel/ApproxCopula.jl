"""
    loglik_obs!(d, y, μ, wt, ϕ)

Get the loglikelihood from the GLM.jl package for each observation
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(abs(ϕ))), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = logpdf(Poisson(μ), y)

# this gets the loglikelihood from the glm.jl package for the component density
"""
    component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})

Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T <: BlasReal, D, Link}
    logl = zero(T)
    @inbounds for j in 1:gc.n
        logl += ApproxCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], one(T), one(T))
    end
    logl
end

### loglikelihood functions
"""
    component_loglikelihood!(gc::Union{NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}, r::T)

Calculates the loglikelihood of observing `y` given parameters for `μ` and `r` for Negative Binomial distribution using the GLM.jl package.
"""
function component_loglikelihood(gc::Union{NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}, r::T) where {T <: BlasReal, D<:NegativeBinomial{T}, Link}
    logl = zero(T)
    @inbounds for j in 1:gc.n
        logl += logpdf(D(r, r / (gc.μ[j] + r)), gc.y[j])
    end
    logl
end
