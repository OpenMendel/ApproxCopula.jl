export vcov!, coef, stderror, confint, MSE, coverage!, logl, get_CI
"""
    vcov!(gcm::GLMCopulaVCModel)
Calculate the asymptotic covariance of the parameters,
based on values `gcm.Hββ`, `gcm.Hθ`, `gcm.data[i].∇β`,
`gcm.data[i].∇θ`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    p, m = gcm.p, gcm.m
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[          1:p,                 1:p      ] = gcm.Hβ
    gcm.Ainv[    (p + 1):(p + m),     (p + 1):(p + m)] = gcm.Hθ
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇θ)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    vcov!(gcm::GaussianCopulaVCModel)
For the Gaussian base, calculate the asymptotic covariance of the parameters,
based on expected information `gcm.Hββ`, `gcm.Hθ`, `gcm.data[i].∇β`,
`gcm.data[i].∇θ`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::GaussianCopulaVCModel{T}) where {T <: BlasReal}
    fill!(gcm.Hθ, 0)
    for i in 1:length(gcm.data)
        qsum  = dot(gcm.θ, gcm.data[i].q)
        tsum = dot(gcm.θ, gcm.data[i].t)
        inv1pq = inv(1 + qsum)
        inv1pt = inv(1 + tsum)
        gcm.data[i].m1 .= gcm.data[i].q
        gcm.data[i].m1 .*= inv1pq
        gcm.data[i].m2 .= gcm.data[i].t
        gcm.data[i].m2 .*= inv1pt
        # hessian for vc
        fill!(gcm.data[i].Hθ, 0.0)
        BLAS.syr!('U', one(T), gcm.data[i].m2, gcm.data[i].Hθ)
        BLAS.syr!('U', -one(T), gcm.data[i].m1, gcm.data[i].Hθ)
        copytri!(gcm.data[i].Hθ, 'U')
        gcm.Hθ .+= gcm.data[i].Hθ
    end
    p, m = gcm.p, gcm.m
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[          1:p,                 1:p      ] = gcm.Hβ
    gcm.Ainv[          p + 1:p + 1,                 p + 1:p + 1      ] = gcm.Hτ
    gcm.Ainv[    (p + 2):(p + 1 + m),     (p + 2):(p + 1 + m)] = gcm.Hθ
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇τ)
        copyto!(gcm.ψ, p + 2, obs.∇θ)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    vcov!(gcm::NBCopulaVCModel)
Calculate the asymptotic covariance of the parameters,
based on values `gcm.Hββ`, `gcm.Hθ`, `gcm.Hr`, `gcm.data[i].∇β`,
`gcm.data[i].∇θ`, `gcm.data[i].∇r`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    p, m = gcm.p, gcm.m
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[          1:p,                 1:p      ] = gcm.Hβ
    gcm.Ainv[          p + 1:p + 1,                 p + 1:p + 1      ] = gcm.Hr
    gcm.Ainv[    (p + 2):(p + 1 + m),     (p + 2):(p + 1 + m)] = gcm.Hθ
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇r)
        copyto!(gcm.ψ, p + 2, obs.∇θ)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    vcov!(gcm::GLMCopulaCSModel)
Calculate the asymptotic covariance of the parameters,
based on values `gcm.Hββ`, `gcm.Hρ`, `gcm.Hσ2`, `gcm.data[i].∇β`,
`gcm.data[i].∇ρ`, `gcm.data[i].∇σ2`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::Union{GLMCopulaCSModel{T, D, Link}, GLMCopulaARModel{T, D, Link}}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    p = gcm.p
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[1:p, 1:p] = gcm.Hβ
    # gcm.Ainv[1:p, (p + 1)] = gcm.Hβρ
    # gcm.Ainv[1:p, (p + 2)] = gcm.Hβσ2
    gcm.Ainv[(p + 1) : (p + 1), (p + 1) : (p + 1)] = gcm.Hρ
    gcm.Ainv[(p + 2) : (p + 2), (p + 2) : (p + 2)] = gcm.Hσ2
    # gcm.Ainv[(p + 1) : (p + 1), (p + 2) : (p + 2)] = gcm.Hρσ2
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇ρ)
        copyto!(gcm.ψ, p + 2, obs.∇σ2)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    vcov!(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}})
Calculate the asymptotic covariance of the parameters,
based on values `gcm.Hββ`, `gcm.Hτ`, `gcm.Hρ`, `gcm.Hσ2`, `gcm.data[i].∇β`, `gcm.data[i].∇τ`,
`gcm.data[i].∇ρ`, `gcm.data[i].∇σ2`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}}) where {T <: BlasReal}
    p = gcm.p
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[1:p, 1:p] = gcm.Hβ
    gcm.Ainv[(p + 1) : (p + 1), (p + 1) : (p + 1)] = gcm.Hρ
    gcm.Ainv[(p + 2) : (p + 2), (p + 2) : (p + 2)] = gcm.Hσ2
    gcm.Ainv[(p + 3) : (p + 3), (p + 3) : (p + 3)] = gcm.Hτ
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇ρ)
        copyto!(gcm.ψ, p + 2, obs.∇σ2)
        copyto!(gcm.ψ, p + 3, obs.∇τ)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    vcov!(gcm::NBCopulaARModel, NBCopulaCSModel)
Calculate the asymptotic covariance of the parameters,
based on values `gcm.Hββ`, `gcm.Hr`, `gcm.Hρ`, `gcm.Hσ2`, `gcm.data[i].∇β`, `gcm.data[i].∇r`,
`gcm.data[i].∇ρ`, `gcm.data[i].∇σ2`, and `gcm.vcov` is updated and returned.
"""
function vcov!(gcm::Union{NBCopulaARModel{T, D, Link}, NBCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    p = gcm.p
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[1:p, 1:p] = gcm.Hβ
    gcm.Ainv[(p + 1) : (p + 1), (p + 1) : (p + 1)] = gcm.Hρ
    gcm.Ainv[(p + 2) : (p + 2), (p + 2) : (p + 2)] = gcm.Hσ2
    gcm.Ainv[(p + 3) : (p + 3), (p + 3) : (p + 3)] = gcm.Hr
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇ρ)
        copyto!(gcm.ψ, p + 2, obs.∇σ2)
        copyto!(gcm.ψ, p + 3, obs.∇r)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * pinv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    nothing
end

"""
    coef(gcm::GLMCopulaVCModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    [gcm.β; gcm.θ]
end

"""
    coef(gcm::GLMCopulaVCModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::GaussianCopulaVCModel{T}) where {T <: BlasReal}
    [gcm.β; gcm.τ; gcm.θ]
end

"""
    coef(gcm::NBCopulaVCModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    [gcm.β; gcm.r; gcm.θ]
end

"""
    coef(gcm::GLMCopulaARModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::Union{GLMCopulaARModel{T, D, Link}, GLMCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    [gcm.β; gcm.ρ; gcm.σ2]
end

"""
    coef(gcm::NBCopulaARModel, NBCopulaCSModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::Union{NBCopulaARModel{T, D, Link}, NBCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    [gcm.β; gcm.ρ; gcm.σ2; gcm.r]
end

"""
    coef(gcm::Union{GaussianCopulaARModel, GaussianCopulaCSModel)
Get the estimated parameter coefficients from the model.
"""
function coef(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}}) where {T <: BlasReal}
    [gcm.β; gcm.ρ; gcm.σ2; gcm.τ]
end

"""
    stderror(gcm::GLMCopulaVCModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + gcm.m)]
end

"""
    stderror(gcm::GLMCopulaVCModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::GaussianCopulaVCModel{T}) where {T <: BlasReal}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + gcm.m + 1)]
end

"""
    stderror(gcm::GLMCopulaVCModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + gcm.m + 1)]
end

"""
    stderror(gcm::GLMCopulaARModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::Union{GLMCopulaARModel{T, D, Link}, GLMCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + 2)]
end

"""
    stderror(gcm::NBCopulaARModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::Union{NBCopulaARModel{T, D, Link}, NBCopulaCSModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + 3)]
end

"""
    stderror(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}})
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
function stderror(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}}) where {T <: BlasReal}
    [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + 3)]
end

"""
    confint(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, level::Real)
Get the confidence interval for each of the estimated parameters at level (default level = 95%).
"""
confint(gcm::Union{GLMCopulaVCModel, GaussianCopulaVCModel, GLMCopulaARModel, GLMCopulaCSModel, NBCopulaVCModel, GaussianCopulaARModel, GaussianCopulaCSModel, NBCopulaARModel, NBCopulaCSModel}, level::Real) = hcat(QuasiCopula.coef(gcm) + QuasiCopula.stderror(gcm) * quantile(Normal(), (1. - level) / 2.), QuasiCopula.coef(gcm) - QuasiCopula.stderror(gcm) * quantile(Normal(), (1. - level) / 2.))

confint(gcm::Union{GLMCopulaVCModel, GaussianCopulaVCModel, GLMCopulaARModel, GLMCopulaCSModel, NBCopulaVCModel, GaussianCopulaARModel, GaussianCopulaCSModel, NBCopulaARModel, NBCopulaCSModel}) = confint(gcm, 0.95)

"""
    MSE(gcm::GLMCopulaVCModel, β::Vector, θ::Vector)
Get the mean squared error of the parameters `β` and `θ`.
"""
function MSE(gcm::GLMCopulaVCModel{T, D, Link}, β::Vector, θ::Vector) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseθ = sum(abs2, gcm.θ .- θ) / gcm.m
    return mseβ, mseθ
end

"""
    MSE(gcm::GLMCopulaVCModel, β::Vector, τ::Float64, θ::Vector)
Get the mean squared error of the parameters `β`, `τ` and `θ`.
"""
function MSE(gcm::GaussianCopulaVCModel{T}, β::Vector, τ::T, θ::Vector) where {T <: BlasReal}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseτ = sum(abs2, inv(gcm.τ[1]) - inv(τ))
    mseθ = sum(abs2, gcm.θ .- θ) / gcm.m
    return mseβ, mseτ, mseθ
end

"""
    MSE(gcm::NBCopulaVCModel, β::Vector, r::T, θ::Vector)
Get the mean squared error of the parameters `β`, `r` and `θ`.
"""
function MSE(gcm::NBCopulaVCModel{T, D, Link}, β::Vector, r::T, θ::Vector) where {T <: BlasReal, D, Link}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mser = sum(abs2, gcm.r .- r)
    mseθ = sum(abs2, gcm.θ .- θ) / gcm.m
    return mseβ, mser, mseθ
end

"""
    MSE(gcm::Union{GLMCopulaARModel, GLMCopulaCSModel}, β::Vector, ρ::Vector, σ2::Vector)
Get the mean squared error of the parameters `β` , `ρ` and `σ2`.
"""
function MSE(gcm::Union{GLMCopulaARModel{T, D, Link}, GLMCopulaCSModel{T, D, Link}}, β::Vector, ρ::Vector, σ2::Vector) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseρ = sum(abs2, gcm.ρ .- ρ)
    mseσ2 = sum(abs2, gcm.σ2 .- σ2)
    return mseβ, mseρ, mseσ2
end

"""
    MSE(gcm::NBCopulaARModel, β::Vector, ρ::Vector, σ2::Vector)
Get the mean squared error of the parameters `β` , `ρ` and `σ2`.
"""
function MSE(gcm::Union{NBCopulaARModel{T, D, Link}, NBCopulaCSModel{T, D, Link}}, β::Vector, ρ::Vector, σ2::Vector, r::T) where {T <: BlasReal, D, Link}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseρ = sum(abs2, gcm.ρ .- ρ)
    mseσ2 = sum(abs2, gcm.σ2 .- σ2)
    mser = sum(abs2, gcm.r .- r)
    return mseβ, mseρ, mseσ2, mser
end

"""
    MSE(gcm::Union{GaussianCopulaARModel, GaussianCopulaCSModel}, β::Vector, τ::Float64, Σ::Vector)
Get the mean squared error of the parameters `β`, `τ` and `Σ`.
"""
function MSE(gcm::Union{GaussianCopulaARModel{T}, GaussianCopulaCSModel{T}}, β::Vector, τ::T, ρ::Vector, σ2::Vector) where {T <: BlasReal}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseρ = sum(abs2, gcm.ρ .- ρ)
    mseσ2 = sum(abs2, gcm.σ2 .- σ2)
    mseτ = sum(abs2, inv(gcm.τ[1]) - inv(τ))
    return mseβ, mseρ, mseσ2, mseτ
end

"""
    coverage!(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, trueparams::Vector,
intervals::Matrix, curcoverage::Vector)
Find the coverage of the estimated parameters `β` and `θ`, given the true parameters.
"""
function coverage!(gcm::Union{GLMCopulaVCModel, GaussianCopulaVCModel, GLMCopulaARModel, GLMCopulaCSModel, NBCopulaVCModel, GaussianCopulaARModel, GaussianCopulaCSModel, NBCopulaARModel, NBCopulaCSModel}, trueparams::Vector,
    intervals::Matrix, curcoverage::Vector)
    copyto!(intervals, confint(gcm))
    lbs = @views intervals[:, 1]
    ubs = @views intervals[:, 2]
    map!((val, lb, ub) -> val >= lb &&
        val <= ub, curcoverage, trueparams, lbs, ubs)
    return curcoverage
end

"""
    logl(gcm)
Get the loglikelihood at the given parameters in gcm, at the optimal solution.

# Arguments
- `gcm`: One of `GaussianCopulaVCModel`, `GaussianCopulaARModel`, `GaussianCopulaCSModel`, `GLMCopulaVCModel`, `GLMCopulaARModel`, `GLMCopulaCSModel`, `NBCopulaVCModel`, `NBCopulaARModel`, `NBCopulaCSModel` model objects.
"""
logl(gcm) = loglikelihood!(gcm, false, false)

"""
    get_CI(gcm)
Get the confidence interval of all parameters, at the optimal solution.

# Arguments
- `gcm`: One of `GaussianCopulaVCModel`, `GaussianCopulaARModel`, `GaussianCopulaCSModel`, `GLMCopulaVCModel`, `GLMCopulaARModel`, `GLMCopulaCSModel`, `NBCopulaVCModel`, `NBCopulaARModel`, `NBCopulaCSModel` model objects.
"""
function get_CI(gcm)
    loglikelihood!(gcm, true, true)
    vcov!(gcm)
    QuasiCopula.confint(gcm)
end
