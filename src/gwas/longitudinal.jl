struct Storages{T <: BlasReal}
    vec_p::Vector{T}
    vec_maxd::Vector{T}
    denom::Vector{T}
    denom2::Vector{T}
    p_storage::Vector{T}
    p_storage2::Vector{T}
    m_storage::Vector{T}
    m_storage2::Vector{T}
    pm_storage::Vector{T} # length p*m (GLM case) or p*m+1 (Gaussian/NegBin case)
    maxd_storage::Vector{T}
end
function storages(p::Int, maxd::Int, m::Int, n_nuisance::Int)
    # p = number of fixed effects
    # maxd = maximum number of observation within a sample
    # m = number of variance components
    vec_p = zeros(p)
    vec_maxd = zeros(maxd)
    denom = zeros(1)
    denom2 = zeros(1)
    p_storage = zeros(p)
    p_storage2 = zeros(p)
    m_storage = zeros(m)
    m_storage2 = zeros(m)
    pm_storage = zeros(p+m+n_nuisance)
    maxd_storage = zeros(maxd)
    return Storages(vec_p, vec_maxd, denom, denom2, p_storage, 
        p_storage2, m_storage, m_storage2, pm_storage, maxd_storage)
end
function Base.fill!(s::Storages, x::Number)
    fill!(s.vec_p, x)
    fill!(s.vec_maxd, x)
    fill!(s.denom, x)
    fill!(s.denom2, x)
    fill!(s.p_storage, x)
    fill!(s.p_storage2, x)
    fill!(s.m_storage, x)
    fill!(s.m_storage2, x)
    fill!(s.pm_storage, x)
    fill!(s.maxd_storage, x)
end

"""
    GWASCopulaVCModel(qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel}, G::SnpArray)

Performs score tests for each SNP in `G`, given a fitted (null) model on the 
non-genetic covariates.

# Inputs
+ `qc_model`: A fitted `GaussianCopulaVCModel`, `GLMCopulaVCModel` or `NBCopulaVCModel` 
    that includes `n` sample points and `p` non-genetic covariates.
+ `G`: A `SnpArray` (compressed `.bed/bim/fam` PLINK file) with `n` samples and 
    `q` SNPs

# Outputs
A length `q` vector of p-values storing the score statistics for each SNP
"""
function GWASCopulaVCModel(
    qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel},
    G::SnpArray;
    check_grad::Bool=true,
    ncores = Threads.nthreads(),
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p = length(qc_model.β) # number of fixed effects in each sample
    m = length(qc_model.θ) # number of variance components in each sample
    s = typeof(qc_model) <: GLMCopulaVCModel ? 0 : 1 # number of nuisance parameters
    maxd = maxclustersize(qc_model) # max number of observations in 1 sample
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")
    
    # test statistics
    pvals = zeros(T, q)
    χ2 = Chisq(1)

    # preallocated arrays for efficiency
    z = [zeros(T, n) for _ in 1:ncores]
    zi = [zeros(T, maxd) for _ in 1:ncores]
    W = [zeros(T, p + m + s) for _ in 1:ncores]
    storage = [storages(p, maxd, m, s) for _ in 1:ncores]
    ∇resγ_store = [zeros(T, maxd) for _ in 1:ncores]
    ∇resβ_store = [zeros(T, maxd, p) for _ in 1:ncores]

    # timers
    Wtime = [0.0 for _ in 1:ncores]
    Qtime = [0.0 for _ in 1:ncores]
    Rtime = [0.0 for _ in 1:ncores]
    grad_res_time = [0.0 for _ in 1:ncores]
    othertime = [0.0 for _ in 1:ncores]
    scoretest_time = [0.0 for _ in 1:ncores]

    # compute Pinv (inverse negative Hessian)
    Pinv = get_Pinv(qc_model)

    # compute Γ = [θ[k] .* V[k] for k in 1:m] (this assumes each sample has same covariance)
    Γ = zeros(T, maxd, maxd)
    i = findfirst(qc -> qc.n == maxd, qc_model.data)
    for k in 1:m
        BLAS.axpy!(qc_model.θ[k], qc_model.data[i].V[k], Γ) # Γ .+= θ .* V[k]
    end

    # score test for each SNP
    @showprogress Threads.@threads for j in 1:q
        # sync thread storages
        tid = Threads.threadid()
        Wₜ = W[tid]
        zₜ = z[tid]
        ziₜ = zi[tid]
        storeₜ = storage[tid]
        ∇resγ_storeₜ = ∇resγ_store[tid]
        ∇resβ_storeₜ = ∇resβ_store[tid]
        # sync vectors
        SnpArrays.copyto!(zₜ, @view(G[:, j]), center=true, scale=false, impute=true)
        Q, R = zero(T), zero(T)
        fill!(Wₜ, zero(T))
        fill!(storeₜ, zero(T))
        # thread timers
        Wtime_t = 0.0
        Qtime_t = 0.0
        Rtime_t = 0.0
        grad_res_time_t = 0.0
        othertime_t = 0.0
        # loop over each sample
        @inbounds for i in 1:n
            # variables for current sample
            qc = qc_model.data[i]
            d = qc.n # number of observations for current sample
            fill!(ziₜ, zₜ[i])
            Γstore = @view(Γ[1:d, 1:d])
            # update gradient of residual with respect to β and γ
            grad_res_time_t += @elapsed begin
                ∇resγ = @view(∇resγ_storeₜ[1:d])
                ∇resβ = @view(∇resβ_storeₜ[1:d, :])
                get_∇resγ!(∇resγ, qc_model, i, @view(ziₜ[1:d]))
                get_∇resβ!(∇resβ, qc_model, i)
            end
            othertime_t += @elapsed begin
                denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
                denom2 = abs2(denom)
                storeₜ.denom[1] = denom
                storeₜ.denom2[1] = denom2
            end
            # calculate W
            Wtime_t += @elapsed update_W!(Wₜ, qc_model, i, @view(ziₜ[1:d]), Γstore, ∇resβ, ∇resγ, storeₜ)
            # update Q
            Qtime_t += @elapsed Q += calculate_Qi(qc_model, i, @view(ziₜ[1:d]), Γstore, ∇resγ, denom, denom2, storeₜ)
            # update R
            Rtime_t += @elapsed R += calculate_Ri(qc_model, i, @view(ziₜ[1:d]), Γstore, ∇resγ, denom, storeₜ)
        end
        # score test
        scoretest_time_t = @elapsed begin
            mul!(storeₜ.pm_storage, Pinv, Wₜ)
            S = R * inv(Q - dot(Wₜ, storeₜ.pm_storage)) * R
            pvals[j] = ccdf(χ2, S)
        end
        # update progress
        Wtime[tid] += Wtime_t
        Qtime[tid] += Qtime_t
        Rtime[tid] += Rtime_t
        grad_res_time[tid] += grad_res_time_t
        othertime[tid] += othertime_t
        scoretest_time[tid] += scoretest_time_t
    end

    # update timer
    Wtime = sum(Wtime) / ncores
    Qtime = sum(Qtime) / ncores
    Rtime = sum(Rtime) / ncores
    grad_res_time = sum(grad_res_time) / ncores
    othertime = sum(othertime) / ncores
    scoretest_time =  sum(scoretest_time) / ncores
    @show Wtime
    @show Qtime
    @show Rtime
    @show grad_res_time
    @show othertime
    @show scoretest_time

    return pvals
end

# update W expression for 1 sample
# i stands for sample i, and zi is a d-vector of SNP value
function update_W!(W, qc_model::GLMCopulaVCModel, i::Int, zi::AbstractVector, 
    Γ, ∇resβ, ∇resγ, storages::Storages)
    p = qc_model.p
    qc = qc_model.data[i]
    get_neg_Hβγ_i!(@view(W[1:p]), qc, Γ, ∇resβ, ∇resγ, zi, storages) # exact
    get_neg_Hθγ_i!(@view(W[p+1:end]), qc, qc_model.θ, ∇resγ, storages) # exact
    return W
end

function update_W!(W, qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resβ, ∇resγ, storages::Storages)
    p = qc_model.p
    qc = qc_model.data[i]
    get_neg_Hβγ_i!(@view(W[1:p]), qc, Γ, zi, qc_model.τ[1], storages) # exact
    get_neg_Hθγ_i!(@view(W[p+1:end-1]), qc, qc_model.θ, ∇resγ, storages) # exact
    W[end] -= get_Hτγ_i(qc, zi, qc_model.θ, qc_model.τ[1], storages) # exact
    return W
end

function calculate_Ri(qc_model::GaussianCopulaVCModel, i::Int, 
    zi::AbstractVector, Γ, ∇resγ, denom, storage::Storages)
    τ = qc_model.τ[1]
    qc = qc_model.data[i]
    d = qc.n # number of observation for sample i
    storage_d = @view(storage.vec_maxd[1:d])
    storage_d2 = @view(storage.maxd_storage[1:d])

    mul!(storage_d, qc.X, qc_model.β)
    storage_d .= qc.y .- storage_d # storage_d = y - Xb
    mul!(storage_d2, Γ, storage_d) # storage_d2 = Γ * res
    return τ * (dot(zi, storage_d) - dot(zi, storage_d2) / denom)
end

function calculate_Ri(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, 
    zi::AbstractVector, Γ, ∇resγ, denom::T, storage::Storages) where T
    qc = qc_model.data[i]
    res = qc.res # y - μ / std(σ), i.e. standardized residuals
    d = qc.n # number of observation for sample i
    storage_d = @view(storage.vec_maxd[1:d])
    # term1 
    storage_d .= qc.w1 .* (qc.y .- qc.μ)
    R = dot(zi, storage_d)
    # term2
    mul!(storage_d, Γ, res)
    R += dot(∇resγ, storage_d) / denom
    return R
end

function calculate_Qi(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resγ, denom, denom2, storages::Storages)
    qc = qc_model.data[i]
    res = qc.y - qc.X*qc_model.β
    d = qc_model.data[i].n # number of observation for sample i
    τ = qc_model.τ[1]
    storage_d = @view(storages.vec_maxd[1:d])

    # term1
    Qi = τ * sum(abs2, zi)
    # term2
    mul!(storage_d, Γ, zi)
    Qi -= τ * dot(zi, storage_d) / denom
    # term3
    mul!(storage_d, Γ, res)
    Qi += τ^2 * abs2(dot(zi, storage_d)) / denom2 
    return Qi
end

function calculate_Qi(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, 
    i::Int, zi::AbstractVector, Γ::AbstractMatrix{T}, ∇resγ::AbstractVector{T}, 
    denom::T, denom2::T, storages::Storages) where T
    qc = qc_model.data[i]
    res = qc.res
    d = qc.n # number of observation for sample i
    storage_d = @view(storages.vec_maxd[1:d])

    # term1
    storage_d .= qc.w2 .* zi
    Qi = dot(zi, storage_d)
    # term2 
    mul!(storage_d, Γ, ∇resγ)
    Qi += abs2(dot(res, storage_d)) / denom2
    # term3
    Qi -= dot(∇resγ, storage_d) / denom
    # term4
    mul!(storage_d, Γ, res)
    for k in 1:d
        Qi -= storage_d[k] * dβdβ_res_ij(
            qc.d, qc.link, zi[1], qc.η[k], qc.μ[k], qc.varμ[k], res[k]) / denom
    end
    return Qi
end

# `get_∇resγ!` is equivalent to `∇resγ .= get_∇resγ(...)`
function get_∇resγ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector)
    ∇resγ = zeros(eltype(qc_model.data[i].X), qc_model.data[i].n)
    return get_∇resγ!(∇resγ, qc_model, i, zi)
end
function get_∇resγ!(∇resγ, qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector)
    qc = qc_model.data[i]
    d = size(qc.X, 1)
    for k in 1:d # loop over this sample's observations
        ∇resγ[k] = update_∇res_ij(qc.d, zi[k], qc.res[k], qc.μ[k], qc.dμ[k], qc.varμ[k])
    end
    return ∇resγ
end
function get_∇resγ(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector)
    d = size(qc_model.data[i].X, 1)
    T = eltype(qc_model.data[i].X)
    ∇resγ = zeros(T, d)
    return get_∇resγ!(∇resγ, qc_model, i, zi)
end
function get_∇resγ!(∇resγ, qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector)
    qc = qc_model.data[i]
    d = size(qc.X, 1)
    T = eltype(zi)
    sqrtτ = -sqrt(qc_model.τ[1])
    for k in 1:d # loop over this sample's observations
        ∇resγ[k] = sqrtτ * update_∇res_ij(Normal(), zi[k], zero(T), zero(T), zero(T), zero(T))
    end
    return ∇resγ
end

function get_∇resβ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int)
    return qc_model.data[i].∇resβ # d × p
end
function get_∇resβ!(∇resβ, qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int)
    ∇resβ .= qc_model.data[i].∇resβ
end
function get_∇resβ(qc_model::GaussianCopulaVCModel, i::Int)
    ∇resβ = -sqrt(qc_model.τ[1]) .* qc_model.data[i].X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
    return ∇resβ
end
function get_∇resβ!(∇resβ, qc_model::GaussianCopulaVCModel, i::Int)
    ∇resβ .= -sqrt(qc_model.τ[1]) .* qc_model.data[i].X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
    return ∇resβ
end

function get_Pinv(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    return inv(P)
end

# gaussian case needs to handle tau term separately
function get_Pinv(qc_model::GaussianCopulaVCModel)
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    Hβτ = -get_Hτβ(qc_model)
    Hθτ = -get_Hτθ(qc_model)
    Hττ = -get_Hττ(qc_model)
    P = [Hββ  Hβθ  Hβτ;
         Hβθ' Hθθ  Hθτ;
         Hβτ' Hθτ' Hττ]
    return inv(P)
end

function get_Hθθ(qc_model::Union{GaussianCopulaVCModel,GLMCopulaVCModel,NBCopulaVCModel})
    # loglikelihood! function for Hθθ is broken under Gaussian case
    # loglikelihood!(qc_model, true, true)
    # return qc_model.Hθ
    m = length(qc_model.data[1].V) # number of variance components
    hess_math = zeros(m, m)
    for i in eachindex(qc_model.data)
        r = qc_model.data[i].res
        Ω = qc_model.data[i].V
        b = [0.5r' * Ω[k] * r for k in 1:m]
        c = [0.5tr(Ω[k]) for k in 1:m]
        hess_math += b*b' / (1 + qc_model.θ'*b)^2 - c*c' / (1 + qc_model.θ'*c)^2
    end
    return -hess_math
end

function get_Hβθ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    m = length(qc_model.data[1].V)  # number of variance components
    p = size(qc_model.data[1].X, 2) # number of fixed effects
    hess_math = zeros(p, m)
    for i in eachindex(qc_model.data)
        r = qc_model.data[i].res
        Ω = qc_model.data[i].V
        θ = qc_model.θ
        ∇resβ = qc_model.data[i].∇resβ
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end
# must distinguish for gaussian case because it doesn't have ∇resβ precomputed
function get_Hβθ(qc_model::GaussianCopulaVCModel)
    m = length(qc_model.data[1].V)  # number of variance components
    p = size(qc_model.data[1].X, 2) # number of fixed effects
    hess_math = zeros(p, m)
    for (i, qc) in enumerate(qc_model.data)
        r = qc.res
        Ω = qc.V
        θ = qc_model.θ
        ∇resβ = -sqrt(qc_model.τ[1]) .* qc.X # see eq8 of 11.3.1 https://arxiv.org/abs/2205.03505, 2nd term is 0, and first term ∇μ = X
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end

function get_neg_Hθγ_i(gc, θ, ∇resγ, storages::Storages)
    m = length(gc.V)  # number of variance components
    r = gc.res
    Ω = gc.V
    b = [0.5r' * Ω[k] * r for k in 1:m]
    A = hcat([∇resγ' * Ω[k] * r for k in 1:m]...)
    # calculate Hγθ, then return Hθγ = Hγθ'
    Hγθ = (A*θ ./ (1 + θ'*b)^2) * b' - A ./ (1 + θ'*b)
    return Hγθ'
end

# `get_neg_Hθγ_i!` is equivalent to `W .+= get_neg_Hθγ_i`
function get_neg_Hθγ_i!(W, qc, θ, ∇resγ, storages::Storages)
    m = length(qc.V)  # number of variance components
    r = qc.res
    Ω = qc.V
    # compute A and b
    vec_d = @view(storages.vec_maxd[1:qc.n])
    A = storages.m_storage
    b = storages.m_storage2
    for i in 1:m
        mul!(vec_d, Ω[i], r)
        A[i] = dot(∇resγ, vec_d)
        b[i] = 0.5dot(r, vec_d)
    end
    # update W
    denom = 1 + dot(θ, b)
    denom2 = denom^2
    W .+= (dot(A, θ) / denom2) .* b .- A ./ denom
    return W
end

function get_neg_Hθγ_i!(W, qc::GaussianCopulaVCObs, θ, ∇resγ, storages::Storages)
    m = length(qc.V)  # number of variance components
    r = qc.res
    Ω = qc.V
    # compute A and b
    vec_d = @view(storages.vec_maxd[1:qc.n])
    A = storages.m_storage
    b = storages.m_storage2
    for i in 1:m
        mul!(vec_d, Ω[i], r)
        A[i] = dot(∇resγ, vec_d)
        b[i] = 0.5dot(r, vec_d)
    end
    # update W
    denom = 1 + dot(θ, b)
    denom2 = denom^2
    W .-= (dot(A, θ) / denom2) .* b .- A ./ denom
    return W
end

function get_Hβγ_i(qc::GLMCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, storages::Storages) # z is a vector of SNP value (length d)
    res = qc.res
    denom = storages.denom
    denom2 = storages.denom2
    # 1st Hessian term
    Hβγ_i = Transpose(qc.X) * Diagonal(qc.w2) * z
    # 2nd Hessian term
    Hβγ_i += (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    # 3rd Hessian terms
    Hβγ_i -= ∇resβ' * Γ * ∇resγ / denom
    # 4th Hessian term
    ej = zeros(qc.n)
    η = qc.η
    μ = qc.μ
    varμ = qc.varμ
    res = qc.res
    for j in 1:qc.n
        fill!(ej, 0)
        ej[j] = 1
        xj = qc.X[j, :]
        zi = z[1]
        Hβγ_i -= dot(ej, Γ, res) * dγdβresβ_ij(qc.d, qc.link, xj, zi, η[j], μ[j], varμ[j], res[j]) / denom
    end
    return Hβγ_i # p × 1
end

# `get_neg_Hβγ_i!` is equivalent to `W .+= get_Hβγ_i`
function get_neg_Hβγ_i!(W, qc::GLMCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, 
    storages::Storages
    ) # z is a vector of SNP value (length d all storing the same SNP value)
    length(W) == length(storages.vec_p) || error("Dimension mismatch in get_neg_Hβγ_i!")
    # storages
    res = qc.res
    denom = storages.denom[1]
    denom2 = storages.denom2[1]
    vec_d = @view(storages.vec_maxd[1:qc.n])
    p_storage = storages.p_storage
    # 1st Hessian term
    vec_d .= qc.w2 .* z
    mul!(p_storage, Transpose(qc.X), vec_d)
    W .+= p_storage
    # 2nd Hessian term (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    mul!(vec_d, Γ, res)
    c = dot(∇resγ, vec_d) / denom2
    mul!(p_storage, Transpose(∇resβ), vec_d)
    W .+= c .* p_storage
    # 3rd Hessian terms
    mul!(vec_d, Γ, ∇resγ)
    mul!(p_storage, Transpose(∇resβ), vec_d)
    W .-= p_storage ./ denom
    # 4th Hessian term
    mul!(vec_d, Γ, res)
    @inbounds for j in 1:qc.n
        c = vec_d[j] / denom
        dγdβresβ_ij!(W, qc.d, qc.link, @view(qc.X[j, :]), z[1], qc.η[j], 
            qc.μ[j], qc.varμ[j], qc.res[j], c, storages)
    end
    return W # p × 1
end

function get_Hβγ_i(qc::GaussianCopulaVCObs, Γ, z::AbstractVector, β, τ, storages::Storages) # z is a vector of SNP value (length d)
    res = qc.y - qc.X * β
    denom = 1 + 0.5τ * (res' * Γ * res)
    # 1st Hessian term
    Hβγ_i = -τ * Transpose(qc.X) * z
    # 2nd Hessian term
    res = qc.y - (qc.X * β)
    Hβγ_i += τ * Transpose(qc.X) * Γ * z / denom
    # 3rd term
    tmp1 = Transpose(qc.X) * Γ * res
    tmp2 = Transpose(z) * Γ * res
    Hβγ_i -= τ^2 * tmp1 * tmp2' / denom^2
    return Hβγ_i
end

# `get_neg_Hβγ_i!` is equivalent to `W .+= get_Hβγ_i`
function get_neg_Hβγ_i!(W, qc::GaussianCopulaVCObs, Γ::AbstractMatrix{T}, 
    z::AbstractVector{T}, τ::T, storages::Storages
    ) where T # z is a vector of SNP value (length d)
    vec_d = @view(storages.vec_maxd[1:qc.n])
    vec_d2 = @view(storages.maxd_storage[1:qc.n])
    res = qc.res # note: qc.res = (y - X*β) * sqrt(τ)
    mul!(vec_d, Γ, res)
    denom = 1 + 0.5dot(res, vec_d)
    # 2nd Hessian term
    mul!(vec_d2, Γ, z)
    BLAS.gemv!('T', -τ / denom, qc.X, vec_d2, one(T), W)
    # 1st Hessian term
    BLAS.gemv!('T', τ, qc.X, z, one(T), W)
    # 3rd Hessian term
    c = τ * dot(z, vec_d) / denom^2
    BLAS.gemv!('T', c, qc.X, vec_d, one(T), W)
    return W
end

function get_Hββ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)    
    # loop over samples
    for (i, qc) in enumerate(qc_model.data)
        d = qc.n # number of observations for current sample
        # GLM term
        H -= Transpose(qc.X) * Diagonal(qc.w2) * qc.X
        # 2nd term
        res = qc.res # d × 1 standardized residuals
        ∇resβ = qc.∇resβ # d × p
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = qc.η
        μ = qc.μ
        varμ = qc.varμ
        res = qc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = qc.X[j, :]
            dist = typeof(qc_model.d[i]) <: NegativeBinomial ? NegativeBinomial(qc_model.r[1]) : qc_model.d[i]
            link = qc_model.link[i]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
        # @show qc.w2
        # @show res
        # @show η
        # @show μ
        # @show varμ
        # j = 1
        # @show dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])
        # fdsa
    end
    return H
end
# must distinguish for gaussian case because it doesn't have ∇resβ precomputed
function get_Hββ(qc_model::GaussianCopulaVCModel)
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)
    τ = qc_model.τ[1]
    # loop over samples
    for qc in qc_model.data
        d = qc.n
        # GLM term
        H -= τ .* (Transpose(qc.X) * qc.X)
        # 2nd term
        res = qc.y - qc.X * qc_model.β
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5τ * (res' * Γ * res)
        H += τ * Transpose(qc.X) * Γ * qc.X / denom
        # 3rd term
        tmp = Transpose(qc.X) * Γ * res
        H -= τ^2 * tmp * tmp' / denom^2
    end
    return H
end

# needed functions for gaussian case (zi is length d vector of SNP values)
# i stands for sample i
function get_Hτγ_i(qc::GaussianCopulaVCObs, zi::AbstractVector{T}, 
    θ::AbstractVector{T}, τ::T, storages::Storages) where T
    Hτγ = zero(T)
    storage_d = @view(storages.vec_maxd[1:qc.n])
    # compute sqrt(τ)z'Γ(y-Xb) (numerator of 2nd term)
    for k in 1:qc.m
        mul!(storage_d, qc.V[k], qc.res) # storage_d = V[k] * res = V[k] * (y-Xb) * sqrt(τ) 
        Hτγ += θ[k] * dot(zi, storage_d) # Hτγ = sqrt(τ)θ[k]z'V[k](y-Xb) = sqrt(τ)z'Γ(y-Xb)
    end
    qsum = dot(θ, qc.q) # qsum = 0.5r(β)*Γ*r(β) where r = (y-Xb) * sqrt(τ)
    denom = abs2(1 + qsum) # denom = (1 + 0.5τ(y-Xb)'Γ(y-Xb) )^2
    Hτγ /= -denom # Hτγ = -sqrt(τ)z'Γ(y-Xb) / denom
    Hτγ += dot(zi, qc.res) # Hτγ = -sqrt(τ)z'Γ(y-Xb) / denom + z'(y-Xb)sqrt(τ)
    return Hτγ / sqrt(τ)
end
function get_Hτβ(qc_model::GaussianCopulaVCModel)
    β = qc_model.β
    θ = qc_model.θ
    τ = qc_model.τ[1]
    T = eltype(β)
    Hτβ = zeros(T, length(β))
    Hτβstore = zeros(T, length(β))
    for qc in qc_model.data
        fill!(Hτβstore, 0)
        # compute Hτβstore = sqrt(τ)X'Γ(y-Xb)
        for k in 1:qc.m
            mul!(qc.storage_n, qc.V[k], qc.res) # storage_n = V[k] * res = V[k] * (y-Xb) * sqrt(τ) 
            BLAS.gemv!('T', θ[k], qc.X, qc.storage_n, one(T), Hτβstore) # Hτβstore = sqrt(τ)θ[k]X'V[k](y-Xb)
        end
        qsum = dot(θ, qc.q) # qsum = 0.5r(β)*Γ*r(β) where r = (y-Xb) * sqrt(τ)
        denom = abs2(1 + qsum) # denom = (1 + 0.5τ(y-Xb)'Γ(y-Xb) )^2
        BLAS.gemv!('T', one(T), qc.X, qc.res, -inv(denom), Hτβstore) # Hτβstore = -sqrt(τ)X'Γ(y-Xb) / denom + X'(y-Xb)sqrt(τ)
        Hτβstore ./= sqrt(τ)
        Hτβ .+= Hτβstore
    end
    return Hτβ
end
function get_Hτθ(qc_model::GaussianCopulaVCModel)
    θ = qc_model.θ
    τ = qc_model.τ[1]
    m = length(θ)
    T = eltype(θ)
    Hτθ = zeros(T, m)
    qstore = zeros(T, m)
    for qc in qc_model.data
        copyto!(qstore, qc.q) # gc.q = 0.5r(β)*V[k]*r(β) where r = (y-Xb) * sqrt(τ)
        qstore ./= qc_model.τ[1] # qstore = 0.5(y-Xb)*V[k]*(y-Xb)
        tmp_dot = τ * dot(θ, qstore)
        Hτθ .+= inv(1 + tmp_dot) .* qstore .- tmp_dot / abs2(1 + tmp_dot) .* qstore
    end
    return Hτθ
end
function get_Hττ(qc_model::GaussianCopulaVCModel)
    # use loglikelihood! function to get Hττ, which is called in get_Hθθ already
    # loglikelihood!(qc_model, true, true)
    return qc_model.Hτ
end