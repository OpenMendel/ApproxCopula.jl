export GaussianCopulaVCObs, GaussianCopulaVCModel
export fitted

"""
GaussianCopulaVCObs
GaussianCopulaVCObs(y, X, V)
A realization of Gaussian copula variance component data.
"""
struct GaussianCopulaVCObs{T <: BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    n::Int
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇θ::Vector{T}   # gradient wrt θ
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2    (res is scaled by tau)
    storage_n::Vector{T}
    storage_p::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    Hθ::Matrix{T}   # Hessian wrt variance components θ
end

function GaussianCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}}
    ) where T <: BlasReal
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    m1        = Vector{T}(undef, m)
    m2        = Vector{T}(undef, m)
    Hθ  = Matrix{T}(undef, m, m)
    # constructor
    GaussianCopulaVCObs{T}(y, X, V, n, p, m, ∇β, ∇τ, ∇θ, Hβ,
        Hτ, res, xtx, t, q, storage_n, storage_p, m1, m2, Hθ)
end

"""
GaussianCopulaVCModel
GaussianCopulaVCModel(gcs)
Gaussian copula variance component model, which contains a vector of
`GaussianCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaVCModel{T <: BlasReal} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaVCObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    θ::Vector{T}    # m-vector: [θ12, ..., θm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇θ::Vector{T}
    Hβ::Matrix{T}    # Hessian from all observations
    Hτ::Matrix{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    # hessain with resp to vc
    # asymptotic covariance for inference
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    penalized::Bool
end

function GaussianCopulaVCModel(gcs::Vector{GaussianCopulaVCObs{T}}; penalized::Bool = false) where T <: BlasReal
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    θ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_θ = Vector{T}(undef, m)
    Ainv    = zeros(T, p + m + 1, p + m + 1)
    Aevec   = zeros(T, p + m + 1, p + m + 1)
    M       = zeros(T, p + m + 1, p + m + 1)
    vcov    = zeros(T, p + m + 1, p + m + 1)
    ψ       = Vector{T}(undef, p + m + 1)
    Hθ  = Matrix{T}(undef, m, m)
    GaussianCopulaVCModel{T}(gcs, ntotal, p, m, β, τ, θ,
        ∇β, ∇τ, ∇θ, Hβ, Hτ, XtX, TR, QF,
        storage_n, storage_m, storage_θ, Ainv, Aevec, M, vcov, ψ, Hθ, penalized)
end

function fitted(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T,
    θ::Vector{T}) where T <: BlasReal
    n, m = length(gc.y), length(gc.V)
    μ̂ = gc.X * β
    Ω = Matrix{T}(undef, n, n)
    @inbounds for k in 1:m
        Ω .+= θ[k] .* gc.V[k]
    end
    θ02 = inv(τ)
    c = inv(1 + dot(θ, gc.t)) # normalizing constant
    V̂ = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:j-1
            V̂[i, j] = c * σ02 * Ω[i, j]
        end
        V̂[j, j] = c * σ02 * (1 + Ω[j, j] + tr(Ω) / 2)
    end
    LinearAlgebra.copytri!(V̂, 'U')
    μ̂, V̂
end

function loglikelihood!(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    θ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where T <: BlasReal
    # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇θ, 0)
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(abs(τ)) # ben: why is there abs here?
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ) # res = (y - Xb) * sqrt(τ)
    rss  = abs2(norm(gc.res)) # RSS of residual
    tsum = dot(abs.(θ), gc.t) # ben: why is there abs here?
    logl = - log(1 + tsum) - (gc.n * log(2π) -  gc.n * log(abs(τ)) + rss) / 2
    @inbounds for k in 1:gc.m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res = V[k] * (y-Xb) * sqrt(τ) 
        if needgrad # ∇β stores X'*Γ*res (standardized residual)
            BLAS.gemv!('T', θ[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q = 0.5 res' * V[k] * res = 0.5τ(y-Xb)^t*V*(y-Xb)
    end
    qsum  = dot(abs.(θ), gc.q) # qsum = 0.5r(β)*Γ*r(β) where r = (y-Xb) * sqrt(τ)
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * dot(θ, θ)
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
            gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
            # # hessian of vc vector use with fit_newton_normal.jl
            inv1pt = inv(1 + tsum)
            gc.m1 .= gc.q
            gc.m1 .*= inv1pq
            gc.m2 .= gc.t
            gc.m2 .*= inv1pt
            # hessian for vc
            fill!(gc.Hθ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            copytri!(gc.Hθ, 'U')
        end
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv1pq, gc.∇β) # ∇β = -sqrt(τ)X'Γ(y-Xb) / (1+τ(y-Xb)'Γ(y-Xb)) + sqrt(τ)X'(y-Xb)
        gc.∇β .*= sqrtτ
        gc.∇τ  .= (gc.n - rss + 2qsum * inv1pq) / 2τ # note: rss and qsum have τ in it
        gc.∇θ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
        if penalized
            gc.∇θ .-= θ
        end
    end
    # output
    logl
end

function loglikelihood!(
    gcm::GaussianCopulaVCModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇θ, 0)
    end
    if needhess
        gcm.Hβ .= - gcm.XtX
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    logl = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
         gcm.τ[1], gcm.θ, needgrad, needhess; penalized = gcm.penalized)
     end
     @inbounds for i in eachindex(gcm.data)
         if needgrad
             gcm.∇β .+= gcm.data[i].∇β
             gcm.∇τ .+= gcm.data[i].∇τ
             gcm.∇θ .+= gcm.data[i].∇θ
         end
         if needhess
             gcm.Hβ .+= gcm.data[i].Hβ
             gcm.Hτ .+= gcm.data[i].Hτ
             gcm.Hθ .+= gcm.data[i].Hθ
         end
     end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    return sum(logl)
end

# uncomment this and exclude fit_gaussian_vc.jl to fit variance components separately using MM-algorithm instead of Joint Newton.
# function fit!(
#     gcm::GaussianCopulaVCModel,
#     solver=Ipopt.IpoptSolver(print_level=0)
#     )
#     initialize_model!(gcm)
#     optm = MathProgBase.NonlinearModel(solver)
#     lb = fill(-Inf, gcm.p)
#     ub = fill( Inf, gcm.p)
#     MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
#     MathProgBase.setwarmstart!(optm, gcm.β)
#     MathProgBase.optimize!(optm)
#     optstat = MathProgBase.status(optm)
#     optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
#     copy_par!(gcm, MathProgBase.getsolution(optm))
#     loglikelihood!(gcm)
#     gcm
# end

# function MathProgBase.initialize(
#     gcm::GaussianCopulaVCModel,
#     requested_features::Vector{Symbol})
#     for feat in requested_features
#         if !(feat in [:Grad, :Hess])
#             error("Unsupported feature $feat")
#         end
#     end
# end

# MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad, :Hess]

# function MathProgBase.eval_f(
#     gcm::GaussianCopulaVCModel,
#     par::Vector)
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_Σ!(gcm)
#     # evaluate loglikelihood
#     loglikelihood!(gcm, false, false)
# end

# function MathProgBase.eval_grad_f(
#     gcm::GaussianCopulaVCModel,
#     grad::Vector,
#     par::Vector)
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_θ!(gcm)
#     # evaluate gradient
#     logl = loglikelihood!(gcm, true, false)
#     copyto!(grad, gcm.∇β)
#     nothing
# end

# function copy_par!(
#     gcm::GaussianCopulaVCModel,
#     par::Vector)
#     copyto!(gcm.β, par)
#     par
# end

# function MathProgBase.hesslag_structure(gcm::GaussianCopulaVCModel)
#     Iidx = Vector{Int}(undef, (gcm.p * (gcm.p + 1)) >> 1)
#     Jidx = similar(Iidx)
#     ct = 1
#     for j in 1:gcm.p
#         for i in j:gcm.p
#             Iidx[ct] = i
#             Jidx[ct] = j
#             ct += 1
#         end
#     end
#     Iidx, Jidx
# end

# function MathProgBase.eval_hesslag(
#     gcm::GaussianCopulaVCModel{T},
#     H::Vector{T},
#     par::Vector{T},
#     σ::T,
#     μ::Vector{T}) where T <: BlasReal
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_θ!(gcm)
#     # evaluate Hessian
#     loglikelihood!(gcm, true, true)
#     # copy Hessian elements into H
#     ct = 1
#     for j in 1:gcm.p
#         for i in j:gcm.p
#             H[ct] = gcm.Hβ[i, j]
#             ct += 1
#         end
#     end
#     H .*= σ
# end
