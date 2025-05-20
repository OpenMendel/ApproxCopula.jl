export NBCopulaVCObs, NBCopulaVCModel
### first structures
mutable struct NBCopulaVCObs{T <: BlasReal, D, Link} # d changes, so must be mutable
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    n::Int
    p::Int
    m::Int # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇θ::Vector{T}   # gradient wrt θ2
    ∇r::Vector{T}   # gradient wrt r (NB)
    Hβ::Matrix{T}   # Hessian wrt β
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    Hr::Matrix{T}   # Hessian wrt r (NB)
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    storage_n2::Vector{T}
    storage_n3::Vector{T}
    storage_p1::Vector{T}
    storage_p2::Vector{T}
    storage_nn::Matrix{T}
    storage_np::Matrix{T}
    storage_pp::Matrix{T}
    added_term_numerator::Matrix{T}
    added_term2::Matrix{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    link::Link      # link function ()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function NBCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    ∇r  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hr  = Matrix{T}(undef, 1, 1)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    storage_n2 = Vector{T}(undef, n)
    storage_n3 = Vector{T}(undef, n)
    storage_p1 = Vector{T}(undef, p)
    storage_p2 = Vector{T}(undef, p)
    storage_nn = Matrix{T}(undef, n, n) # stores Γ = a1*V1 + ... am*Vm
    storage_np = Matrix{T}(undef, n, p)
    storage_pp = Matrix{T}(undef, p, p)
    added_term_numerator = Matrix{T}(undef, n, p)
    added_term2 = Matrix{T}(undef, p, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    wt = Vector{T}(undef, n)
    fill!(wt, one(T))
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    NBCopulaVCObs{T, D, Link}(y, X, V, n, p, m, ∇β, ∇resβ, ∇τ, ∇θ, ∇r, Hβ, Hθ, Hr,
        Hτ, res, t, q, xtx, storage_n, storage_n2, storage_n3, storage_p1,
        storage_p2, storage_nn, storage_np, storage_pp, added_term_numerator, added_term2,
        η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
NBCopulaVCModel
NBCopulaVCModel(gcs)
Negative Binomial copula variance component model, which contains a vector of
`NBCopulaVCObs` as data, model parameters, and working arrays.
"""
struct NBCopulaVCModel{T <: BlasReal, D, Link} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{NBCopulaVCObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    n::Int          # number of random vectors
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    θ::Vector{T}    # m-vector: [θ12, ..., θm2]
    r::Vector{T}    # r parameter for negative binomial
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇θ::Vector{T}
    ∇r::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hθ::Matrix{T}
    Hr::Matrix{T}
    Hτ::Matrix{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    TR::Matrix{T}         # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}         # n-by-m matrix with qik = res_i' Vi[k] res_i
    hess1::Matrix{T}      # holds transpose(gcm.QF) * Diagonal(gcm.storage_n) required for outer product in hessian term 1
    hess2::Matrix{T}      # holds transpose(gcm.TR) * Diagonal(gcm.storage_n2) required for outer product in hessian term 2
    storage_n::Vector{T}
    storage_n2::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
    penalized::Bool
end

function NBCopulaVCModel(gcs::Vector{NBCopulaVCObs{T, D, Link}}; penalized::Bool = false) where {T <: BlasReal, D, Link}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    θ   = Vector{T}(undef, m)
    r   = [1.0]
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    ∇r  = Vector{T}(undef, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hr  = Matrix{T}(undef, 1, 1)
    Hτ  = Matrix{T}(undef, 1, 1)
    Ainv    = zeros(T, p + m + 1, p + m + 1)
    Aevec   = zeros(T, p + m + 1, p + m + 1)
    M       = zeros(T, p + m + 1, p + m + 1)
    vcov    = zeros(T, p + m + 1, p + m + 1)
    ψ       = Vector{T}(undef, p + m + 1)
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0.0
    ntotal = 0.0
    d = Vector{D}(undef, n)
    link = Vector{Link}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
        d[i] = gcs[i].d
        link[i] = gcs[i].link
    end
    QF        = Matrix{T}(undef, n, m)
    hess1     = Matrix{T}(undef, m, n)
    hess2     = Matrix{T}(undef, m, n)
    storage_n = Vector{T}(undef, n)
    storage_n2 = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_θ = Vector{T}(undef, m)
    NBCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, n, p, m, β, τ, θ, r,
        ∇β, ∇τ, ∇θ, ∇r, XtX, Hβ, Hθ, Hr, Hτ, Ainv, Aevec, M, vcov, ψ, TR, QF, hess1, hess2,
        storage_n, storage_n2, storage_m, storage_θ, d, link, penalized)
end

"""
    loglikelihood!(gc::NBCopulaVCObs{T, D, Link}, β, τ, θ, r)
Calculates the loglikelihood of observing `y` given parameters for `β`, `τ = 1`, `θ`, and `r`for Negative Binomial distribution.
"""
function loglikelihood!(
  gc::NBCopulaVCObs{T, D, Link},
  β::Vector{T},
  τ::T, # inverse of linear regression variance
  θ::Vector{T},
  r::T,
  needgrad::Bool = false,
  needhess::Bool = false;
  penalized::Bool = false
  ) where {T <: BlasReal, D, Link}
  # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
  needgrad = needgrad || needhess
  if needgrad
      fill!(gc.∇β, 0)
      fill!(gc.∇τ, 0)
      fill!(gc.∇θ, 0)
  end
  needhess && fill!(gc.Hβ, 0)
  fill!(gc.∇β, 0.0)
  update_res!(gc, β)
  standardize_res!(gc)
  fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
  std_res_differential!(gc) # this will compute ∇resβ

  # evaluate copula loglikelihood
  @inbounds for k in 1:gc.m
      mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
      if needgrad
          BLAS.gemv!('T', θ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual) SHOULDNT THIS BE res^t*Γ*res? NOT ∇resβ*Γ*res
      end
      gc.q[k] = dot(gc.res, gc.storage_n) / 2
  end
  # 2nd term of loglikelihood
  logl = ApproxCopula.component_loglikelihood(gc, r)
  # 1st term of loglikelihood
  tsum = dot(θ, gc.t)
  logl += -log(1 + tsum)
  # 3rd term of loglikelihood
  qsum  = dot(θ, gc.q)
  logl += log(1 + qsum)

  # add L2 ridge penalty
  if penalized
      logl -= 0.5 * dot(θ, θ)
  end
  if needgrad
      inv1pq = inv(1 + qsum)

        # gradient with respect to r
        gc.∇r .= nb_first_derivative(gc, θ, r)
        if penalized
            gc.∇θ .-= θ
        end
      if needhess
          BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
          # does adding this term to the approximation of the hessian violate negative semidefinite properties?
          fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
          fill!(gc.added_term2, 0.0) # fill hessian with 0
          @inbounds for k in 1:gc.m
              mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
              BLAS.gemm!('T', 'N', θ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
          end
          gc.added_term2 .*= inv1pq
          gc.Hβ .+= gc.added_term2
          gc.Hβ .+= ApproxCopula.glm_hessian(gc)

          # hessian for r
          gc.Hr .= nb_second_derivative(gc, θ, r)
      end
      gc.storage_p2 .= gc.∇β .* inv1pq
      gc.res .= gc.y .- gc.μ
      gc.∇β .= ApproxCopula.glm_gradient(gc)
      gc.∇β .+= gc.storage_p2
  end
  logl
end

function loglikelihood!(
    gcm::NBCopulaVCModel{T, D, Link},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D, Link}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇θ, 0.0)
        fill!(gcm.∇r, 0.0)
    end
    if needhess
        fill!(gcm.Hβ, 0.0)
        fill!(gcm.Hθ, 0.0)
        fill!(gcm.Hr, 0.0)
    end
    logl = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
            gcm.τ[1], gcm.θ, gcm.r[1], needgrad, needhess; penalized = gcm.penalized)
    end
    @inbounds for i in eachindex(gcm.data)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇r .+= gcm.data[i].∇r
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hr .+= gcm.data[i].Hr
        end
    end
    if needgrad
        gcm.∇θ .= update_∇θ!(gcm)
    end
    if needhess
        gcm.Hθ .= update_Hθ!(gcm)
    end
    # logl
    return sum(logl)
end

"""
1st derivative of loglikelihood of a sample with θ being the variance components
"""
function nb_first_derivative(gc::NBCopulaVCObs, θ::Vector{T}, r::Number) where T <: BlasReal
    s = zero(T)
    # 2nd term of logl
    @inbounds for j in eachindex(gc.y)
        yi, μi = gc.y[j], gc.μ[j]
        s += -(yi+r)/(μi+r) - log(μi+r) + 1 + log(r) + digamma(r+yi) - digamma(r)
    end
    # 3rd term of logl
    # Γ = θ' * gc.V # Γ = a1*V1 + ... + am*Vm
    # η = gc.η
    # D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
    # dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
    # dresid = -inv(D)*dD*resid
    # s += resid'*Γ*dresid / (1 + 0.5resid'*Γ*resid)
    resid = gc.res # res = inv(D)(y - μ)
    Γ = gc.storage_nn # Γ = a1*V1 + ... + am*Vm
    η = gc.η
    for j in 1:length(η)
        gc.storage_n[j] = inv(sqrt(exp(η[j])*(exp(η[j])+r) / r)) # storage_n = inv(Di) = 1 / sqrt(var(yi))
    end
    for j in 1:length(η)
        # storage_n2 = -inv(D) * d2D
        gc.storage_n2[j] = -gc.storage_n[j] *
            ((exp(3η[j]) / (4r^1.5 * (exp(η[j])*(exp(η[j])+r))^(1.5))) +
            (3exp(2η[j]) / (4r^(2.5)*sqrt(exp(η[j])*(exp(η[j])+r)))))
    end
    for j in 1:length(η)
        # storage_n = inv(D) * dD
        gc.storage_n[j] *= -exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r)))
    end
    for j in 1:length(η)
        # storage_n2 = 2inv(D)*dD*inv(D)*dD -inv(D)*d2D
        gc.storage_n2[j] += 2 * abs2(gc.storage_n[j])
    end
    gc.storage_n .*= -1.0 .* resid # storage_n = dr(β) = derivative of residuals
    gc.storage_n2 .*= resid # storage_n2 = dr²(β) = 2nd derivative of residuals
    mul!(gc.storage_n3, Γ, resid) # storage_n3 = Γ * resid
    denom = 1 + 0.5 * dot(resid, gc.storage_n3)
    mul!(gc.storage_n3, Γ, gc.storage_n) # storage_n3 = Γ * dresid
    term1 = (dot(resid, gc.storage_n3) / denom)^2 # (resid'*Γ*dresid / denom)^2
    term2 = dot(gc.storage_n, gc.storage_n3) / denom # term2 = dresid'*Γ*dresid / denom
    mul!(gc.storage_n3, Γ, gc.storage_n2) # storage_n3 = Γ * d2resid
    term3 = dot(resid, gc.storage_n3) / denom # term3 = resid'*Γ*d2resid / denom
    s += -term1 + term2 + term3
    return s
end

"""
2nd derivative of loglikelihood of a sample with θ being the variance components
"""
function nb_second_derivative(gc::NBCopulaVCObs, θ::Vector{T}, r::Number) where T <: BlasReal
    s = zero(T)
    # 2nd term of logl
    @inbounds for j in eachindex(gc.y)
        yi, μi = gc.y[j], gc.μ[j]
        s += (yi+r)/(μi+r)^2 - 2/(μi+r) + 1/r + trigamma(r+yi) - trigamma(r)
    end
    # 3rd term of logl
    # Γ = θ' * gc.V # Γ = a1*V1 + ... + am*Vm
    # η = gc.η
    # D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
    # dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
    # d2D = Diagonal([(exp(3η[i]) / (4r^1.5 * (exp(η[i])*(exp(η[i])+r))^(1.5))) +
    #     (3exp(2η[i]) / (4r^(2.5)*sqrt(exp(η[i])*(exp(η[i])+r)))) for i in 1:length(η)])
    # resid = gc.res
    # dresid = -inv(D)*dD*resid
    # d2resid = (2inv(D)*dD*inv(D)*dD - inv(D)*d2D)*resid
    # denom = 1 + 0.5resid'*Γ*resid
    # term1 = (resid'*Γ*dresid / denom)^2
    # term2 = dresid'*Γ*dresid / denom
    # term3 = resid'*Γ*d2resid / denom
    # s += -term1 + term2 + term3
    resid = gc.res # res = inv(D)(y - μ)
    Γ = gc.storage_nn # Γ = a1*V1 + ... + am*Vm
    η = gc.η
    for j in 1:length(η)
        gc.storage_n[j] = inv(sqrt(exp(η[j])*(exp(η[j])+r) / r)) # storage_n = inv(Di) = 1 / sqrt(var(yi))
    end
    for j in 1:length(η)
        # storage_n2 = -inv(D) * d2D
        gc.storage_n2[j] = -gc.storage_n[j] *
            ((exp(3η[j]) / (4r^1.5 * (exp(η[j])*(exp(η[j])+r))^(1.5))) +
            (3exp(2η[j]) / (4r^(2.5)*sqrt(exp(η[j])*(exp(η[j])+r)))))
    end
    for j in 1:length(η)
        # storage_n = inv(D) * dD
        gc.storage_n[j] *= -exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r)))
    end
    for j in 1:length(η)
        # storage_n2 = 2inv(D)*dD*inv(D)*dD -inv(D)*d2D
        gc.storage_n2[j] += 2 * abs2(gc.storage_n[j])
    end
    gc.storage_n .*= -1.0 .* resid # storage_n = dr(β) = derivative of residuals
    gc.storage_n2 .*= resid # storage_n2 = dr²(β) = 2nd derivative of residuals
    mul!(gc.storage_n3, Γ, resid) # storage_n3 = Γ * resid
    denom = 1 + 0.5 * dot(resid, gc.storage_n3)
    mul!(gc.storage_n3, Γ, gc.storage_n) # storage_n3 = Γ * dresid
    term1 = (dot(resid, gc.storage_n3) / denom)^2 # (resid'*Γ*dresid / denom)^2
    term2 = dot(gc.storage_n, gc.storage_n3) / denom # term2 = dresid'*Γ*dresid / denom
    mul!(gc.storage_n3, Γ, gc.storage_n2) # storage_n3 = Γ * d2resid
    term3 = dot(resid, gc.storage_n3) / denom # term3 = resid'*Γ*d2resid / denom
    s += -term1 + term2 + term3

    return s
end
