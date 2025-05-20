mutable struct GLMCopulaVCObs{T <: BlasReal, D, Link}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    n::Int
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇θ::Vector{T}   # gradient wrt θ
    Hβ::Matrix{T}   # Hessian wrt β
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    storage_p1::Vector{T}
    storage_p2::Vector{T}
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

function GLMCopulaVCObs(
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
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    m1        = Vector{T}(undef, m)
    m2        = Vector{T}(undef, m)
    storage_p1 = Vector{T}(undef, p)
    storage_p2 = Vector{T}(undef, p)
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
    GLMCopulaVCObs{T, D, Link}(y, X, V, n, p, m, ∇β, ∇resβ, ∇τ, ∇θ, Hβ, Hθ,
        Hτ, res, t, q, xtx, storage_n, m1, m2, storage_p1, storage_p2, storage_np,
        storage_pp, added_term_numerator, added_term2, η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
GLMCopulaVCModel
GLMCopulaVCModel(gcs)
Gaussian copula variance component model, which contains a vector of
`GLMCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GLMCopulaVCModel{T <: BlasReal, D, Link} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{GLMCopulaVCObs{T, D, Link}}
    Ytotal::T
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
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hθ::Matrix{T}
    Hτ::Matrix{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    TR::Matrix{T}         # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}         # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
    penalized::Bool
end

function GLMCopulaVCModel(gcs::Vector{GLMCopulaVCObs{T, D, Link}}; penalized::Bool = false) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β       = Vector{T}(undef, p)
    τ       = [1.0]
    θ       = Vector{T}(undef, m)
    ∇β      = Vector{T}(undef, p)
    ∇τ      = Vector{T}(undef, 1)
    ∇θ      = Vector{T}(undef, m)
    XtX     = zeros(T, p, p) # sum_i xi'xi
    Hβ      = Matrix{T}(undef, p, p)
    Hθ      = Matrix{T}(undef, m, m)
    Hτ      = Matrix{T}(undef, 1, 1)
    Ainv    = zeros(T, p + m, p + m)
    Aevec   = zeros(T, p + m, p + m)
    M       = zeros(T, p + m, p + m)
    vcov    = zeros(T, p + m, p + m)
    ψ       = Vector{T}(undef, p + m)
    TR      = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal  = 0.0
    ntotal  = 0.0
    d       = Vector{D}(undef, n)
    link    = Vector{Link}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
        d[i] = gcs[i].d
        link[i] = gcs[i].link
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_θ = Vector{T}(undef, m)
    GLMCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, p, m, β, τ, θ,
        ∇β, ∇τ, ∇θ, XtX, Hβ, Hθ, Hτ, Ainv, Aevec, M, vcov, ψ, TR, QF,
        storage_n, storage_m, storage_θ, d, link, penalized)
end

"""
    loglikelihood!(gc, β, τ, θ)
Calculates the loglikelihood of observing `y` given mean `μ`, for the Poisson and Bernoulli base distribution using the GLM.jl package.
"""
function loglikelihood!(
    gc::GLMCopulaVCObs,
    β::Vector{T},
    θ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where {T <: BlasReal}
    # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇θ, 0) # maybe fill Hessian with 0 too??
    end
    needhess && fill!(gc.Hβ, 0)
    update_res!(gc, β)
    # @show gc.res
    standardize_res!(gc)
    # @show gc.res
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # evaluate copula loglikelihood
    @inbounds for k in 1:gc.m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad
            BLAS.gemv!('T', θ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
    end
    # loglikelihood
    logl = QuasiCopula.component_loglikelihood(gc)
    tsum = dot(θ, gc.t)
    logl += -log(1 + tsum)
    qsum  = dot(θ, gc.q) # qsum = 0.5 r'Γr (matches)
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * dot(θ, θ)
    end
    if needgrad
        inv1pq = inv(1 + qsum) # inv1pq = 1 / (1 + 0.5r'Γr)
        inv1pt = inv(1 + tsum) # inv1pt = 1 / (1 + 0.5tr(Γ))
        # gc.∇θ .= inv1pq * gc.q .- inv1pt * gc.t
        gc.m1 .= gc.q
        gc.m1 .*= inv1pq # m1[k] = 0.5 r' * V[k] * r / (1 + 0.5r'Γr)
        gc.m2 .= gc.t
        gc.m2 .*= inv1pt
        gc.∇θ .= gc.m1 .- gc.m2
        if penalized
            gc.∇θ .-= θ
        end
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # gc.Hβ = -[∇resβ*Γ*res][∇resβ*Γ*res]'/(1 + 0.5r'Γr)^2 (gc.∇β stores ∇resβ*Γ*res)
            copytri!(gc.Hβ, 'L') # syrk! filled only lower triangular 
            # does adding this term to the approximation of the hessian violate negative semidefinite properties?
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            @inbounds for k in 1:gc.m
                mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # added_term_numerator = V[k] * ∇resβ
                BLAS.gemm!('T', 'N', θ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2) # added_term2 = ∇resβ' * V[k] * ∇resβ
            end
            gc.added_term2 .*= inv1pq # added_term2 = ∇resβ * V[k] * ∇resβ / (1 + 0.5r'Γr)
            gc.Hβ .+= gc.added_term2
            gc.Hβ .+= QuasiCopula.glm_hessian(gc) # Hβ += -X'*W2*X
            # hessian for vc
            fill!(gc.Hθ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            copytri!(gc.Hθ, 'U')
            # gc.Hθ .= gc.m2 * transpose(gc.m2) - gc.m1 * transpose(gc.m1)
        end
        # set gc.storage_p2 = ∇r'*Γ*r / (1 + 0.5r'Γr) (which is 2nd term of ∇β)
        gc.storage_p2 .= gc.∇β .* inv1pq
        gc.res .= gc.y .- gc.μ
        gc.∇β .= QuasiCopula.glm_gradient(gc)
        gc.∇β .+= gc.storage_p2
        standardize_res!(gc) # ensure that residuals are standardized
    end
    logl
end

function loglikelihood!(
    gcm::GLMCopulaVCModel,
    needgrad::Bool = false,
    needhess::Bool = false
    )
    logl = 0.0
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇θ, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0)
        fill!(gcm.Hθ, 0)
    end
    logl = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
            gcm.θ, needgrad, needhess; penalized = gcm.penalized)
    end
    @inbounds for i in eachindex(gcm.data)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇θ .+= gcm.data[i].∇θ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hθ .+= gcm.data[i].Hθ
        end
    end
    sum(logl)
end
