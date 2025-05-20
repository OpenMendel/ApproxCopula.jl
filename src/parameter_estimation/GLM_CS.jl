export GLMCopulaCSObs, GLMCopulaCSModel

# abstract type QCobs end

struct GLMCopulaCSObs{T <: BlasReal, D, Link} # <: QCobs
    # data
    n::Int
    p::Int
    y::Vector{T}
    X::Matrix{T}
    V::SymmetricToeplitz{T}
    vec::Vector{T}
    # working arrays
    ∇CSV::SymmetricToeplitz{T}
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇ρ::Vector{T}
    ∇σ2::Vector{T}
    Hβ::Matrix{T}   # Hessian wrt β
    Hρ::Matrix{T}   # Hessian wrt ρ
    Hσ2::Matrix{T}   # Hessian wrt ρ
    Hρσ2::Matrix{T}  # Hessian wrt ρ, σ2
    Hβσ2::Vector{T}  # Hessian wrt β and σ2
    Hβρ::Vector{T}
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    storage_p1::Vector{T}
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

function GLMCopulaCSObs(
    y::Vector{T},
    X::Matrix{T},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p = size(X, 1), size(X, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    V = SymmetricToeplitz(ones(T, n))
    vec = Vector{T}(undef, n)
    ∇CSV = SymmetricToeplitz(ones(T, n))
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    Hβρ = zeros(T, p)
    res = Vector{T}(undef, n)
    t   = [tr(V)/2]
    q   = Vector{T}(undef, 1)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    storage_p1 = Vector{T}(undef, p)
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
    GLMCopulaCSObs{T, D, Link}(n, p, y, X, V, vec, ∇CSV, ∇β, ∇resβ, ∇ρ, ∇σ2, Hβ, Hρ, Hσ2, Hρσ2, Hβσ2, Hβρ,
       res, t, q, xtx, storage_n, storage_p1, storage_np, storage_pp, added_term_numerator, added_term2,
        η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
    GLMCopulaCSModel
GLMCopulaCSModel(gcs)
Gaussian copula compound symmetric (CS) model, which contains a vector of
`GLMCopulaCSObs` as data, model parameters, and working arrays.
"""
struct GLMCopulaCSModel{T <: BlasReal, D, Link} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{GLMCopulaCSObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    ρ::Vector{T}            # autocorrelation parameter
    σ2::Vector{T}           # autoregressive noise parameter
    θ::Vector{T}
    # working arrays
    ∇β::Vector{T}   # gradient of beta from all observations
    ∇ρ::Vector{T}           # gradient of rho from all observations
    ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hρ::Matrix{T}    # Hessian from all observations
    Hσ2::Matrix{T}    # Hessian from all observations
    Hρσ2::Matrix{T}
    Hβσ2::Vector{T}
    Hβρ::Vector{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    TR::Matrix{T}
    QF::Matrix{T}         # n-by-1 matrix with qik = res_i' Vi res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
    penalized::Bool
end

function GLMCopulaCSModel(gcs::Vector{GLMCopulaCSObs{T, D, Link}}; penalized::Bool = false) where {T <: BlasReal, D, Link}
    n, p = length(gcs), size(gcs[1].X, 2)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    ρ = [1.0]
    σ2 = [1.0]
    θ   = Vector{T}(undef, 1)
    ∇β  = Vector{T}(undef, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    Hβρ = zeros(T, p)
    Ainv    = zeros(T, p + 2, p + 2)
    Aevec   = zeros(T, p + 2, p + 2)
    M       = zeros(T, p + 2, p + 2)
    vcov    = zeros(T, p + 2, p + 2)
    ψ       = Vector{T}(undef, p + 2)
    TR  = Matrix{T}(undef, n, 1) # collect trace terms
    QF  = Matrix{T}(undef, n, 1)
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
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, 1)
    storage_θ = Vector{T}(undef, 1)
    GLMCopulaCSModel{T, D, Link}(gcs, Ytotal, ntotal, p, β, τ, ρ, σ2, θ,
        ∇β, ∇ρ, ∇σ2, XtX, Hβ, Hρ, Hσ2, Hρσ2, Hβσ2, Hβρ, Ainv, Aevec,  M, vcov, ψ,
        TR, QF, storage_n, storage_m, storage_θ, d, link, penalized)
end

# use ToeplitzMatrices
"""
    get_V!(ρ, gc)
Forms the CS covariance structure given ρ (correlation parameter), gc (single cluster observation) object.
"""
function get_V!(ρ, gc::Union{GLMCopulaCSObs{T, D, Link}, NBCopulaCSObs{T, D, Link}, GaussianCopulaCSObs{T}}) where {T, D, Link}
    gc.vec[1] = 1.0
    @inbounds for i in 2:gc.n
        gc.vec[i] = ρ
    end
    gc.V.vc .= gc.vec
end

"""
    get_∇V!(gc)
Forms the first derivative of CS covariance structure wrt to ρ, given ρ (correlation parameter)
"""
function get_∇V!(gc::Union{GLMCopulaCSObs{T, D, Link}, NBCopulaCSObs{T, D, Link}, GaussianCopulaCSObs{T}}) where {T, D, Link}
    gc.vec[1] = 0.0
    @inbounds for i in 2:gc.n
        gc.vec[i] = 1.0
    end
    gc.∇CSV.vc .= gc.vec
end

function loglikelihood!(
    gc::GLMCopulaCSObs,
    β::Vector{T},
    ρ::T,
    σ2::T,
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where {T <: BlasReal}
    # n, p = size(gc.X, 1), size(gc.X, 2)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        get_∇V!(gc)
    end
    needhess && fill!(gc.Hβ, 0)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    standardize_res!(gc)
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # form V
    get_V!(ρ, gc)

    #evaluate copula loglikelihood
    mul!(gc.storage_n, gc.V, gc.res, one(T), zero(T)) # storage_n = V[k] * res
    if needgrad
        BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    end

    q = dot(gc.res, gc.storage_n)
    c1 = 1 + (0.5 * gc.n * σ2)
    # if q < 0
    #     q = 0
    # end
    c2 = 1 + (0.5 * σ2 * q)
    # loglikelihood
    logl = ApproxCopula.component_loglikelihood(gc)
    logl += -log(c1)
    # @show logl
    logl += log(c2)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * (σ2)^2
    end
    # @show logl
    if needgrad
        inv1pq = inv(c2)
        # gradient with respect to rho
        mul!(gc.storage_n, gc.∇CSV, gc.res, one(T), zero(T)) # storage_n = ∇CSV * res
        q2 = dot(gc.res, gc.storage_n) #
        # gc.∇ρ .= inv(c2) * 0.5 * σ2 * transpose(gc.res) * gc.∇CSV * gc.res
        gc.∇ρ .= inv1pq * 0.5 * σ2 * q2
        # gradient with respect to sigma2
        gc.∇σ2 .= 0.5 * ((q * inv1pq) - gc.n * inv(c1))
        if penalized
            gc.∇σ2 .-= σ2
        end
      if needhess
            gc.Hρ .= -(inv(c2) * 0.5 * σ2 * q2)^2

            # hessian for sigma2
            gc.Hσ2 .= 0.25 * gc.n^2 * inv(c1)^2 - inv1pq^2 * (0.25 * q^2)
            if penalized
                gc.Hσ2 .-= 1.0
            end
            # hessian cross term for rho and sigma2
            # gc.Hρσ2 .= 0.5 * q2 * inv1pq - 0.25 * σ2 * q * q2 * inv1pq^2
            gc.Hρσ2 .= (0.5 * q2 * c2 - 0.25 * σ2 * q * q2) * inv1pq^2

            # hessian cross term for beta and sigma2
            # gc.Hβσ2 .= inv1pq * gc.∇β - 0.5 * q * inv1pq^2 * σ2 * gc.∇β
            gc.Hβσ2 .= ((c2 * gc.∇β * inv(σ2)) - (gc.∇β * 0.5 * q)) * inv1pq^2

            #  hessian cross term for beta and rho
            # gc.Hβρ .= inv1pq * σ2 * transpose(gc.∇resβ) * gc.∇ARV * gc.res - 0.5 * σ2^2 * inv1pq^2 * q2 * transpose(gc.∇resβ) * gc.V * gc.res
            BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.storage_p1) # stores ∇resβ*Γ*res (standardized residual)

            # gc.Hβρ .= ((c2 * gc.storage_p1) .- (0.5 * gc.∇β * σ2 * q2)) * inv1pq^2

            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 0.0, gc.Hβ) # only lower triangular
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            # gc.V .= get_AR_cov(n, ρ, σ2, gc.V)
            mul!(gc.added_term_numerator, gc.V, gc.∇resβ, one(T), zero(T)) # storage_n = V[k] * res
            BLAS.gemm!('T', 'N', σ2, gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
            gc.added_term2 .*= inv1pq
            gc.Hβ .+= gc.added_term2
            gc.Hβ .+= ApproxCopula.glm_hessian(gc)
      end
      gc.∇β .= gc.∇β .* inv1pq
      gc.res .= gc.y .- gc.μ
      gc.∇β .+= ApproxCopula.glm_gradient(gc)
      standardize_res!(gc) # ensure that residuals are standardized
    end
    logl
end

function loglikelihood!(
    gcm::GLMCopulaCSModel,
    needgrad::Bool = false,
    needhess::Bool = false
    )
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇ρ, 0.0)
        fill!(gcm.∇σ2, 0.0)
    end
    if needhess
        fill!(gcm.Hβ, 0.0)
        fill!(gcm.Hρ, 0.0)
        fill!(gcm.Hσ2, 0.0)
        fill!(gcm.Hρσ2, 0.0)
        fill!(gcm.Hβσ2, 0.0)
        # @show gcm.Hβσ2
        fill!(gcm.Hβρ, 0)
    end
    logl = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
         gcm.ρ[1], gcm.σ2[1], needgrad, needhess; penalized = gcm.penalized)
     end
     @inbounds for i in eachindex(gcm.data)
         if needgrad
             gcm.∇β .+= gcm.data[i].∇β
             gcm.∇ρ .+= gcm.data[i].∇ρ
             gcm.∇σ2 .+= gcm.data[i].∇σ2
         end
         if needhess
             gcm.Hβ .+= gcm.data[i].Hβ
             gcm.Hρ .+= gcm.data[i].Hρ
             gcm.Hσ2 .+= gcm.data[i].Hσ2
             gcm.Hρσ2 .+= gcm.data[i].Hρσ2
             gcm.Hβσ2 .+= gcm.data[i].Hβσ2
         end
     end
    return sum(logl)
end
