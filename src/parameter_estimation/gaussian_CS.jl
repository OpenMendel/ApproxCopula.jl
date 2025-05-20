export GaussianCopulaCSObs, GaussianCopulaCSModel

"""
    GaussianCopulaCSObs
GaussianCopulaCSObs(y, X, V)
A realization of Gaussian copula variance component data.
"""
struct GaussianCopulaCSObs{T <: BlasReal}
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
    ∇τ::Vector{T}   # gradient wrt τ
    ∇ρ::Vector{T}
    ∇σ2::Vector{T}
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    Hρ::Matrix{T}   # Hessian wrt ρ
    Hσ2::Matrix{T}   # Hessian wrt ρ
    Hρσ2::Matrix{T}  # Hessian wrt ρ, σ2
    Hβσ2::Vector{T}  # Hessian wrt β and σ2
    # Hβρ::Vector{T}
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
end

function GaussianCopulaCSObs(
    y::Vector{T},
    X::Matrix{T}
    ) where T <: BlasReal
    n, p = size(X, 1), size(X, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    V = SymmetricToeplitz(ones(T, n))
    vec = Vector{T}(undef, n)
    ∇CSV = SymmetricToeplitz(ones(T, n))
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    # Hβρ = Vector{T}(undef, p)
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
    # constructor
    GaussianCopulaCSObs{T}(n, p, y, X, V, vec, ∇CSV, ∇β, ∇τ, ∇ρ, ∇σ2, Hβ, Hτ, Hρ, Hσ2, Hρσ2, Hβσ2,# Hβρ,
   res, t, q, xtx, storage_n, storage_p1, storage_np, storage_pp, added_term_numerator, added_term2)
end

"""
    GaussianCopulaCSModel
GaussianCopulaCSModel(gcs)
Gaussian copula compound symmetric CS model, which contains a vector of
`GaussianCopulaCSObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaCSModel{T <: BlasReal} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaCSObs{T}}
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
    ∇τ::Vector{T}   # gradient of tau from all observations
    ∇ρ::Vector{T}           # gradient of rho from all observations
    ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian β from all observations
    Hτ::Matrix{T}    # Hessian τ from all observations
    Hρ::Matrix{T}    # Hessian from all observations
    Hσ2::Matrix{T}    # Hessian from all observations
    Hρσ2::Matrix{T}
    Hβσ2::Vector{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    # Hβρ::Vector{T}
    TR::Matrix{T}
    QF::Matrix{T}         # n-by-1 matrix with qik = res_i' Vi res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    penalized::Bool
end

function GaussianCopulaCSModel(gcs::Vector{GaussianCopulaCSObs{T}}; penalized::Bool = false) where T <: BlasReal
    n, p = length(gcs), size(gcs[1].X, 2)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    ρ = [1.0]
    σ2 = [1.0]
    θ   = Vector{T}(undef, 1) # initial MM update for crude estimate of noise
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    Ainv    = zeros(T, p + 3, p + 3)
    Aevec   = zeros(T, p + 3, p + 3)
    M       = zeros(T, p + 3, p + 3)
    vcov    = zeros(T, p + 3, p + 3)
    ψ       = Vector{T}(undef, p + 3)
    # Hβρ = Vector{T}(undef, p)
    TR  = Matrix{T}(undef, n, 1) # collect trace terms
    QF  = Matrix{T}(undef, n, 1)
    Ytotal = 0.0
    ntotal = 0.0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
    end
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, 1)
    storage_θ = Vector{T}(undef, 1)
    GaussianCopulaCSModel{T}(gcs, Ytotal, ntotal, p, β, τ, ρ, σ2, θ,
        ∇β, ∇τ, ∇ρ, ∇σ2, XtX, Hβ, Hτ, Hρ, Hσ2, Hρσ2, Hβσ2, Ainv, Aevec,  M, vcov, ψ,
        TR, QF, storage_n, storage_m, storage_θ, penalized)
end

"""
    initialize_model!(gcm)
Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least
squares solution.
"""
function initialize_model!(
    gcm::GaussianCopulaCSModel{T}
    ) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p)
    @inbounds for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    @show gcm.β
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    println("initializing dispersion using residual sum of squares")
    gcm.τ[1] = gcm.ntotal / rss
    @show gcm.τ
    println("initializing CS noise paramter using MM-algorithm")
    fill!(gcm.ρ, 1.0)
    fill!(gcm.θ, 1.0)
    update_θ!(gcm)
    copyto!(gcm.σ2, gcm.θ)
    nothing
end

"""
    update_res!(gc, β)
Update the residual vector according to `β`.
"""
function update_res!(
    gc::GaussianCopulaCSObs{T},
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(
    gcm::GaussianCopulaCSModel{T}
    ) where T <: BlasReal
    @inbounds for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::GaussianCopulaCSObs{T},
    σinv::T
    ) where T <: BlasReal
    gc.res .*= σinv
end

function standardize_res!(
    gcm::GaussianCopulaCSModel{T}
    ) where T <: BlasReal
    σinv = sqrt(gcm.τ[1])
    # standardize residual
    @inbounds for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i], σinv)
    end
    nothing
end

"""
    update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::GaussianCopulaCSObs)
    gc.q[1] = dot(gc.res, mul!(gc.storage_n, gc.V, gc.res)) / 2
    gc.q
end

function update_θ_jensen!(
    gcm::GaussianCopulaCSModel{T},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where T <: BlasReal
    rsstotal = zero(T)
    @inbounds for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_θ, gcm.θ)
        # update τ
        mul!(gcm.storage_n, gcm.QF, gcm.θ) # gcm.storage_n[i] = q[i]
        gcm.τ[1] = update_τ(gcm.τ[1], gcm.storage_n, gcm.ntotal, rsstotal, 1)
        # numerator in the multiplicative update
        gcm.storage_n .= inv.(inv(gcm.τ[1]) .+ gcm.storage_n) # use newest τ to update θ
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n)
        gcm.θ .*= gcm.storage_m
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.storage_θ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.θ ./= gcm.storage_m
        # monotonicity diagnosis
        verbose && println(sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.θ)) -
            sum(log, 1 .+ gcm.TR * gcm.θ) +
            gcm.ntotal / 2 * (log(gcm.τ[1]) - log(2π)) -
            rsstotal / 2 * gcm.τ[1])
        # convergence check
        gcm.storage_m .= gcm.θ .- gcm.storage_θ
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_θ) + 1) && break
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_θ) + 1)
            verbose && println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.θ
end

# loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(abs(ϕ))), y)

# this gets the loglikelihood from the glm.jl package for the component density
"""
    component_loglikelihood(gc::GaussianCopulaCSObs{T, D, Link}, τ, logl)
Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::GaussianCopulaCSObs{T}, β::Vector{T}, τ::T) where {T <: BlasReal}
  logl = zero(T)
  μ = zeros(Integer(gc.n))
  mul!(μ, gc.X, β)
    @inbounds for j in eachindex(gc.y)
      logl += ApproxCopula.loglik_obs(Normal(), gc.y[j], μ[j], 1.0, inv(τ))
  end
  logl
end

function loglikelihood!(
    gc::GaussianCopulaCSObs{T},
    β::Vector{T},
    τ::T,
    ρ::T,
    σ2::T,
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where {T <: BlasReal}
    # n, p = size(gc.X, 1), size(gc.X, 2)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0.0)
        fill!(gc.∇τ, 0.0)
        fill!(gc.∇σ2, 0.0)
        fill!(gc.∇ρ, 0.0)
        get_∇V!(gc)
    end
    needhess && fill!(gc.Hβ, 0.0)
    sqrtτ = sqrt(abs(τ))
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    get_V!(ρ, gc)

    #evaluate copula loglikelihood
    # mul!(gc.storage_n, gc.V, gc.res) # storage_n = V[k] * res
    mul!(gc.storage_n, gc.V, gc.res, one(T), zero(T)) # storage_n = V[k] * res

    if needgrad
        BLAS.gemv!('T', σ2, gc.X, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    end
    q = dot(gc.res, gc.storage_n)
    tsum = 0.5 * gc.n * σ2
    qsum = 0.5 * σ2 * q
    # @show q
    c1 = 1 + tsum # 1 + tsum
    c2 = 1 + qsum # 1 + qsum
    # loglikelihood
    # logl = ApproxCopula.component_loglikelihood(gc, β, τ) #### make this function for normal
    logl = -log(c1) - (gc.n * log(2π) -  gc.n * log(abs(τ)) + rss) / 2
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
        # mul!(gc.storage_n, gc.∇CSV, gc.res) # storage_n = ∇CSV * res
        mul!(gc.storage_n, gc.∇CSV, gc.res, one(T), zero(T)) # storage_n = ∇ARV * res
        q2 = dot(gc.res, gc.storage_n) #
        gc.∇ρ .= inv(c2) * 0.5 * σ2 * q2
        # gradient with respect to sigma2
        gc.∇σ2 .= -(0.5 * gc.n * inv(c1)) .+ (inv(c2) * 0.5 * q)
        if penalized
            gc.∇σ2 .-= σ2
        end
      if needhess
            gc.Hρ .= - (inv(c2) * 0.5 * σ2 * q2)^2

            # hessian for sigma2
            gc.Hσ2 .= 0.25 * gc.n^2 * inv(c1)^2 - inv(c2)^2 * (0.25 * q^2)

            # hessian cross term for rho and sigma2
            gc.Hρσ2 .= 0.5 * q2 * inv1pq - 0.25 * σ2 * q * q2 * inv1pq^2

            # hessian cross term for beta and sigma2
            gc.Hβσ2 .= inv1pq * gc.∇β - 0.5 * q * inv1pq^2 * σ2 * gc.∇β

            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 0.0, gc.Hβ) # only lower triangular
            gc.Hτ[1, 1] = - abs2(q * inv1pq / τ)
      end
      BLAS.gemv!('T', one(T), gc.X, gc.res, -inv1pq, gc.∇β)
      gc.∇β .*= sqrtτ
      gc.∇τ  .= (gc.n - rss + 2qsum * inv1pq) / 2τ
    end
    logl
end

function loglikelihood!(
    gcm::GaussianCopulaCSModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal}
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇τ, 0.0)
        fill!(gcm.∇ρ, 0.0)
        fill!(gcm.∇σ2, 0.0)
    end
    if needhess
        gcm.Hβ .= - gcm.XtX
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
        fill!(gcm.Hρ, 0.0)
        fill!(gcm.Hσ2, 0.0)
        fill!(gcm.Hρσ2, 0.0)
        fill!(gcm.Hβσ2, 0.0)
        # @show gcm.Hβσ2
        # fill!(gcm.Hβρ, 0)
    end
    logl = zeros(Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
         gcm.τ[1], gcm.ρ[1], gcm.σ2[1], needgrad, needhess; penalized = gcm.penalized)
     end
     @inbounds for i in eachindex(gcm.data)
         if needgrad
             gcm.∇β .+= gcm.data[i].∇β
             gcm.∇τ .+= gcm.data[i].∇τ
             gcm.∇ρ .+= gcm.data[i].∇ρ
             gcm.∇σ2 .+= gcm.data[i].∇σ2
         end
         if needhess
             gcm.Hβ .+= gcm.data[i].Hβ
             gcm.Hτ .+= gcm.data[i].Hτ
             gcm.Hρ .+= gcm.data[i].Hρ
             gcm.Hσ2 .+= gcm.data[i].Hσ2
             gcm.Hρσ2 .+= gcm.data[i].Hρσ2
             gcm.Hβσ2 .+= gcm.data[i].Hβσ2
         end
     end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    return sum(logl)
end
