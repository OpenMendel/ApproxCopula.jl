# source: https://github.com/Hua-Zhou/GLMCopula.jl/blob/master/src/GLMCopula.jl

"""
GaussianCopulaLMMObs
GaussianCopulaLMMObs(y, X, Z)

A realization of Gaussian copula linear mixed model data instance.
"""
struct GaussianCopulaLMMObs{T <: LinearAlgebra.BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Matrix{T}   # gradient wrt Σ 
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    HΣ::Matrix{T}   # Hessian wrt Σ
    res::Vector{T}  # residual vector
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    xtz::Matrix{T}  # Xi'Zi (p-by-q)
    storage_q1::Vector{T}
    storage_q2::Vector{T}
end

function GaussianCopulaLMMObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
    ) where T <: BlasReal
    n, p, q = size(X, 1), size(X, 2), size(Z, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    ztz = transpose(Z) * Z
    xtz = transpose(X) * Z
    storage_q1 = Vector{T}(undef, q)
    storage_q2 = Vector{T}(undef, q)
    # constructor
    GaussianCopulaLMMObs{T}(y, X, Z, 
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ,
        res, xtx, ztz, xtz,
        storage_q1, storage_q2)
end

"""
GaussianCopulaLMMModel
GaussianCopulaLMMModel(gcs)

Gaussian copula linear mixed model, which contains a vector of 
`GaussianCopulaLMMObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaLMMModel{T <: BlasReal} <: MOI.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaLMMObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Matrix{T}    # q-by-q (psd) matrix
    # working arrays
    ΣL::Matrix{T}
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Matrix{T}
    Hβ::Matrix{T}   # Hessian from all observations
    Hτ::Matrix{T}
    HΣ::Matrix{T}
    XtX::Matrix{T}      # X'X = sum_i Xi'Xi
    storage_qq::Matrix{T}
    storage_nq::Matrix{T}
end

function GaussianCopulaLMMModel(gcs::Vector{GaussianCopulaLMMObs{T}}) where T <: BlasReal
    n, p, q = length(gcs), size(gcs[1].X, 2), size(gcs[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Matrix{T}(undef, q, q)
    ΣL  = similar(Σ)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    XtX = zeros(T, p, p) # sum_i xi'xi
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
    end
    storage_qq = Matrix{T}(undef, q, q)
    storage_nq = Matrix{T}(undef, n, q)
    GaussianCopulaLMMModel{T}(gcs, ntotal, p, q, 
        β, τ, Σ, ΣL,
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ, 
        XtX, storage_qq, storage_nq)
end

function loglikelihood!(
    gcm::GaussianCopulaLMMModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇Σ, 0)
    end
    if needhess
        gcm.Hβ .= - gcm.XtX
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇τ .+= gcm.data[i].∇τ
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hτ .+= gcm.data[i].Hτ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    logl
end

function loglikelihood!(
    gc::GaussianCopulaLMMObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    n, p, q = size(gc.X, 1), size(gc.X, 2), size(gc.Z, 2)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇Σ, 0)
    end
    if needhess
        fill!(gc.Hβ, 0)
        fill!(gc.Hτ, 0)
        fill!(gc.HΣ, 0)
    end
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss = abs2(norm(gc.res)) # RSS of standardized residual
    tr = (1//2)dot(gc.ztz, Σ)
    mul!(gc.storage_q1, transpose(gc.Z), gc.res) # storage_q1 = Z' * std residual
    mul!(gc.storage_q2, Σ, gc.storage_q1)        # storage_q2 = Σ * Z' * std residual
    qf = (1//2)dot(gc.storage_q1, gc.storage_q2)
    logl = - (n * log(2π) -  n * log(τ) + rss) / 2 - log(1 + tr) + log(1 + qf)
    # gradient
    if needgrad
        # wrt β
        mul!(gc.∇β, transpose(gc.X), gc.res)
        BLAS.gemv!('N', -inv(1 + qf), gc.xtz, gc.storage_q2, one(T), gc.∇β)
        gc.∇β .*= sqrtτ
        # wrt τ
        gc.∇τ[1] = (n - rss + 2qf / (1 + qf)) / 2τ
        # wrt Σ
        copyto!(gc.∇Σ, gc.ztz)
        BLAS.syrk!('U', 'N', (1//2)inv(1 + qf), gc.storage_q1, (-1//2)inv(1 + tr), gc.∇Σ)
        copytri!(gc.∇Σ, 'U')
    end
    # Hessian: TODO
    if needhess; end;
    # output
    logl
end

function fit!(
    gcm::GaussianCopulaLMMModel{T},
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 5, 
             "tol"                        => 10^-6,
             "max_iter"                   => 1000,
             "accept_after_max_steps"     => 50,
             "warm_start_init_point"      => "yes", 
             "limited_memory_max_history" => 6, # default value
             "hessian_approximation"      => "limited-memory",
            #  "warm_start_init_point"      => "yes",
            #  "mehrotra_algorithm"         => "yes"
            #  "derivative_test"            => "first-order",
             ),
    ) where T <: BlasReal
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    n, p, q = length(gcm.data), size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    par0 = zeros(npar)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # set up NLP optimization problem
    # adapted from: https://github.com/OpenMendel/WiSER.jl/blob/master/src/fit.jl#L56
    # I'm not really sure why this block of code is needed, but not having it
    # would result in objective value staying at 0
    lb = T[]
    ub = T[]
    NLPBlock = MOI.NLPBlockData(
        MOI.NLPBoundsPair.(lb, ub), gcm, true
    )
    MOI.set(solver, MOI.NLPBlock(), NLPBlock)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    # optimize
    MOI.optimize!(solver)
    optstat = MOI.get(solver, MOI.TerminationStatus())
    optstat in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) || 
        @warn("Optimization unsuccesful; got $optstat")

    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MOI.get(solver, MOI.VariablePrimal(), solver_pars))
    loglikelihood!(gcm, true, true)
    gcm
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
    gcm::GaussianCopulaLMMModel, 
    par::Vector)
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    copyto!(gcm.β, 1, par, 1, p)
    gcm.τ[1] = exp(par[p+1])
    fill!(gcm.ΣL, 0)
    offset = p + 2
    for j in 1:q
        gcm.ΣL[j, j] = exp(par[offset])
        offset += 1
        for i in j+1:q
            gcm.ΣL[i, j] = par[offset]
            offset += 1
        end
    end
    mul!(gcm.Σ, gcm.ΣL, transpose(gcm.ΣL))
    nothing
end

"""
    modelpar_to_optimpar!(gcm, par)

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    gcm::GaussianCopulaLMMModel
    )
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    copyto!(par, gcm.β)
    par[p+1] = log(gcm.τ[1])
    Σchol = cholesky(Symmetric(gcm.Σ))
    gcm.ΣL .= Σchol.L
    offset = p + 2
    for j in 1:q
        par[offset] = log(gcm.ΣL[j, j])
        offset += 1
        for i in j+1:q
            par[offset] = gcm.ΣL[i, j]
            offset += 1
        end
    end
    par
end

function MOI.initialize(
    gcm::GaussianCopulaLMMModel, 
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(gcm::GaussianCopulaLMMModel) = [:Grad]

function MOI.eval_objective(
    gcm::GaussianCopulaLMMModel, 
    par::Vector)
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false)
end

function MOI.eval_objective_gradient(
    gcm::GaussianCopulaLMMModel, 
    grad::Vector, 
    par::Vector)
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt log(τ)
    grad[p+1] = gcm.∇τ[1] * gcm.τ[1]
    # gradient wrt L
    mul!(gcm.storage_qq, gcm.∇Σ, gcm.ΣL)
    offset = p + 2
    for j in 1:q
        grad[offset] = 2gcm.storage_qq[j, j] * gcm.ΣL[j, j]
        offset += 1
        for i in j+1:q
            grad[offset] = 2gcm.storage_qq[i, j]
            offset += 1
        end
    end
    nothing
end

"""
init_β(gcm)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    gcm::GaussianCopulaLMMModel{T}
    ) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p) 
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    gcm.τ[1] = gcm.ntotal / rss
    gcm.β
end

"""
update_res!(gc, β)

Update the residual vector according to `β`.
"""
function update_res!(
    gc::GaussianCopulaLMMObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(
    gcm::GaussianCopulaLMMModel{T}
    ) where T <: BlasReal
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::GaussianCopulaLMMObs{T}, 
    σinv::T
    ) where T <: BlasReal
    gc.res .*= σinv
end

function standardize_res!(
    gcm::GaussianCopulaLMMModel{T}
    ) where T <: BlasReal
    σinv = sqrt(gcm.τ[1])
    # standardize residual
    for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i], σinv)
    end
    nothing
end
