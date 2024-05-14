# struct holding intermediate arrays for a given sample
struct MultivariateCopulaObs{T <: BlasReal}
    η::Vector{T} # η = B'x (linear predictor value of current sample)
    μ::Vector{T} # μ = linkinv(link, η) (mean of current sample)
    res::Vector{T} # res[i] = yᵢ - μᵢ (residual)
    std_res::Vector{T} # std_res[i] = (yᵢ - μᵢ) / σᵢ (standardized residual)
    dμ::Vector{T} # intermediate GLM quantity
    varμ::Vector{T} # intermediate GLM quantity
    w1::Vector{T} # intermediate GLM quantity
    w2::Vector{T} # intermediate GLM quantity
    q::Vector{T} # q[k] = res_i' * V_i[k] * res_i / 2 (this is variable b in VC model, see sec 6.2 of QuasiCopula paper)
    ∇resβ::Matrix{T} # gradient of standardized residual with respect to beta
    ∇vecB::Vector{T} # gradient of loglikelihood wrt β = vec(B)
    ∇θ::Vector{T}   # gradient of loglikelihood wrt θ (variance components)
    ∇ϕ::Vector{T}   # gradient of loglikelihood wrt ϕ (nuisnace parameters)
    storage_d::Vector{T}
end

function MultivariateCopulaObs(T, d, p, m, s)
    η = zeros(T, d)
    μ = zeros(T, d)
    res = zeros(T, d)
    std_res = zeros(T, d)
    dμ = zeros(T, d)
    varμ = zeros(T, d)
    w1 = zeros(T, d)
    w2 = zeros(T, d)
    q = zeros(T, m) # q is variable b in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2
    ∇resβ = zeros(T, d * p, d)
    ∇vecB = zeros(T, d * p)
    ∇θ = zeros(T, m)
    ∇ϕ = zeros(T, s)
    storage_d = zeros(T, d)
    obs = MultivariateCopulaObs(
        η, μ, res, std_res, dμ, varμ, w1, w2, q, ∇resβ, ∇vecB, ∇θ, ∇ϕ, storage_d
    )
    return obs
end

struct MultivariateCopulaModel{T <: BlasReal} <: MOI.AbstractNLPEvaluator
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    vecdist::Vector{<:UnivariateDistribution} # length d vector of marginal distributions for each phenotype
    veclink::Vector{<:Link} # length d vector of link functions for each phenotype's marginal distribution
    data::Vector{MultivariateCopulaObs{T}}
    # data dimension
    n::Int # sample size
    d::Int # number of phenotypes per sample
    p::Int # number of (non-genetic) covariates per sample
    s::Int # number of nuisance parameters 
    m::Int # number of parameters in cholesky matrix L
    # parameters
    B::Matrix{T}    # p × d matrix of mean regression coefficients, Y = XB
    L::Cholesky{Float64} # cholesky factor of Γ: L.L*L.L' = Γ
    ϕ::Vector{T}    # s-vector of nuisance parameters
    nuisance_idx::Vector{Int} # indices that are nuisance parameters, indexing into vecdist
    # working arrays
    vechL::Vector{T}     # vechL = vech(L.L) where vech() computes the lower-triangular part of L
    ∇vechL::Vector{T}    # gradient of vech(B)
    ∇vecB::Vector{T}     # length pd vector, its the gradient of vec(B) 
    ∇ϕ::Vector{T}        # length s vector, gradient of nuisance parameters
    penalized::Bool
end

function MultivariateCopulaModel(
    Y::Matrix{T}, # n × d
    X::Matrix{T}, # n × p
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each phenotype
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    penalized = false,
    supported_nuisance_dist = [Normal, NegativeBinomial]
    ) where T <: BlasReal
    n, d = size(Y)
    p = size(X, 2)
    m = (d * (d + 1)) >> 1
    # check for errors
    n == size(X, 1) || error("Number of samples in Y and X mismatch")
    nuisance_idx = findall(x -> 
        x ∈ supported_nuisance_dist || typeof(x) ∈ supported_nuisance_dist, vecdist)
    s = length(nuisance_idx)
    any(x -> typeof(x) <: NegativeBinomial, vecdist) && 
        error("Negative binomial base not supported yet")
    # initialize variables
    B = zeros(T, p, d)
    ϕ = fill(one(T), s)
    ∇vecB = zeros(T, p*d)
    ∇ϕ = zeros(T, s)
    data = MultivariateCopulaObs{T}[]
    for _ in 1:n
        push!(data, MultivariateCopulaObs(T, d, p, m, s))
    end
    # covariance matrix
    Γ = cov(Y)
    L = cholesky(Symmetric(Γ, :L)) # use lower triangular part of Γ
    vechL = vech(L.L)
    ∇vechL = zeros(m)
    # change type of variables to match struct
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    return MultivariateCopulaModel(
        Y, X, vecdist, veclink, data,
        n, d, p, s, m, 
        B, L, ϕ, nuisance_idx, 
        vechL, ∇vechL, ∇vecB, ∇ϕ, 
        penalized
    )
end

function Base.copyto!(L::Cholesky, C::Cholesky)
    size(L) == size(C) || error("L and C has different dimensions")
    L.factors .= C.factors
end

"""
    vech!(v::AbstractVector, A::AbstractVecOrMat)
    vech!(v::AbstractVector, A::Cholesky)

Overwrite vector `v` by the entries from lower triangular part of `A`. 
Source = https://github.com/OpenMendel/WiSER.jl/blob/77e723b4769eb54f9eaa72aab038b4b5366365cd/src/multivariate_calculus.jl#L2
"""
function vech!(v::AbstractVector, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    idx = 1
    @inbounds for j in 1:n, i in j:m
        v[idx] = A[i, j]
        idx += 1
    end
    v
end
function vech!(v::AbstractVector, L::Cholesky)
    Ldata = L.factors
    if L.uplo === 'L'
        vech!(v, Ldata)
    else
        error("L.uplo !== 'L'! Construct cholesky factors using cholesky(x, :L)")
    end
    return v
end

"""
    un_vech!(A::AbstractMatrix, v::AbstractVector)
    un_vech!(A::Cholesky, v::AbstractVector)

Overwrite lower triangular part of `A` by the entries from `v`. Upper triangular
part of `A` is untouched.  
"""
function un_vech!(A::AbstractMatrix, v::AbstractVector)
    m, n = size(A, 1), size(A, 2)
    idx = 1
    @inbounds for j in 1:n, i in j:m
        A[i, j] = v[idx]
        idx += 1
    end
    A
end
function un_vech!(L::Cholesky, v::AbstractVector)
    un_vech!(L.factors, v)
end

"""
    vech(A::AbstractVecOrMat) -> AbstractVector

Return the entries from lower triangular part of `A` as a vector.
Source = https://github.com/OpenMendel/WiSER.jl/blob/77e723b4769eb54f9eaa72aab038b4b5366365cd/src/multivariate_calculus.jl#L2
"""
function vech(A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    vech!(similar(A, n * m - (n * (n - 1)) >> 1), A)
end

"""
    fit!(qc_model::MultivariateCopulaModel, solver=Ipopt.IpoptSolver)

Fit an `MultivariateCopulaModel` object by MLE using a nonlinear programming
solver. Start point should be provided in `qc_model.β`, `qc_model.`, `qc_model.ϕ`

# Arguments
- `qc_model`: A `MultivariateCopulaModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton
    iterations with convergence tolerance 10^-6. (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    qc_model::MultivariateCopulaModel,
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 5, 
             "tol"                        => 10^-3,
             "max_iter"                   => 100,
             "accept_after_max_steps"     => 50,
             "warm_start_init_point"      => "yes", 
             "limited_memory_max_history" => 6, # default value
             "hessian_approximation"      => "limited-memory",
            #  "derivative_test"            => "first-order",
             ),
    )
    T = eltype(qc_model.X)
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")
    
    # Pass options to solver
    config_solver(solver, solver_config)

    # initialize conditions
    p, d, m, s = qc_model.p, qc_model.d, qc_model.m, qc_model.s
    initialize_model!(qc_model)
    npar = p * d + m + s # pd fixed effects, m covariance params, s nuisance params
    npar ≥ 0.2qc_model.n && 
        error("Estimating $npar params with $(size(Y, 1)) samples, not recommended")
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, qc_model)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints (nuisance parameters must be >0)
    for k in p*d+m+1:npar
        solver.variables.lower[k] = 0
    end

    # set up NLP optimization problem
    # adapted from: https://github.com/OpenMendel/WiSER.jl/blob/master/src/fit.jl#L56
    # I'm not really sure why this block of code is needed, but not having it
    # would result in objective value staying at 0
    lb = T[]
    ub = T[]
    NLPBlock = MOI.NLPBlockData(
        MOI.NLPBoundsPair.(lb, ub), qc_model, true
    )
    MOI.set(solver, MOI.NLPBlock(), NLPBlock)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    # optimize
    MOI.optimize!(solver)
    optstat = MOI.get(solver, MOI.TerminationStatus())
    optstat in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) || 
        @warn("Optimization unsuccesful; got $optstat")

    # update parameters and refresh gradient
    optimpar_to_modelpar!(qc_model, MOI.get(solver, MOI.VariablePrimal(), solver_pars))
    return loglikelihood!(qc_model, true, false)
end

"""
    modelpar_to_optimpar!(par, qc_model)

Translate model parameters in `qc_model` to optimization variables in `par`
"""
function modelpar_to_optimpar!(
    par :: Vector,
    qc_model :: MultivariateCopulaModel
    )
    # β
    copyto!(par, qc_model.B)
    # variance params
    var_range = qc_model.p * qc_model.d + 1:qc_model.p * qc_model.d + qc_model.m
    vech!(@view(par[var_range]), qc_model.L)
    # nuisance params
    offset = qc_model.p * qc_model.d + qc_model.m + 1
    par[offset:end] .= qc_model.ϕ
    return par
end

"""
    optimpar_to_modelpar_quasi!(qc_model, par)

Translate optimization variables in `par` to the model parameters in `qc_model`.
"""
function optimpar_to_modelpar!(
    qc_model :: MultivariateCopulaModel,
    par :: Vector
    )
    # β
    copyto!(qc_model.B, 1, par, 1, qc_model.p * qc_model.d)
    # Γ
    var_range = qc_model.p * qc_model.d + 1:qc_model.p * qc_model.d + qc_model.m
    qc_model.vechL .= @view(par[var_range])
    un_vech!(qc_model.L, @view(par[var_range]))
    # nuisance parameters
    offset = qc_model.p * qc_model.d + qc_model.m + 1
    qc_model.ϕ .= par[offset:end]
    return qc_model
end

function MOI.initialize(
    qc_model::MultivariateCopulaModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(qc_model))
            error("Unsupported feature $feat, requested = $requested_features")
        end
    end
end

MOI.features_available(qc_model::MultivariateCopulaModel) = [:Grad]

function MOI.eval_objective(
    qc_model :: MultivariateCopulaModel,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    return loglikelihood!(qc_model, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
    qc_model :: MultivariateCopulaModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    obj = loglikelihood!(qc_model, true, false)
    # gradient wrt β
    copyto!(grad, qc_model.∇vecB)
    # gradient wrt vech(L)
    var_range = qc_model.p * qc_model.d + 1:qc_model.p * qc_model.d + qc_model.m
    grad[var_range] .= qc_model.∇vechL
    # gradient wrt to nuisance parameters
    offset = qc_model.p * qc_model.d + qc_model.m + 1
    grad[offset:end] .= qc_model.∇ϕ
    return obj
end

"""
    initialize_model!(qc_model)

Initializes mean parameters B with univariate regression values (we fit a GLM
to each y separately). 
"""
function initialize_model!(qc_model::MultivariateCopulaModel)
    # univariate GLMs
    for (j, y) in enumerate(eachcol(qc_model.Y))
        fit_glm = glm(qc_model.X, y, qc_model.vecdist[j], qc_model.veclink[j])
        qc_model.B[:, j] .= fit_glm.pp.beta0
    end
    # covariance 
    Γ = cov(qc_model.Y)
    L = cholesky(Symmetric(Γ, :L)) # use lower triangular part of Γ
    copyto!(qc_model.L, L)
    vech!(qc_model.vechL, LowerTriangular(L.factors))
    # nuisance parameters
    fill!(qc_model.ϕ, 1)
    return nothing
end

function loglikelihood!(
    qc_model::MultivariateCopulaModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    if needgrad
        fill!(qc_model.∇vecB, 0)
        fill!(qc_model.∇vechL, 0)
        fill!(qc_model.∇ϕ, 0)
    end
    needhess && error("Hessian not implemented for MultivariateCopulaModel!")
    logl = zero(T)
    for i in 1:qc_model.n
        logl += loglikelihood!(qc_model, i, needgrad, needhess)
        if needgrad
            qc_model.∇vecB .+= qc_model.data[i].∇vecB
            qc_model.∇vechL .+= qc_model.data[i].∇vechL
            qc_model.∇ϕ .+= qc_model.data[i].∇ϕ
        end
    end
    return logl
end

# evaluates the loglikelihood for sample i
# function loglikelihood!(
#     qc_model::MultivariateCopulaModel{T},
#     i::Int,
#     needgrad::Bool = false,
#     needhess::Bool = false;
#     ) where T <: BlasReal
#     d = qc_model.d        # number of phenotypes
#     p = qc_model.p        # number of covarites
#     θ = qc_model.θ        # variance components
#     qc = qc_model.data[i] # sample i's storage
#     if needgrad
#         fill!(qc.∇vecB, 0)
#         fill!(qc.∇θ, 0)
#         fill!(qc.∇ϕ, 0)
#     end
#     # update residuals and its gradient
#     update_res!(qc_model, i)
#     std_res_differential!(qc_model, i) # compute ∇resβ
#     # loglikelihood term 2 i.e. sum sum ln(f_ij | β)
#     logl = QuasiCopula.component_loglikelihood(qc_model, i)
#     # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
#     tsum = dot(θ, qc_model.t) # tsum = 0.5tr(Γ)
#     logl += -log(1 + tsum)
#     # compute ∇resβ*Γ*res and variable b for variance component model
#     @inbounds for k in 1:qc_model.m # loop over m variance components
#         mul!(qc.storage_d, qc_model.V[k], qc.std_res) # storage_d = V[k] * r
#         if needgrad
#             BLAS.gemv!('N', θ[k], qc.∇resβ, qc.storage_d, one(T), qc.∇vecB) # ∇β = ∇r*Γ*r
#         end
#         qc.q[k] = dot(qc.std_res, qc.storage_d) / 2 # q[k] = 0.5 r * V[k] * r
#     end
#     # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
#     qsum = dot(θ, qc.q) # qsum = 0.5 r*Γ*r
#     logl += log(1 + qsum)
#     # add L2 ridge penalty
#     if qc_model.penalized
#         logl -= 0.5 * dot(θ, θ)
#     end
#     # gradient
#     if needgrad
#         inv1pq = inv(1 + qsum) # inv1pq = 1 / (1 + 0.5r'Γr)
#         if needhess
#             error("Hessian not implemented for MultivariateCopulaModel!")
#         end
#         # compute X'*Diagonal(dg/varμ)*(y-μ) + ∇r'Γr/(1+0.5r'Γr)  (gradient of logl wrt vecB)
#         xi = @view(qc_model.X[i, :])
#         for j in 1:d
#             out = @view(qc.∇vecB[(j-1)*p+1:j*p])
#             out .*= inv1pq
#             out .+= j in qc_model.nuisance_idx ? xi .* qc.w1[j] * qc.std_res[j] :
#                 xi .* qc.w1[j] * qc.res[j]
#             # BLAS.gemv!('T', one(T), xi, qc.storage_d, inv1pq, out)
#         end
#         # Gaussian case: compute ∇τ and undo scaling by τ (vecB used std_res which includes extra factor of √τ)
#         for (j, idx) in enumerate(qc_model.nuisance_idx)
#             τ = abs(qc_model.ϕ[j])
#             vecB_range = (idx-1)*p+1:idx*p
#             qc.∇vecB[vecB_range] .*= sqrt(τ)
#             rss = abs2(qc.std_res[idx]) # std_res contains a factor of sqrt(τ)
#             qc.∇ϕ[j] = (1 - rss + 2qsum * inv1pq) / 2τ # this is kind of wrong by autodiff but I can't figure out why
#         end
#         qc.∇θ .= inv1pq .* qc.q .- inv(1 + tsum) .* qc_model.t
#         if qc_model.penalized
#             qc.∇θ .-= θ
#         end
#     end
#     # output
#     return logl
# end

function update_res!(qc_model::MultivariateCopulaModel, i::Int)
    # data for sample i
    xi = @view(qc_model.X[i, :])
    yi = @view(qc_model.Y[i, :])
    nuisance_counter = 1
    vecdist = qc_model.vecdist
    veclink = qc_model.veclink
    obs = qc_model.data[i]
    mul!(obs.η, qc_model.B', xi)
    @inbounds for j in eachindex(yi)
        obs.μ[j] = GLM.linkinv(veclink[j], obs.η[j])
        obs.varμ[j] = GLM.glmvar(vecdist[j], obs.μ[j]) # Note: for negative binomial, d.r is used
        obs.dμ[j] = GLM.mueta(veclink[j], obs.η[j])
        obs.w1[j] = obs.dμ[j] / obs.varμ[j]
        obs.w2[j] = obs.w1[j] * obs.dμ[j]
        obs.res[j] = yi[j] - obs.μ[j]
        if typeof(vecdist[j]) <: Normal
            τ = abs(qc_model.ϕ[nuisance_counter])
            obs.std_res[j] = obs.res[j] * sqrt(τ)
            nuisance_counter += 1
        else
            obs.std_res[j] = obs.res[j] / sqrt(obs.varμ[j])
        end
    end
    return nothing
end

# """
# qc_model.data[i].∇resβ is dp × d matrix that stores ∇rᵢ(β), i.e. gradient of sample i's
# residuals with respect to the dp × 1 vector β. Each of the d columns of ∇rᵢ(β) 
# stores ∇rᵢⱼ(β), a length dp vector.
# """
# function std_res_differential!(qc_model::MultivariateCopulaModel, i::Int)
#     obs = qc_model.data[i]
#     d = qc_model.d
#     p = qc_model.p # p = number of covariates, d = number of phenotypes
#     xi = @view(qc_model.X[i, :])
#     @inbounds for j in 1:d # loop over columns
#         for k in 1:p
#             obs.∇resβ[(j-1)*p + k, j] = update_∇res_ij(qc_model.vecdist[j], xi[k], 
#                 obs.std_res[j], obs.μ[j], obs.dμ[j], obs.varμ[j])
#         end
#     end
#     return nothing
# end

function component_loglikelihood(qc_model::MultivariateCopulaModel{T}, i::Int) where T <: BlasReal
    y = @view(qc_model.Y[i, :])
    obs = qc_model.data[i]
    nuisance_counter = 1
    logl = zero(T)
    @inbounds for j in eachindex(y)
        dist = qc_model.vecdist[j]
        if typeof(dist) <: Normal
            τ = inv(qc_model.ϕ[nuisance_counter])
            logl += QuasiCopula.loglik_obs(dist, y[j], obs.μ[j], one(T), τ)
            nuisance_counter += 1
        else
            logl += QuasiCopula.loglik_obs(dist, y[j], obs.μ[j], one(T), one(T))
        end
    end
    return logl
end
