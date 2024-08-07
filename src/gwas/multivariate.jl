"""
Stores all variables that can be treated as "fixed" during automatic differentiation
"""
struct MultivariateCopulaData{T, D, L}
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    vecdist::Vector{D} # length d vector of marginal distributions for each phenotype
    veclink::Vector{L} # length d vector of link functions for each phenotype's marginal distribution
    # data dimension
    n::Int # sample size
    d::Int # number of phenotypes per sample
    p::Int # number of (non-genetic) covariates per sample
    s::Int # number of nuisance parameters 
    m::Int # number of parameters in cholesky matrix L
end

"""
The full QuasiCopula struct for multivariate response model
"""
struct MultivariateCopulaModel{T} <: MOI.AbstractNLPEvaluator
    data::MultivariateCopulaData
    # parameters
    B::Matrix{T}              # p × d matrix of mean regression coefficients, Y = XB
    L::Cholesky{T, Matrix{T}} # cholesky factor of Γ: L.L*L.L' = Γ
    ϕ::Vector{T}              # s-vector of nuisance parameters
    nuisance_idx::Vector{Int} # indices that are nuisance parameters, indexing into vecdist
    # working arrays
    grad::Vector{T}  # length pd + m + s vector, gradient of parameters
    vechL::Vector{T} # vechL = vech(L.L) where vech() computes the lower-triangular part of L
    # res::Vector      # non-standardized residuals, res = y - μ in GLM regression
    # std_res::Vector  # standardized residuals, res = (y - μ) / σ in GLM regression
    # η::Matrix        # linear predictors η in GLM regression, each row is ηi
    # mutating vectors
    # par_store::Vector{T}    # length pd+m+s storage vector
    # res_storage::Vector     # storage vector for res
    # std_res_storage::Vector # storage vector for std_res
    # η_storage::Vector       # storage vector for η
    # μ_storage::Vector       # storage vector for μ
    # varμ_storage::Vector    # storage vector for varμ
    # storage_d::Vector       # another storage vector
end

function MultivariateCopulaModel(
    Y::Matrix{T}, # n × d
    X::Matrix{T}, # n × p
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each phenotype
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    supported_nuisance_dist = [Normal, NegativeBinomial, Normal{Float64}, NegativeBinomial{Float64}]
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
    # fixed data variables
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    data = MultivariateCopulaData(Y, X, vecdist, veclink, n, d, p, s, m)
    # initialize variables
    B = zeros(T, p, d)
    ϕ = fill(one(T), s)
    grad = zeros(T, p*d+m+s)
    # res = zeros(T, d)
    # std_res = zeros(T, d)
    # η = zeros(T, n, d)
    # covariance matrix
    Γ = cor(Y)
    L = cholesky(Symmetric(Γ, :L)) # use lower triangular part of Γ
    vechL = vech(L.L)
    # tmp storages
    # par_store = zeros(T, p*d+m+s)
    # res_storage = zeros(T, d)
    # std_res_storage = zeros(T, d)
    # storage_d = zeros(T, d)
    return MultivariateCopulaModel(
        data,
        B, L, ϕ, nuisance_idx, 
        grad, vechL, # res, std_res, η, 
        # par_store, res_storage, std_res_storage, storage_d
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
    for j in 1:n, i in j:m
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
    for j in 1:n, i in j:m
        A[i, j] = v[idx]
        idx += 1
    end
    A
end
function un_vech!(L::Cholesky, v::AbstractVector)
    if L.uplo === 'L'
        un_vech!(L.factors, v)
    else
        error("L.uplo !== 'L'! Construct cholesky factors using cholesky(x, :L)")
    end
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
    qc_model::MultivariateCopulaModel{T},
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 5, 
             "tol"                        => 10^-3,
             "max_iter"                   => 10000,
             "accept_after_max_steps"     => 50,
             "warm_start_init_point"      => "yes", 
             "limited_memory_max_history" => 6, # default value
             "hessian_approximation"      => "limited-memory",
            #  "derivative_test"            => "first-order",
             ),
    verbose::Bool = true
    ) where T <: BlasReal
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initialize conditions
    data = qc_model.data
    n, p, d, m, s = data.n, data.p, data.d, data.m, data.s
    dimL = Int((-1 + sqrt(1 + 8m)) / 2) # side length of L
    initialize_model!(qc_model)
    npar = p * d + m + s # pd fixed effects, m covariance params, s nuisance params
    npar ≥ 0.2n && error("Estimating $npar params with $n samples, not recommended")
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, qc_model)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints (nuisance parameters and diagonal of cholesky must be >0)
    for k in p*d+m+1:npar
        solver.variables.lower[k] = 0
    end
    offset = p*d + 1
    for k in 1:dimL
        solver.variables.lower[offset] = 0
        offset += dimL - (k - 1)
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
    verbose && optstat ∉ (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) && 
        @warn("Optimization unsuccesful; got $optstat")

    # update parameters and refresh gradient
    optimpar_to_modelpar!(qc_model, MOI.get(solver, MOI.VariablePrimal(), solver_pars))

    return loglikelihood!(MOI.get(solver, MOI.VariablePrimal(), solver_pars), qc_model.data)
end

"""
    modelpar_to_optimpar!(par, qc_model)

Translate model parameters in `qc_model` to optimization variables in `par`
"""
function modelpar_to_optimpar!(
    par :: Vector,
    qc_model :: MultivariateCopulaModel
    )
    data = qc_model.data
    p, d, m = data.p, data.d, data.m
    # β
    copyto!(par, qc_model.B)
    # variance params
    var_range = p * d + 1:p * d + m
    vech!(@view(par[var_range]), qc_model.L)
    # nuisance params
    offset = p * d + m + 1
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
    optimpar_to_modelpar!(qc_model.B, qc_model.L, qc_model.vechL, qc_model.ϕ, par)
    return qc_model
end

function optimpar_to_modelpar!(
    B :: Matrix,
    L :: Cholesky,
    vechL :: Vector,
    ϕ :: Vector,
    par :: Vector
    )
    p, d = size(B)
    m = length(vechL)
    # β
    copyto!(B, 1, par, 1, p * d)
    # Γ
    var_range = p * d + 1:p * d + m
    vechL .= @view(par[var_range])
    un_vech!(L, @view(par[var_range]))
    # nuisance parameters
    offset = p * d + m + 1
    ϕ .= par[offset:end]
    return nothing
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
    return loglikelihood!(par, qc_model.data)
end

function MOI.eval_objective_gradient(
    qc_model :: MultivariateCopulaModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    # objective value
    data = qc_model.data
    obj = loglikelihood!(par, data)
    # compute grad with Enzyme.jl autodiff
    # grad_storage = qc_model.grad
    grad_storage = zeros(length(par))
    GC.enable(false)
    Enzyme.autodiff(
        Reverse, loglikelihood!,
        Duplicated(par, grad_storage),
        Const(data),
    )
    GC.enable(true)
    copyto!(grad, grad_storage)
    copyto!(qc_model.grad, grad_storage)
    return obj
end

"""
    initialize_model!(qc_model)

Initializes mean parameters B with univariate regression values (we fit a GLM
to each y separately). 
"""
function initialize_model!(qc_model::MultivariateCopulaModel)
    # univariate GLMs
    for (j, y) in enumerate(eachcol(qc_model.data.Y))
        fit_glm = glm(qc_model.data.X, y, qc_model.data.vecdist[j], qc_model.data.veclink[j])
        qc_model.B[:, j] .= fit_glm.pp.beta0
    end
    # covariance 
    Γ = cor(qc_model.data.Y)
    L = cholesky(Symmetric(Γ, :L)) # use lower triangular part of Γ
    copyto!(qc_model.L, L)
    vech!(qc_model.vechL, LowerTriangular(L.factors))
    # nuisance parameters
    fill!(qc_model.ϕ, 1)
    return nothing
end

function loglikelihood!(
    par::Vector, # first p*d are β, next m are for vech(L), next s are for nuisance
    data::MultivariateCopulaData,
    )
    n, p, m, d = data.n, data.p, data.m, data.d

    # allocate everything for now
    B = zeros(p, d)
    copyto!(B, 1, par, 1, p * d)
    L = cholesky(Symmetric(zeros(d, d), :L), check=false)
    un_vech!(L, @view(par[p * d + 1:p * d + m]))
    ϕ = par[p * d + m + 1:end]
    std_res = zeros(d)
    η = zeros(n, d)
    storage_d = zeros(d)

    # update η
    mul!(η, data.X, B)
    # loglikelihood for each sample
    logl = zero(eltype(data.X))
    for i in 1:data.n
        # update res and std_res
        update_res!(data, i, std_res, η, ϕ)
        # loglikelihood term 2, i.e. sum sum ln(f_ij | β)
        logl += component_loglikelihood(data, i, η, ϕ)
        # loglikelihood term 1, i.e. -sum ln(1 + 0.5tr(Γ))    # todo: move term 1
        logl -= log(1 + 0.5tr(L))
        # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
        mul!(storage_d, Transpose(L.L), std_res)
        logl += log(1 + 0.5sum(abs2, storage_d))
    end
    return logl
end

# computes trace of Γ = L.L*L.L' = vec(L.L)'vec(L.L)
function LinearAlgebra.tr(L::Cholesky)
    s = zero(eltype(L.factors))
    for Lij in LowerTriangular(L.factors)
        s += abs2(Lij)
    end
    return s
end

function update_res!(data::MultivariateCopulaData, i::Int, std_res::Vector, η::Matrix, ϕ::Vector)
    yi = @view(data.Y[i, :])
    ηi = @view(η[i, :])
    nuisance_counter = 1
    vecdist = data.vecdist
    veclink = data.veclink
    for j in eachindex(yi)
        μ_j = GLM.linkinv(veclink[j], ηi[j])
        varμ_j = GLM.glmvar(vecdist[j], μ_j) # Note: for negative binomial, d.r is used
        res_j = yi[j] - μ_j
        if typeof(vecdist[j]) <: Normal
            σ2 = abs(ϕ[nuisance_counter]) # need abs since IPOPT can guess values like -2.0092714282576747e14
            std_res[j] = res_j / sqrt(σ2)
            nuisance_counter += 1
        else
            std_res[j] = res_j / sqrt(varμ_j)
        end
    end
    return nothing
end

function component_loglikelihood(
    data::MultivariateCopulaData, i::Int, η::Matrix, ϕ::Vector
    )
    yi = @view(data.Y[i, :])
    ηi = @view(η[i, :])
    nuisance_counter = 1
    logl = 0.0
    for j in eachindex(yi)
        dist = data.vecdist[j]
        link = data.veclink[j]
        μ_ij = GLM.linkinv(link, ηi[j])
        if typeof(dist) <: Normal
            σ2 = abs(ϕ[nuisance_counter])
            logl += QuasiCopula.loglik_obs(dist, yi[j], μ_ij, 1.0, σ2)::Float64
            nuisance_counter += 1
        else
            logl += QuasiCopula.loglik_obs(dist, yi[j], μ_ij, 1.0, 1.0)::Float64
        end
    end
    return logl
end
