"""
Stores all variables that can be treated as "fixed" during automatic differentiation
"""
struct MultivariateCopulaData_AD{T, D, L}
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    V::Vector{Matrix{T}} # length m vector of d × d matrices
    vecdist::Vector{D} # length d vector of marginal distributions for each phenotype
    veclink::Vector{L} # length d vector of link functions for each phenotype's marginal distribution
    # data dimension
    n::Int # sample size
    d::Int # number of phenotypes per sample
    p::Int # number of (non-genetic) covariates per sample
    s::Int # number of nuisance parameters 
    m::Int # number of parameters in θ
end

"""
The full ApproxCopula struct for multivariate response model
"""
struct MultivariateCopulaModel_AD{T} <: MOI.AbstractNLPEvaluator
    data::MultivariateCopulaData_AD
    # parameters
    B::Matrix{T}              # p × d matrix of mean regression coefficients, Y = XB
    θ::Vector{T}
    ϕ::Vector{T}              # s-vector of nuisance parameters
    nuisance_idx::Vector{Int} # indices that are nuisance parameters, indexing into vecdist
    # working arrays
    grad::Vector{T}  # length pd + m + s vector, gradient of parameters
    # vechL::Vector{T} # vechL = vech(L.L) where vech() computes the lower-triangular part of L
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

function MultivariateCopulaModel_AD(
    Y::Matrix{T}, # n × d
    X::Matrix{T}, # n × p
    V::Union{Matrix{T}, Vector{Matrix{T}}}, # variance component matrices of the phenotypes
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each phenotype
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    supported_nuisance_dist = [Normal, NegativeBinomial]
    ) where T <: BlasReal
    n, d = size(Y)
    p = size(X, 2)
    m = typeof(V) <: Matrix ? 1 : length(V)
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
    data = MultivariateCopulaData_AD(Y, X, V, vecdist, veclink, n, d, p, s, m)
    # initialize variables
    B = zeros(T, p, d)
    θ = zeros(T, m)
    ϕ = fill(one(T), s)
    grad = zeros(T, p*d+m+s)
    # res = zeros(T, d)
    # std_res = zeros(T, d)
    # η = zeros(T, n, d)
    # covariance matrix
    # Γ = cov(Y)
    # L = cholesky(Symmetric(Γ, :L)) # use lower triangular part of Γ
    # vechL = vech(L.L)
    # tmp storages
    # par_store = zeros(T, p*d+m+s)
    # res_storage = zeros(T, d)
    # std_res_storage = zeros(T, d)
    # storage_d = zeros(T, d)
    return MultivariateCopulaModel_AD(
        data,
        B, θ, ϕ, nuisance_idx, grad
        # grad, vechL, # res, std_res, η, 
        # par_store, res_storage, std_res_storage, storage_d
    )
end

"""
    fit!(qc_model::MultivariateCopulaModel_AD, solver=Ipopt.IpoptSolver)

Fit an `MultivariateCopulaModel_AD` object by MLE using a nonlinear programming
solver. Start point should be provided in `qc_model.β`, `qc_model.`, `qc_model.ϕ`

# Arguments
- `qc_model`: A `MultivariateCopulaModel_AD` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton
    iterations with convergence tolerance 10^-6. (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    qc_model::MultivariateCopulaModel_AD{T},
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 5, 
             "tol"                        => 10^-3,
             "max_iter"                   => 100,
             "accept_after_max_steps"     => 10,
             "warm_start_init_point"      => "yes", 
             "limited_memory_max_history" => 6, # default value
             "hessian_approximation"      => "limited-memory",
            #  "derivative_test"            => "first-order",
             ),
    ) where T <: BlasReal
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")
    
    # Pass options to solver
    config_solver(solver, solver_config)

    # initialize conditions
    data = qc_model.data
    n, p, d, m, s = data.n, data.p, data.d, data.m, data.s
    initialize_model!(qc_model)
    npar = p * d + m + s # pd fixed effects, m covariance params, s nuisance params
    npar ≥ 0.2n && error("Estimating $npar params with $n samples, not recommended")
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, qc_model)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints (VC and nuisance parameters must be >0)
    for k in p*d+1:npar
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

    return loglikelihood!(MOI.get(solver, MOI.VariablePrimal(), solver_pars), qc_model.data)
end

"""
    modelpar_to_optimpar!(par, qc_model)

Translate model parameters in `qc_model` to optimization variables in `par`
"""
function modelpar_to_optimpar!(
    par :: Vector,
    qc_model :: MultivariateCopulaModel_AD
    )
    data = qc_model.data
    p, d, m = data.p, data.d, data.m
    # β
    copyto!(par, qc_model.B)
    # variance params
    var_range = p * d + 1:p * d + m
    vech!(@view(par[var_range]), qc_model.θ)
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
    qc_model :: MultivariateCopulaModel_AD,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model.B, qc_model.θ, qc_model.ϕ, par)
    return qc_model
end

function optimpar_to_modelpar!(
    B :: Matrix,
    θ :: Vector,
    ϕ :: Vector,
    par :: Vector
    )
    p, d = size(B)
    m = length(θ)
    # β
    copyto!(B, 1, par, 1, p * d)
    # Γ
    var_range = p * d + 1:p * d + m
    θ .= @view(par[var_range])
    # nuisance parameters
    offset = p * d + m + 1
    ϕ .= par[offset:end]
    return nothing
end

function MOI.initialize(
    qc_model::MultivariateCopulaModel_AD,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(qc_model))
            error("Unsupported feature $feat, requested = $requested_features")
        end
    end
end

MOI.features_available(qc_model::MultivariateCopulaModel_AD) = [:Grad]

function MOI.eval_objective(
    qc_model :: MultivariateCopulaModel_AD,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    return loglikelihood!(par, qc_model.data)
end

function MOI.eval_objective_gradient(
    qc_model :: MultivariateCopulaModel_AD,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    # objective value
    data = qc_model.data
    obj = loglikelihood!(par, data)
    # compute grad with Enzyme.jl autodiff
    grad_storage = zeros(length(par))
    Enzyme.autodiff(
        Reverse, loglikelihood!,
        Duplicated(par, grad_storage),
        Const(data),
    )
    copyto!(grad, grad_storage)
    return obj
end

"""
    initialize_model!(qc_model)

Initializes mean parameters B with univariate regression values (we fit a GLM
to each y separately). 
"""
function initialize_model!(qc_model::MultivariateCopulaModel_AD)
    # univariate GLMs
    for (j, y) in enumerate(eachcol(qc_model.data.Y))
        fit_glm = glm(qc_model.data.X, y, qc_model.data.vecdist[j], qc_model.data.veclink[j])
        qc_model.B[:, j] .= fit_glm.pp.beta0
    end
    # covariance 
    fill!(qc_model.θ, 0.5)
    # nuisance parameters
    fill!(qc_model.ϕ, 1)
    return nothing
end

function loglikelihood!(
    par::Vector, # first p*d are β, next m are for θ, next s are for nuisance
    data::MultivariateCopulaData_AD,
    )
    n, p, m, d = data.n, data.p, data.m, data.d

    # allocate everything for now
    B = zeros(p, d)
    copyto!(B, 1, par, 1, p * d)
    θ = par[p * d + 1:p * d + m]
    ϕ = par[p * d + m + 1:end]
    std_res = zeros(d)
    η = zeros(n, d)
    Γ = θ[1] * data.V[1] + θ[2] * data.V[2]

    # update η
    mul!(η, data.X, B)
    # loglikelihood for each sample
    # logl = zero(eltype(data.X))
    # loglikelihood term 1, i.e. -sum ln(1 + 0.5tr(Γ))
    logl = -n * log(1 + 0.5tr(Γ))
    for i in 1:data.n
        # update res and std_res
        update_res!(data, i, std_res, η, ϕ)
        # loglikelihood term 2, i.e. sum sum ln(f_ij | β)
        logl += component_loglikelihood(data, i, η, ϕ)
        # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
        logl += log(1 + 0.5dot(std_res, Γ, std_res))
    end
    return logl
end

function update_res!(data::MultivariateCopulaData_AD, i::Int, std_res::Vector, η::Matrix, ϕ::Vector)
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
    data::MultivariateCopulaData_AD, i::Int, η::Matrix, ϕ::Vector
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
            logl += ApproxCopula.loglik_obs(dist, yi[j], μ_ij, 1.0, σ2)::Float64
            nuisance_counter += 1
        else
            logl += ApproxCopula.loglik_obs(dist, yi[j], μ_ij, 1.0, 1.0)::Float64
        end
    end
    return logl
end
