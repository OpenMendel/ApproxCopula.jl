# struct holding intermediate arrays for a given sample
struct MultivariateCopulaVCObs{T <: BlasReal}
    η::Vector{T} # η = B'x (linear predictor value of current sample)
    μ::Vector{T} # μ = linkinv(link, η) (mean of current sample)
    res::Vector{T} # res[i] = yᵢ - μᵢ (residual)
    std_res::Vector{T} # std_res[i] = (yᵢ - μᵢ) / σᵢ (standardized residual)
    dμ::Vector{T} # intermediate GLM quantity
    varμ::Vector{T} # intermediate GLM quantity
    w1::Vector{T} # intermediate GLM quantity
    w2::Vector{T} # intermediate GLM quantity
    q::Vector{T} # q[k] = res_i' * V_i[k] * res_i / 2 (this is variable b in VC model, see sec 6.2 of ApproxCopula paper)
    ∇resβ::Matrix{T} # gradient of standardized residual with respect to beta
    ∇vecB::Vector{T} # gradient of loglikelihood wrt β = vec(B)
    ∇θ::Vector{T}   # gradient of loglikelihood wrt θ (variance components)
    ∇ϕ::Vector{T}   # gradient of loglikelihood wrt ϕ (nuisnace parameters)
    # m1::Vector{T}
    # m2::Vector{T}
    storage_d::Vector{T}
    # storage_dp::Matrix{T}
end

function MultivariateCopulaVCObs(T, d, p, m, s)
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
    # m1 = zeros(T, m)
    # m2 = zeros(T, m)
    storage_d = zeros(T, d)
    # storage_dp = zeros(T, d, p)
    obs = MultivariateCopulaVCObs(
        η, μ, res, std_res, dμ, varμ, w1, w2, q, ∇resβ, ∇vecB, ∇θ, ∇ϕ, storage_d
    )
    return obs
end

struct MultivariateCopulaVCModel{T <: BlasReal} <: MOI.AbstractNLPEvaluator
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    V::Vector{Matrix{T}} # length m vector of d × d matrices
    vecdist::Vector{<:UnivariateDistribution} # length d vector of marginal distributions for each phenotype
    veclink::Vector{<:Link} # length d vector of link functions for each phenotype's marginal distribution
    data::Vector{MultivariateCopulaVCObs{T}}
    # data dimension
    n::Int # sample size
    d::Int # number of phenotypes per sample
    p::Int # number of (non-genetic) covariates per sample
    m::Int # number of variance components
    s::Int # number of nuisance parameters 
    # parameters
    B::Matrix{T}    # p × d matrix of mean regression coefficients, Y = XB
    θ::Vector{T}    # length m vector of variance components
    ϕ::Vector{T}    # s-vector of nuisance parameters. Currently only Gaussian works, so ϕ is just a vector of `τ`s (inverse variance)
    nuisance_idx::Vector{Int} # indices that are nuisance parameters, indexing into vecdist
    # working arrays
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2 (this is variable c in VC model)
    # Γ::Matrix{T}      # d × d covariance matrix, in VC model this is θ[1]*V[1] + ... + θ[m]*V[m]
    ∇vecB::Vector{T}  # length pd vector, its the gradient of vec(B) 
    ∇θ::Vector{T}     # length m vector, gradient of variance components
    ∇ϕ::Vector{T}     # length s vector, gradient of nuisance parameters
    # HvecB::Matrix{T}  # pd × pd matrix of Hessian
    # Hθ::Matrix{T}     # m × m matrix of Hessian for variance components
    # Hϕ::Matrix{T}
    penalized::Bool
end

function MultivariateCopulaVCModel(
    Y::Matrix{T},
    X::Matrix{T},
    V::Union{Matrix{T}, Vector{Matrix{T}}}, # variance component matrices of the phenotypes
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each phenotype
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    penalized = false
    ) where T <: BlasReal
    n, d = size(Y)
    p = size(X, 2)
    m = typeof(V) <: Matrix ? 1 : length(V)
    n == size(X, 1) || error("Number of samples in Y and X mismatch")
    nuisance_idx = findall(x -> x == Normal || typeof(x) <: Normal, vecdist)
    s = length(nuisance_idx)
    any(x -> typeof(x) <: NegativeBinomial, vecdist) && 
        error("Negative binomial base not supported yet")
    # initialize variables
    B = zeros(T, p, d)
    θ = zeros(T, m)
    ϕ = fill(one(T), s)
    # t is variable c in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2
    # Because all Vs are the same in multivariate analysis, all samples share the same t
    t = [tr(V[k])/2 for k in 1:m]
    # Γ = zeros(T, d, d)
    ∇vecB = zeros(T, p*d)
    ∇θ = zeros(T, m)
    ∇ϕ = zeros(T, s)
    # HvecB = zeros(T, p*d, p*d)
    # Hθ = zeros(T, m, m)
    # construct MultivariateCopulaVCObs that hold intermediate variables for each sample
    data = MultivariateCopulaVCObs{T}[]
    for _ in 1:n
        push!(data, MultivariateCopulaVCObs(T, d, p, m, s))
    end
    # change type of variables to match struct
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    typeof(V) <: Matrix && (V = [V])
    return MultivariateCopulaVCModel(
        Y, X, V, vecdist, veclink, data,
        n, d, p, m, s, 
        B, θ, ϕ, nuisance_idx, t, 
        ∇vecB, ∇θ, ∇ϕ, 
        penalized
    )
end

"""
    fit!(qc_model::MultivariateCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `MultivariateCopulaVCModel` object by MLE using a nonlinear programming
solver. Start point should be provided in `qc_model.β`, `qc_model.θ`, `qc_model.ϕ`

# Arguments
- `qc_model`: A `MultivariateCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton
    iterations with convergence tolerance 10^-6. (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    qc_model::MultivariateCopulaVCModel,
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
    npar = p * d + m + s # pd fixed effects, m variance-components, s nuisance params
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, qc_model)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints (variance components and nuisance parameters must be >0)
    for k in p*d+1:p*d+m+s
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

Translate model parameters in `qc_model` to optimization variables in `par` for Normal base.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    qc_model :: MultivariateCopulaVCModel
    )
    # β
    copyto!(par, qc_model.B)
    # variance components
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        par[offset] = qc_model.θ[k]
        offset += 1
    end
    @inbounds for i in 1:qc_model.s
        par[offset] = qc_model.ϕ[i]
        offset += 1
    end
    return par
end

"""
    optimpar_to_modelpar_quasi!(qc_model, par)

Translate optimization variables in `par` to the model parameters in `qc_model`.
"""
function optimpar_to_modelpar!(
    qc_model :: MultivariateCopulaVCModel,
    par :: Vector
    )
    # β
    copyto!(qc_model.B, 1, par, 1, qc_model.p * qc_model.d)
    # variance components
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        qc_model.θ[k] = par[offset]
        offset += 1
    end
    # nuisance parameters
    @inbounds for i in 1:qc_model.s
        qc_model.ϕ[i] = par[offset]
        offset += 1
    end
    return qc_model
end

function MOI.initialize(
    qc_model::MultivariateCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(qc_model))
            error("Unsupported feature $feat, requested = $requested_features")
        end
    end
end

MOI.features_available(qc_model::MultivariateCopulaVCModel) = [:Grad]

function MOI.eval_objective(
    qc_model :: MultivariateCopulaVCModel,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    return loglikelihood!(qc_model, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
    qc_model :: MultivariateCopulaVCModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    obj = loglikelihood!(qc_model, true, false)
    # gradient wrt β
    copyto!(grad, qc_model.∇vecB)
    # gradient wrt variance comps
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        grad[offset] = qc_model.∇θ[k]
        offset += 1
    end
    # gradient wrt to nuisance parameters
    @inbounds for k in 1:qc_model.s
        grad[offset] = qc_model.∇ϕ[k]
        offset += 1
    end
    return obj
end

"""
    initialize_model!(qc_model)

Initializes mean parameters B with univariate regression values (we fit a GLM
to each y separately)
"""
function initialize_model!(qc_model::MultivariateCopulaVCModel)
    for (j, y) in enumerate(eachcol(qc_model.Y))
        fit_glm = glm(qc_model.X, y, qc_model.vecdist[j], qc_model.veclink[j])
        qc_model.B[:, j] .= fit_glm.pp.beta0
    end
    fill!(qc_model.ϕ, 1)
    fill!(qc_model.θ, 0.5)
    return nothing
end

function loglikelihood!(
    qc_model::MultivariateCopulaVCModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    if needgrad
        fill!(qc_model.∇vecB, 0)
        fill!(qc_model.∇θ, 0)
        fill!(qc_model.∇ϕ, 0)
    end
    if needhess
        error("Hessian not implemented for MultivariateCopulaVCModel!")
        # fill!(qc_model.HvecB, 0)
        # fill!(qc_model.Hθ, 0)
    end
    logl = zero(T)
    for i in 1:qc_model.n
        logl += loglikelihood!(qc_model, i, needgrad, needhess)
        if needgrad
            qc_model.∇vecB .+= qc_model.data[i].∇vecB
            qc_model.∇θ .+= qc_model.data[i].∇θ
            qc_model.∇ϕ .+= qc_model.data[i].∇ϕ
        end
        if needhess
            # qc_model.Hβ .+= qc_model.data[i].HvecB
            # qc_model.Hθ .+= qc_model.data[i].Hθ
        end
    end
    return logl
end

# evaluates the loglikelihood for sample i
function loglikelihood!(
    qc_model::MultivariateCopulaVCModel{T},
    i::Int,
    needgrad::Bool = false,
    needhess::Bool = false;
    ) where T <: BlasReal
    d = qc_model.d        # number of phenotypes
    p = qc_model.p        # number of covarites
    θ = qc_model.θ        # variance components
    qc = qc_model.data[i] # sample i's storage
    if needgrad
        fill!(qc.∇vecB, 0)
        fill!(qc.∇θ, 0)
        fill!(qc.∇ϕ, 0)
    end
    # update residuals and its gradient
    update_res!(qc_model, i)
    std_res_differential!(qc_model, i) # compute ∇resβ
    # loglikelihood term 2 i.e. sum sum ln(f_ij | β)
    logl = ApproxCopula.component_loglikelihood(qc_model, i)
    # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
    tsum = dot(θ, qc_model.t) # tsum = 0.5tr(Γ)
    logl += -log(1 + tsum)
    # compute ∇resβ*Γ*res and variable b for variance component model
    @inbounds for k in 1:qc_model.m # loop over m variance components
        mul!(qc.storage_d, qc_model.V[k], qc.std_res) # storage_d = V[k] * r
        if needgrad
            BLAS.gemv!('N', θ[k], qc.∇resβ, qc.storage_d, one(T), qc.∇vecB) # ∇β = ∇r*Γ*r
        end
        qc.q[k] = dot(qc.std_res, qc.storage_d) / 2 # q[k] = 0.5 r * V[k] * r
    end
    # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
    qsum = dot(θ, qc.q) # qsum = 0.5 r*Γ*r
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if qc_model.penalized
        logl -= 0.5 * dot(θ, θ)
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum) # inv1pq = 1 / (1 + 0.5r'Γr)
        if needhess
            # # approximate Hessian of β
            # mul!(gc.storage_dp, Diagonal(gc.w2), gc.X)
            # BLAS.gemm!('T', 'N', -one(T), gc.X, gc.storage_dp, zero(T), gc.Hβ) # Hβ = -Xi'*Diagonal(W2)*Xi
            # BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # Hβ = -Xi'*Diagonal(W2)*Xi - ∇β*∇β' / (1 + 0.5r'Γr)^2
            # copytri!(gc.Hβ, 'L') # syrk! above only lower triangular
            # # println(gc.Hβ)
            # # println(-1 .* (transpose(gc.X)*Diagonal(gc.w2)*gc.X) - abs2(inv1pq) .* (gc.∇β * transpose(gc.∇β)))
            # # Hessian of vc vector (todo: are these correct?)
            # inv1pt = inv(1 + tsum) # inv1pt = 1 / (1 + 0.5tr(Γ))
            # gc.m1 .= gc.q
            # gc.m1 .*= inv1pq # m1[k] = 0.5 r' * V[k] * r / (1 + 0.5r'Γr)
            # gc.m2 .= gc.t
            # gc.m2 .*= inv1pt
            # BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            # BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            # copytri!(gc.Hθ, 'U')
            # # Hessian of ϕ (todo)
            # # gc.Hϕ[1, 1] = - abs2(qsum * inv1pq / ϕ)
        end
        # compute X'*Diagonal(dg/varμ)*(y-μ) + ∇r'Γr/(1+0.5r'Γr)  (gradient of logl wrt vecB)
        xi = @view(qc_model.X[i, :])
        for j in 1:d
            out = @view(qc.∇vecB[(j-1)*p+1:j*p])
            out .*= inv1pq
            out .+= j in qc_model.nuisance_idx ? xi .* qc.w1[j] * qc.std_res[j] :
                xi .* qc.w1[j] * qc.res[j]
            # BLAS.gemv!('T', one(T), xi, qc.storage_d, inv1pq, out)
        end
        # Gaussian case: compute ∇τ and undo scaling by τ (vecB used std_res which includes extra factor of √τ)
        for (j, idx) in enumerate(qc_model.nuisance_idx)
            τ = abs(qc_model.ϕ[j])
            vecB_range = (idx-1)*p+1:idx*p
            qc.∇vecB[vecB_range] .*= sqrt(τ)
            rss = abs2(qc.std_res[idx]) # std_res contains a factor of sqrt(τ)
            qc.∇ϕ[j] = (1 - rss + 2qsum * inv1pq) / 2τ # this is kind of wrong by autodiff but I can't figure out why
        end
        qc.∇θ .= inv1pq .* qc.q .- inv(1 + tsum) .* qc_model.t
        if qc_model.penalized
            qc.∇θ .-= θ
        end
    end
    # output
    return logl
end

function update_res!(qc_model::MultivariateCopulaVCModel, i::Int)
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
            τ = abs(inv(qc_model.ϕ[nuisance_counter]))
            obs.std_res[j] = obs.res[j] * sqrt(τ)
            nuisance_counter += 1
        else
            obs.std_res[j] = obs.res[j] / sqrt(obs.varμ[j])
        end
    end
    return nothing
end

"""
qc_model.data[i].∇resβ is dp × d matrix that stores ∇rᵢ(β), i.e. gradient of sample i's
residuals with respect to the dp × 1 vector β. Each of the d columns of ∇rᵢ(β) 
stores ∇rᵢⱼ(β), a length dp vector.
"""
function std_res_differential!(qc_model::MultivariateCopulaVCModel, i::Int)
    obs = qc_model.data[i]
    d = qc_model.d
    p = qc_model.p # p = number of covariates, d = number of phenotypes
    xi = @view(qc_model.X[i, :])
    @inbounds for j in 1:d # loop over columns
        for k in 1:p
            obs.∇resβ[(j-1)*p + k, j] = update_∇res_ij(qc_model.vecdist[j], xi[k], 
                obs.std_res[j], obs.μ[j], obs.dμ[j], obs.varμ[j])
        end
    end
    return nothing
end

function component_loglikelihood(qc_model::MultivariateCopulaVCModel{T}, i::Int) where T <: BlasReal
    y = @view(qc_model.Y[i, :])
    obs = qc_model.data[i]
    nuisance_counter = 1
    logl = zero(T)
    @inbounds for j in eachindex(y)
        dist = qc_model.vecdist[j]
        if typeof(dist) <: Normal
            τ = inv(qc_model.ϕ[nuisance_counter])
            logl += ApproxCopula.loglik_obs(dist, y[j], obs.μ[j], one(T), τ)
            nuisance_counter += 1
        else
            logl += ApproxCopula.loglik_obs(dist, y[j], obs.μ[j], one(T), one(T))
        end
    end
    return logl
end

