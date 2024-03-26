"""
    fit!(qc_model::GLMCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit a `GLMCopulaVCModel` model object by MLE using a nonlinear programming solver.
This is for Poisson and Bernoulli base distributions.
Start point should be provided in `qc_model.β`, `qc_model.θ`.

# Arguments
- `qc_model`: A `GLMCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, hessian_approximation = "limited-memory")`)
"""
function fit!(
        qc_model :: GLMCopulaVCModel{T, D, Link},
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
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initial conditions
    initialize_model!(qc_model)
    npar = qc_model.p + qc_model.m
    par0 = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, qc_model)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints
    for k in qc_model.p+1:npar
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
    loglikelihood!(qc_model, true, false)
end

"""
    modelpar_to_optimpar!(par, qc_model)

Translate model parameters in `qc_model` to optimization variables in `par` for Poisson and Bernoulli base with only mean parameters.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        qc_model :: GLMCopulaVCModel{T, D, Link}
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    # β
    copyto!(par, qc_model.β)
    # L
    offset = qc_model.p + 1
    @inbounds for k in 1:qc_model.m
        par[offset] = qc_model.θ[k]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(qc_model, par)

Translate optimization variables in `par` to the model parameters in `qc_model`.
"""
function optimpar_to_modelpar!(
        qc_model :: GLMCopulaVCModel{T, D, Link},
        par :: Vector
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    # β
    copyto!(qc_model.β, 1, par, 1, qc_model.p)
    offset = qc_model.p + 1
    @inbounds for k in 1:qc_model.m
        qc_model.θ[k] = par[offset]
        offset += 1
    end
    qc_model
end

function MOI.initialize(
    qc_model::GLMCopulaVCModel,
    requested_features::Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in MOI.features_available(qc_model))
            error("Unsupported feature $feat, requested = $requested_features")
        end
    end
end

MOI.features_available(qc_model::GLMCopulaVCModel) = [:Grad, :Hess]

function MOI.eval_objective(
    qc_model :: GLMCopulaVCModel,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    loglikelihood!(qc_model, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
    qc_model :: GLMCopulaVCModel{T, D, Link},
    grad :: Vector,
    par  :: Vector
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    optimpar_to_modelpar!(qc_model, par)
    obj = loglikelihood!(qc_model, true, false)
    # gradient wrt β
    copyto!(grad, qc_model.∇β)
    # gradient wrt variance comps
    offset = qc_model.p + 1
    @inbounds for k in 1:qc_model.m
        grad[offset] = qc_model.∇θ[k]
        offset += 1
    end
    obj
end

# MathProgBase.eval_g(qc_model::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}, g, par) = nothing
# MathProgBase.jac_structure(qc_model::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}) = Int[], Int[]
# MathProgBase.eval_jac_g(qc_model::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}, J, par) = nothing

function MOI.eval_constraint(
    m   :: GLMCopulaVCModel,
    g   :: Vector{T},
    par :: AbstractVector{T}
    ) where {T<:BlasReal}
    return nothing
end

function MOI.hessian_lagrangian_structure(qc_model::GLMCopulaVCModel)
    m◺ = ◺(qc_model.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(qc_model.p) + m◺)
    arr2 = Vector{Int}(undef, ◺(qc_model.p) + m◺)
    # Hββ block
    idx = 1
    for j in 1:qc_model.p
        for i in j:qc_model.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # variance components
    for j in 1:qc_model.m
        for i in 1:j
            arr1[idx] = qc_model.p + i
            arr2[idx] = qc_model.p + j
            idx += 1
        end
    end
    return collect(zip(arr1, arr2))
end

function MOI.eval_hessian_lagrangian(
    qc_model :: GLMCopulaVCModel,
    H   :: AbstractVector{T},
    par :: AbstractVector{T},
    σ   :: T,
    μ   :: AbstractVector{T}
    )where T <: BlasReal
    optimpar_to_modelpar!(qc_model, par)
    loglikelihood!(qc_model, true, true)
    # Hβ block
    idx = 1
    @inbounds for j in 1:qc_model.p, i in 1:j
        H[idx] = qc_model.Hβ[i, j]
        idx += 1
    end
    # Haa block
    @inbounds for j in 1:qc_model.m, i in 1:j
        H[idx] = qc_model.Hθ[i, j]
        idx += 1
    end
    # lmul!(σ, H)
    H .*= σ
end
