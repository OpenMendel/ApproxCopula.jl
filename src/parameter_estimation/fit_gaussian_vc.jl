"""
    fit!(gcm::GaussianCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `GaussianCopulaVCModel` object by MLE using a nonlinear programming 
solver. This is for Normal base.

# Arguments
- `gcm`: A `GaussianCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton
    iterations with convergence tolerance `10^-3`.
"""
function fit!(
        gcm::GaussianCopulaVCModel,
        solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
        solver_config :: Dict = 
            Dict("print_level"                => 5, 
                 "tol"                        => 10^-6,
                 "max_iter"                   => 1000,
                 "accept_after_max_steps"     => 50,
                 "warm_start_init_point"      => "yes", 
                 "limited_memory_max_history" => 6, # default value
                 "hessian_approximation"      => "limited-memory",
                #  "derivative_test"            => "first-order",
                 ),
    )
    T = eltype(gcm.β)
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initial conditions
    initialize_model!(gcm)
    npar = gcm.p + gcm.m + 1
    par0 = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, gcm)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints
    for k in gcm.p+1:npar
        solver.variables.lower[k] = 0
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
    loglikelihood!(gcm, true, false)
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par` for Normal base.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: GaussianCopulaVCModel
    )
    # β
    copyto!(par, gcm.β)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.θ[k]
        offset += 1
    end
    par[offset] = gcm.τ[1]
    par
end

"""
    optimpar_to_modelpar_quasi!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: GaussianCopulaVCModel,
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset   += 1
    end
    gcm.τ[1] = par[offset]
    gcm
end

function MOI.initialize(
    gcm::GaussianCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(gcm::GaussianCopulaVCModel) = [:Grad, :Hess]

function MOI.eval_objective(
    gcm :: GaussianCopulaVCModel,
    par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
    gcm  :: GaussianCopulaVCModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt variance comps
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        grad[offset] = gcm.∇θ[k]
        offset += 1
    end
    grad[offset] = gcm.∇τ[1]
    obj
end

function MOI.hessian_lagrangian_structure(gcm::GaussianCopulaVCModel)
    m◺ = ◺(gcm.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    # Hββ block
    idx = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # variance components
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    arr1[idx] = gcm.p + gcm.m + 1
    arr2[idx] = gcm.p + gcm.m + 1
    return collect(zip(arr1, arr2))
end

function MOI.eval_hessian_lagrangian(
    gcm :: GaussianCopulaVCModel,
    H   :: AbstractVector{T},
    par :: AbstractVector{T},
    σ   :: T,
    μ   :: AbstractVector{T}
    )where {T <: BlasReal}
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, true)
    # Hβ block
    idx = 1
    @inbounds for j in 1:gcm.p, i in 1:j
        H[idx] = gcm.Hβ[i, j]
        idx += 1
    end
    # Haa block
    @inbounds for j in 1:gcm.m, i in 1:j
        H[idx] = gcm.Hθ[i, j]
        idx += 1
    end
    H[idx] = gcm.Hτ[1, 1]
    # lmul!(σ, H)
    H .*= σ
end
