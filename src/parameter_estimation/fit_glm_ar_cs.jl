"""
    fit!(gcm::GLMCopulaARModel, solver=Ipopt.IpoptSolver)

Fit an `GLMCopulaARModel` object by MLE using a nonlinear programming solver.
This is for Poisson and Bernoulli base distributions with no additional base distribution parameters than the mean.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`.

# Arguments
- `gcm`: A `GLMCopulaARModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, hessian_approximation = "limited-memory")`)
"""
function fit!(
        gcm::GLMCopulaARModel,
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
    T = eltype(gcm.β)
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initial conditions
    initialize_model!(gcm)
    npar = gcm.p + 2 # rho and sigma squared
    par0 = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, gcm)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints
    solver.variables.lower[gcm.p + 1] = 0
    solver.variables.upper[gcm.p + 1] = 1
    solver.variables.lower[gcm.p + 2] = 0

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
    fit!(gcm::GLMCopulaCSModel, solver=Ipopt.IpoptSolver)

Fit an `GLMCopulaCSModel` object by MLE using a nonlinear programming solver.
This is for Poisson and Bernoulli base distributions with no additional base distribution parameters than the mean.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`.

# Arguments
- `gcm`: A `GLMCopulaCSModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, hessian_approximation = "limited-memory")`)
"""
function fit!(
        gcm::GLMCopulaCSModel,
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
    T = eltype(gcm.β)
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initial conditions
    initialize_model!(gcm)
    npar = gcm.p + 2 # rho and sigma squared
    par0 = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, gcm)
    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    # constraints for rho
    solver.variables.lower[gcm.p + 1] = -inv(gcm.data[1].n - 1)
    solver.variables.upper[gcm.p + 1] = 1

    # constraints for sigma2
    solver.variables.lower[gcm.p + 2] = 0

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

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}
    )
    # β
    copyto!(par, gcm.β)
    # ρ, σ2
    par[end - 1] = gcm.ρ[1]
    par[end] = gcm.σ2[1]
    par
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # ρ, σ2
    gcm.ρ[1] = par[gcm.p + 1]
    gcm.σ2[1] = par[gcm.p + 2]
    gcm
end

function MOI.initialize(
    gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}) = [:Grad, :Hess]

function MOI.eval_objective(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
        gcm  :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
        grad :: Vector,
        par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt ρ
    grad[gcm.p + 1] = gcm.∇ρ[1]
    # gradient wrt σ2
    grad[gcm.p + 2] = gcm.∇σ2[1]
    obj
end

function MOI.hessian_lagrangian_structure(gcm::GLMCopulaCSModel)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    # Hββ block
    idx = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # rho and sigma2
    for j in 1:2
        arr1[idx] = gcm.p + j
        arr2[idx] = gcm.p + j
        idx += 1
    end
    arr1[idx] = gcm.p + 1
    arr2[idx] = gcm.p + 2
    idx += 1
    for k in 1:gcm.p
        arr1[idx] = gcm.p + 2
        arr2[idx] = k
        idx += 1
    end
    # for k in 1:gcm.p
    #     arr1[idx] = gcm.p + 1
    #     arr2[idx] = k
    #     idx += 1
    # end
    return collect(zip(arr1, arr2))
end

function MOI.eval_hessian_lagrangian(
        gcm :: GLMCopulaCSModel,
        H   :: AbstractVector{T},
        par :: AbstractVector{T},
        σ   :: T,
        μ   :: AbstractVector{T}
    ) where {T}
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, true)
    # Hβ block
    idx = 1
    @inbounds for j in 1:gcm.p, i in 1:j
        H[idx] = gcm.Hβ[i, j]
        idx   += 1
    end
    # Haa block
    H[idx] = gcm.Hρ[1, 1]
    idx += 1
    H[idx] = gcm.Hσ2[1, 1]
    idx += 1
    H[idx] = gcm.Hρσ2[1, 1]
    idx += 1
    for k in 1:gcm.p
        H[idx] = gcm.Hβσ2[k]
        idx += 1
    end
    # for k in 1:gcm.p
    #     H[idx] = gcm.Hβρ[k]
    #     idx += 1
    # end
    # lmul!(σ, H)
    H .*= σ
end

function MOI.hessian_lagrangian_structure(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel})
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    # Hββ block
    idx = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # rho and sigma2
    for j in 1:2
        arr1[idx] = gcm.p + j
        arr2[idx] = gcm.p + j
        idx += 1
    end
    arr1[idx] = gcm.p + 1
    arr2[idx] = gcm.p + 2
    idx += 1
    for k in 1:gcm.p
        arr1[idx] = gcm.p + 2
        arr2[idx] = k
        idx += 1
    end
    # for k in 1:gcm.p
    #     arr1[idx] = gcm.p + 1
    #     arr2[idx] = k
    #     idx += 1
    # end
    return collect(zip(arr1, arr2))
end

function MOI.eval_hessian_lagrangian(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel},
        H   :: AbstractVector{T},
        par :: AbstractVector{T},
        σ   :: T,
        μ   :: AbstractVector{T}
    ) where T
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, true)
    # Hβ block
    idx = 1
    @inbounds for j in 1:gcm.p, i in 1:j
        H[idx] = gcm.Hβ[i, j]
        idx += 1
    end
    # Haa block
    H[idx] = gcm.Hρ[1, 1]
    idx += 1
    H[idx] = gcm.Hσ2[1, 1]
    idx += 1
    H[idx] = gcm.Hρσ2[1, 1]
    idx += 1
    for k in 1:gcm.p
        H[idx] = gcm.Hβσ2[k]
        idx += 1
    end
    # for k in 1:gcm.p
    #     H[idx] = gcm.Hβρ[k]
    #     idx += 1
    # end
    # lmul!(σ, H)
    H .*= σ
end
