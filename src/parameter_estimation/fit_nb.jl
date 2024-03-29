"""
    fit!(gcm::NBCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaVCModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.θ`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.θ` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
        gcm::NBCopulaVCModel,
        solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
        solver_config :: Dict = 
            Dict("print_level"                => 0, 
                 "tol"                        => 10^-3,
                 "max_iter"                   => 15,
                 "accept_after_max_steps"     => 50,
                 "warm_start_init_point"      => "yes", 
                 "limited_memory_max_history" => 6, # default value
                 "hessian_approximation"      => "limited-memory",
                #  "derivative_test"            => "first-order",
                 ),
        tol::Float64 = 1e-6,
        maxBlockIter::Int=10
    )
    T = eltype(gcm.β)
    solvertype = typeof(solver)
    solvertype <: Ipopt.Optimizer ||
        @warn("Optimizer object is $solvertype, `solver_config` may need to be defined.")

    # Pass options to solver
    config_solver(solver, solver_config)

    # initial conditions
    initialize_model!(gcm)
    npar = gcm.p + gcm.m
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
    logl0 = MOI.eval_objective(gcm, par0)

    # optimize
    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    for i in 1:maxBlockIter
        MOI.optimize!(solver)
        modelpar_to_optimpar!(par0, gcm)
        logl = MOI.eval_objective(gcm, par0)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
        # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            logl0 = logl
        end
    end

    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MOI.get(solver, MOI.VariablePrimal(), solver_pars))
    loglikelihood!(gcm, true, false)
end

"""
    fit!(gcm::NBCopulaARModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaARModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.ρ`, `gcm.σ2` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaARModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
    gcm::NBCopulaARModel,
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 0, 
             "tol"                        => 10^-3,
             "max_iter"                   => 15,
             "accept_after_max_steps"     => 50,
             "warm_start_init_point"      => "yes", 
             "limited_memory_max_history" => 6, # default value
             "hessian_approximation"      => "limited-memory",
            #  "derivative_test"            => "first-order",
             ),
    tol::Float64 = 1e-6,
    maxBlockIter::Int=10
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
    logl0 = MOI.eval_objective(gcm, par0)

    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    # optimize
    for i in 1:maxBlockIter
        MOI.optimize!(solver)
        modelpar_to_optimpar!(par0, gcm)
        logl = MOI.eval_objective(gcm, par0)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
            # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            logl0 = logl
        end
    end

    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MOI.get(solver, MOI.VariablePrimal(), solver_pars))
    loglikelihood!(gcm, true, false)
end

"""
    fit!(gcm::NBCopulaCSModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaCSModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.ρ`, `gcm.σ2` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaCSModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
    gcm::NBCopulaCSModel,
    solver=Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes", mu_strategy = "adaptive",
                             hessian_approximation = "limited-memory");
    tol::Float64 = 1e-6,
    maxBlockIter::Int=10
    )
    initialize_model!(gcm)
    npar = gcm.p + 2 # rho and sigma squared
    optm = MOI.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill(Inf, npar)
    offset = gcm.p + 1
    # rho
    ub[offset] = 1
    # lb[offset] = 0
    lb[offset] = -inv(gcm.data[1].n - 1)
    offset += 1
    # sigma2
    lb[offset] = 0
    MOI.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, gcm)
    MOI.setwarmstart!(optm, par0)
    logl0 = MOI.getobjval(optm)
    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    # optimize
    for i in 1:maxBlockIter
        MOI.optimize!(optm)
        logl = MOI.getobjval(optm)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
            # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            logl0 = logl
        end
    end
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MOI.getsolution(optm))
    loglikelihood!(gcm, true, false)
    # gcm
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: NBCopulaVCModel
    )
    # β
    copyto!(par, gcm.β)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.θ[k]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: NBCopulaVCModel,
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset   += 1
    end
    gcm
end

function MOI.initialize(
    gcm::NBCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(qc_model::NBCopulaVCModel) = [:Grad, :Hess]

function MOI.eval_objective(
        gcm :: NBCopulaVCModel,
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
        gcm  :: NBCopulaVCModel,
        grad :: Vector,
        par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        grad[offset] = gcm.∇θ[k]
        offset += 1
    end
    obj
end

function MOI.hessian_lagrangian_structure(gcm::NBCopulaVCModel)
    m◺ = ◺(gcm.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + m◺)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + m◺)
    # Hββ block
    idx = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Haa block
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    return collect(zip(arr1, arr2))
end

function MOI.eval_hessian_lagrangian(
        gcm :: NBCopulaVCModel,
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
    @inbounds for j in 1:gcm.m, i in 1:j
        H[idx] = gcm.Hθ[i, j]
        idx += 1
    end
    # lmul!(σ, H)
    H .*= σ
end
