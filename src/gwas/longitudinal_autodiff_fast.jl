# par: all parameters being estimated, typically includes β, θ, nuisance params, and γ
# β: fixed effect of nongenetic covariates
# θ: variance component parameters
# γ: effect of a SNP

function loglikelihood(
    par::AbstractVector, # p+m+1 × 1 where m is number of VCs, 1 is for the SNP
    qc_model::GLMCopulaVCModel, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    )
    p = qc_model.p
    m = qc_model.m
    β = [par[1:end-(m+1)]; par[end]] # nongenetic + genetic beta
    θ = @view(par[end-m:end-1])             # vc parameters
    T = eltype(par)
    # allocate storage vectors of type T
    nmax = maximum(size(qc_model.data[i].X, 1) for i in 1:length(qc_model.data))
    η_store = zeros(T, nmax)
    μ_store = zeros(T, nmax)
    varμ_store = zeros(T, nmax)
    res_store = zeros(T, nmax)
    storage_n_store = zeros(T, nmax)
    Xstore = zeros(T, nmax, p+1)
    q = zeros(T, length(θ))
    logl = zero(T)
    for (i, gc) in enumerate(qc_model.data)
        n = size(gc.X, 1)
        X = @view(Xstore[1:n, :])
        η = @view(η_store[1:n])
        μ = @view(μ_store[1:n])
        varμ = @view(varμ_store[1:n])
        res = @view(res_store[1:n])
        storage_n = @view(storage_n_store[1:n])
        # sync nogenetic + genetic covariates
        copyto!(X, gc.X)
        X[:, end] .= z[i]
        y = gc.y
        # update_res! + standardize_res! step (need to avoid BLAS)
        A_mul_b!(η, X, β)
        for j in 1:gc.n
            μ[j] = GLM.linkinv(gc.link, η[j])
            varμ[j] = GLM.glmvar(gc.d, μ[j]) # Note: for negative binomial, d.r is used
            res[j] = (y[j] - μ[j]) / sqrt(varμ[j])
        end
        # update Γ
        for k in 1:gc.m
            A_mul_b!(storage_n, gc.V[k], res)
            q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        end
        # component_loglikelihood
        for j in 1:gc.n
            logl += ApproxCopula.loglik_obs(gc.d, y[j], μ[j], one(T), one(T))
        end
        tsum = dot(θ, gc.t)
        logl += -log(1 + tsum)
        qsum  = dot(θ, q) # qsum = 0.5 r'Γr
        logl += log(1 + qsum)
    end
    return logl
end

function GWASCopulaVCModel_autodiff_fast(
    qc_model::GLMCopulaVCModel,
    G::SnpArray;
    check_grad::Bool=true
    )
    # some needed constants
    p = qc_model.p
    m = qc_model.m
    n, q = size(G)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-2, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-2, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")
    # timers 
    grad_time = 0.0
    hess_time = 0.0
    # estimated parameters
    β = qc_model.β
    θ = qc_model.θ
    γ = zero(T)
    par = [β; θ; γ]
    # needed internal helper functions
    loglikelihood(par::AbstractVector) = ApproxCopula.loglikelihood(par, qc_model, z)
    loglikelihood(βθ, γ, qc_model, z) = ApproxCopula.loglikelihood([βθ; γ], qc_model, z)
    loglikelihood(γ::Number) = ApproxCopula.loglikelihood([β; θ; γ], qc_model, z)
    function hessian_column(par) # computes last column of Hessian
        function element_derivative(par)
            βθ = @view(par[1:end-1])
            γ  = par[end]
            return ForwardDiff.derivative(γ -> loglikelihood(βθ,γ,qc_model,z), γ)
        end
        ForwardDiff.gradient(element_derivative, par)
    end
    get_grad_last(γ) = ForwardDiff.derivative(loglikelihood, γ)
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]))
    Hfull = ForwardDiff.hessian(loglikelihood, par)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # storages
    pvals = zeros(T, q)
    W = zeros(T, p + m)
    # score test for each SNP
    @showprogress for j in 1:q
        # sync SNP values
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # compute W/Q/R
        grad_time += @elapsed R = get_grad_last(0.0)
        hess_time += @elapsed Hlast = hessian_column(par)
        Hlast .*= -1
        W .= @view(Hlast[1:end-1])
        Q = Hlast[end]
        # if j == 1
        #     fdsa
        # end
        S = R * inv(Q - dot(W, Pinv, W)) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    println("grad time = ", grad_time)
    println("hess time = ", hess_time)
    return pvals
end

function GWASCopulaVCModel_autodiff_fast(
    qc_model::GaussianCopulaVCModel,
    G::SnpArray;
    check_grad::Bool=true
    )
    # some needed constants
    p = qc_model.p
    m = qc_model.m
    n, q = size(G)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")
    # timers 
    grad_time = 0.0
    hess_time = 0.0
    # estimated parameters
    β = qc_model.β
    θ = qc_model.θ
    τ = qc_model.τ
    γ = zero(T)
    fullβ = [qc_model.β; qc_model.θ; qc_model.τ; 0.0]
    # needed internal helper functions
    loglikelihood(par::AbstractVector) = ApproxCopula.loglikelihood(par, qc_model, z)
    loglikelihood(βθτ, γ, qc_model, z) = ApproxCopula.loglikelihood([βθτ; γ], qc_model, z)
    loglikelihood(γ::Number) = ApproxCopula.loglikelihood([β; θ; τ; γ], qc_model, z)
    function hessian_column(par) # computes last column of Hessian
        function element_derivative(par)
            βθτ = @view(par[1:end-1])
            γ   = par[end]
            return ForwardDiff.derivative(γ -> loglikelihood(βθτ,γ,qc_model,z), γ)
        end
        ForwardDiff.gradient(element_derivative, par)
    end
    get_grad_last(γ) = ForwardDiff.derivative(loglikelihood, γ)
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=true)
    Hfull = ForwardDiff.hessian(loglikelihood, fullβ)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # storages
    pvals = zeros(T, q)
    W = zeros(T, p + m + 1)
    # score test for each SNP
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # compute W/Q/R
        grad_time += @elapsed R = get_grad_last(0.0)
        hess_time += @elapsed Hlast = hessian_column(fullβ)
        Hlast .*= -1
        W .= @view(Hlast[1:end-1])
        Q = Hlast[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    println("grad time = ", grad_time)
    println("hess time = ", hess_time)
    return pvals
end
