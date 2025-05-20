# par: all parameters being estimated, typically includes β, θ, nuisance params, and γ
# β: fixed effect of nongenetic covariates
# θ: variance component parameters
# γ: effect of a SNP

function loglikelihood_enzyme(
    par::AbstractVector, # p+m+1 × 1 where m is number of VCs, 1 is for the SNP
    qc_model::GLMCopulaVCModel, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    )
    # parameters
    T = eltype(par)
    m = qc_model.m
    β = @view(par[1:end-(m+1)]) # nongenetic effects
    θ = @view(par[end-m:end-1]) # vc parameters
    γ = par[end]                # genetic effects
    # compute loglikelihood
    logl = zero(T)
    for (i, qc) in enumerate(qc_model.data)
        # include SNP effect into mean
        η = qc.X * β .+ γ * z[i]
        # mul!(qc.η, qc.X, β)
        # qc.η .+= γ * z[i]

        # update_res! + standardize_res! step
        μ = GLM.linkinv.(qc.link, η)
        varμ = GLM.glmvar.(qc.d, μ)
        res = (qc.y .- μ) ./ sqrt.(varμ)
        # for j in 1:qc.n
        #     qc.μ[j] = GLM.linkinv(qc.link, qc.η[j])
        #     qc.varμ[j] = GLM.glmvar(qc.d, qc.μ[j]) # Note: for negative binomial, d.r is used
        #     qc.res[j] = (qc.y[j] - qc.μ[j]) / sqrt(qc.varμ[j])
        # end

        # component_loglikelihood
        for j in 1:qc.n
            logl += ApproxCopula.loglik_obs(qc.d, qc.y[j], μ[j], one(T), one(T))
        end

        # quadratic term
        qsum = zero(T)
        for k in 1:qc.m
            qsum += compute_rΓr(res, qc.V[k], 0.5θ[k])
        end
        # for k in 1:qc.m
        #     mul!(qc.storage_n, qc.V[k], qc.res)
        #     qc.q[k] = dot(qc.res, qc.storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        # end
        # qsum  = dot(θ, qc.q) # qsum = 0.5 r'Γr

        tsum = dot(θ, qc.t)
        logl += -log(1 + tsum)
        logl += log(1 + qsum)
    end
    return logl
end

# computes r' (θi .* Γ) r
function compute_rΓr(r::AbstractVector, Γ::AbstractMatrix, θ::AbstractFloat)
    p = length(r)
    p == size(Γ, 1) == size(Γ, 2) || error("Dimension of r and Γ mismatch")
    s = zero(eltype(r))
    for i in 1:p, j in 1:p
        s += θ * r[i] * Γ[i, j] * r[j]
    end
    return s
end

function GWASCopulaVCModel_enzyme(
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
    check_grad && any(x -> abs(x) > 1e-2, qc_model.∇β) && 
        error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-2, qc_model.∇θ) && 
        error("Null model gradient of variance components is not zero!")
    # needed internal helper functions for autodiffing loglikelihood 
    loglikelihood(par::AbstractVector) = ApproxCopula.loglikelihood_enzyme(par, qc_model, z)
    # loglikelihood(βθ, γ, qc_model, z) = ApproxCopula.loglikelihood([βθ; γ], qc_model, z)
    loglikelihood(γ::Number) = ApproxCopula.loglikelihood_enzyme([β; θ; γ], qc_model, z)
    # function hessian_column(par) # computes last column of Hessian
    #     function element_derivative(par)
    #         βθ = @view(par[1:end-1])
    #         γ  = par[end]
    #         return ForwardDiff.derivative(γ -> loglikelihood(βθ,γ,qc_model,z), γ)
    #     end
    #     ForwardDiff.gradient(element_derivative, par)
    # end
    # get_grad_last(γ) = ForwardDiff.derivative(loglikelihood, γ)
    # timers 
    grad_time = 0.0
    hess_time = 0.0
    # estimated parameters
    β = qc_model.β
    θ = qc_model.θ
    γ = zero(T)
    par = [β; θ; γ]
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]))

    @show loglikelihood(par)

    # Zygote.jl
    zygote_grad = Zygote.gradient(loglikelihood, γ)
    @show zygote_grad
    @time Zygote.gradient(loglikelihood, γ)
    fdsa

    # Enzyme.jl
    # dγ = zero(eltype(γ))
    # Enzyme.autodiff(Reverse, loglikelihood, Active, Duplicated(γ, dγ))
    # @show dγ # doesn't work
    # fdsa

    Hfull = Enzyme.hessian(loglikelihood_all_params, par)
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
        # score test
        S = R * inv(Q - dot(W, Pinv, W)) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    println("grad time = ", grad_time)
    println("hess time = ", hess_time)
    return pvals
end
