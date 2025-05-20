# function loglikelihood(
#     par::AbstractVector{T}, # p+m+1 × 1 where m is number of VCs, 1 is for the SNP
#     qc_model::GLMCopulaVCModel, # fitted null model
#     z::AbstractVector # n × 1 genotype vector
#     ) where T
#     p = qc_model.p
#     m = qc_model.m
#     β = [par[1:end-(m+1)]; par[end]] # nongenetic + genetic beta
#     θ = par[end-m:end-1]             # vc parameters
#     # allocate storage vectors of type T
#     nmax = maximum(size(qc_model.data[i].X, 1) for i in 1:length(qc_model.data))
#     η_store = zeros(T, nmax)
#     μ_store = zeros(T, nmax)
#     varμ_store = zeros(T, nmax)
#     res_store = zeros(T, nmax)
#     storage_n_store = zeros(T, nmax)
#     Xstore = zeros(T, nmax, p+1)
#     q = zeros(T, length(θ))
#     logl = zero(T)
#     for (i, gc) in enumerate(qc_model.data)
#         n = size(gc.X, 1)
#         X = @view(Xstore[1:n, :])
#         η = @view(η_store[1:n])
#         μ = @view(μ_store[1:n])
#         varμ = @view(varμ_store[1:n])
#         res = @view(res_store[1:n])
#         storage_n = @view(storage_n_store[1:n])
#         # sync nogenetic + genetic covariates
#         copyto!(X, gc.X)
#         X[:, end] .= z[i]
#         y = gc.y
#         # update_res! step (need to avoid BLAS)
#         A_mul_b!(η, X, β)
#         for j in 1:gc.n
#             μ[j] = GLM.linkinv(gc.link, η[j])
#             varμ[j] = GLM.glmvar(gc.d, μ[j]) # Note: for negative binomial, d.r is used
#             # dμ[j] = GLM.mueta(gc.link, η[j])
#             # w1[j] = dμ[j] / varμ[j]
#             # w2[j] = w1[j] * dμ[j]
#             res[j] = y[j] - μ[j]
#         end
#         # standardize_res! step
#         for j in eachindex(y)
#             res[j] /= sqrt(varμ[j])
#         end
#         # std_res_differential! step (this will compute ∇resβ)
#         # for i in 1:gc.p
#         #     for j in 1:gc.n
#         #         ∇resβ[j, i] = -sqrt(varμ[j]) * X[j, i] - (0.5 * res[j] * (1 - (2 * μ[j])) * X[j, i])
#         #     end
#         # end
#         # update Γ
#         @inbounds for k in 1:gc.m
#             A_mul_b!(storage_n, gc.V[k], res)
#             q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
#         end
#         # component_loglikelihood
#         for j in 1:gc.n
#             logl += ApproxCopula.loglik_obs(gc.d, y[j], μ[j], one(T), one(T))
#         end
#         tsum = dot(θ, gc.t)
#         logl += -log(1 + tsum)
#         qsum  = dot(θ, q) # qsum = 0.5 r'Γr
#         logl += log(1 + qsum)
#     end
#     return logl
# end

function loglikelihood(
    par::AbstractVector{T}, # p+m+1 × 1 where m is number of VCs, 1 is for r, 1 is for the SNP
    qc_model::NBCopulaVCModel, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    ) where T
    p = qc_model.p
    m = gcm.m
    β = [par[1:end-(m+2)]; par[end]] # nongenetic + genetic beta
    θ = par[end-(m+1):end-2]         # vc parameters
    r = par[end-1]                   # dispersion for gaussian
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
        # update_res! step (need to avoid BLAS)
        A_mul_b!(η, X, β)
        for j in 1:gc.n
            μ[j] = GLM.linkinv(gc.link, η[j])
            varμ[j] = GLM.glmvar(gc.d, μ[j]) # Note: for negative binomial, d.r is used
            res[j] = y[j] - μ[j]
        end
        # standardize_res! step
        for j in eachindex(y)
            res[j] /= sqrt(varμ[j])
        end
        # update Γ
        @inbounds for k in 1:gc.m
            A_mul_b!(storage_n, gc.V[k], res)
            q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        end
        # component_loglikelihood
        for j in 1:gc.n
            logl += logpdf(NegativeBinomial(r, r / (μ[j] + r)), y[j])
        end
        tsum = dot(θ, gc.t)
        logl += -log(1 + tsum)
        qsum  = dot(θ, q) # qsum = 0.5 r'Γr
        logl += log(1 + qsum)
    end
    return logl
end

function loglikelihood(
    par::AbstractVector{T}, # p+m+1+1, where m is number of VCs, 1 is τ, and last 1 is for the SNP
    gcm::GaussianCopulaVCModel,
    z::AbstractVector # n × 1 genotype vector
    ) where T
    p = gcm.p
    m = gcm.m
    β = [par[1:end-(m+2)]; par[end]] # nongenetic + genetic beta
    θ = par[end-(m+1):end-2]         # vc parameters
    τ = par[end-1]                   # dispersion for gaussian
    # allocate vector of type T
    nmax = maximum(size(gcm.data[i].X, 1) for i in 1:length(gcm.data))
    μ_store = zeros(T, nmax)
    res_store = zeros(T, nmax)
    storage_n_store = zeros(T, nmax)
    Xstore = zeros(T, nmax, p+1)
    q = zeros(T, length(θ))
    logl = zero(T)
    for (i, gc) in enumerate(gcm.data)
        n = size(gc.X, 1)
        X = @view(Xstore[1:n, :])
        μ = @view(μ_store[1:n])
        res = @view(res_store[1:n])
        storage_n = @view(storage_n_store[1:n])
        # sync nogenetic + genetic covariates
        copyto!(X, gc.X)
        X[:, end] .= z[i]
        y = gc.y
        sqrtτ = sqrt(abs(τ))
        # update_res! step (need to avoid BLAS)
        A_mul_b!(μ, X, β)
        for j in 1:gc.n
            res[j] = y[j] - μ[j]
        end
        # standardize_res! step
        res .*= sqrtτ
        rss  = abs2(norm(res)) # RSS of standardized residual
        tsum = dot(abs.(θ), gc.t)
        logl += - log(1 + tsum) - (gc.n * log(2π) -  gc.n * log(abs(τ)) + rss) / 2
        # update Γ
        @inbounds for k in 1:gc.m
            A_mul_b!(storage_n, gc.V[k], res)
            q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        end
        qsum  = dot(θ, q)
        logl += log(1 + qsum)
    end
    return logl
end

# Matrix-vector multiply friendly to autodiffing
function A_mul_b!(c::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    n, p = size(A)
    fill!(c, zero(T))
    for j in 1:p, i in 1:n
        c[i] += A[i, j] * b[j]
    end
    return c
end

function GWASCopulaVCModel_autodiff(
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
    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(β) = loglikelihood(β, qc_model, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    ∇logl! = (grad, x) -> ForwardDiff.gradient!(grad, autodiff_loglikelihood, x)
    ∇²logl! = (hess, x) -> ForwardDiff.hessian!(hess, autodiff_loglikelihood, x)
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=true)
    fullβ = [qc_model.β; qc_model.θ; 0.0]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # score test for each SNP
    pvals = zeros(T, q)
    grad_store = zeros(T, p + m + 1)
    W = zeros(T, p + m)
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # compute W/Q/R using in-place versions of ForwardDiff grad/hess
        ∇²logl!(Hfull, fullβ)
        for i in eachindex(W)
            W[i] = -Hfull[i, end]
        end
        Q = -Hfull[end, end]
        ∇logl!(grad_store, fullβ)
        R = grad_store[end]
        # compute W/Q/R using not-inplace version of ForwardDiff grad/hess
        # Hfull = ∇²logl(fullβ)
        # W = -Hfull[1:end-1, end]
        # Q = -Hfull[end, end]
        # R = ∇logl(fullβ)[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    return pvals
end

function GWASCopulaVCModel_autodiff(
    qc_model::NBCopulaVCModel,
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
    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(β) = loglikelihood(β, qc_model, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=true)
    fullβ = [qc_model.β; qc_model.θ; qc_model.r; 0.0]
    Hfull = ∇²logl(fullβ)
    @show size(Hfull, 1)
    @show rank(Hfull)
    @show rank(Hfull[1:end-1, 1:end-1])
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # score test for each SNP
    pvals = zeros(T, q)
    W = zeros(T, p + m)
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # compute W/Q/R
        Hfull = ∇²logl(fullβ)
        W = -Hfull[1:end-1, end]
        Q = -Hfull[end, end]
        R = ∇logl(fullβ)[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
        @show R
        @show Q
        @show W
        @show S
        if j == 1
            fdsa
        end
    end
    return pvals
end

function GWASCopulaVCModel_autodiff(
    gcm::GaussianCopulaVCModel,
    G::SnpArray;
    check_grad::Bool=true
    )
    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(β) = loglikelihood(β, gcm, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    # some needed constants
    n, q = size(G)
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-1, gcm.∇β) && error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-1, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=true)
    fullβ = [gcm.β; gcm.θ; gcm.τ; 0.0]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # score test for each SNP
    pvals = zeros(T, q)
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # compute W/Q/R
        Hfull = ∇²logl(fullβ)
        W = -Hfull[1:end-1, end]
        Q = -Hfull[end, end]
        R = ∇logl(fullβ)[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
        # @show Pinv
        # @show R
        # @show Q
        # @show W
        # @show S
        # j == 1 && fdsa

    end
    return pvals
end
