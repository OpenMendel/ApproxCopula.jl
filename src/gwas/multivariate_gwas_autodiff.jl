struct gwas_result
    description::String # how was p-values produced
    logl_H0::Float64 # loglikelihood of null model
    p::Int # number of SNPs
    logl_Ha::Vector{Float64} # loglikelihood of alt model, one value for each SNP
    pvals::Vector{Float64} # p-value for each SNP
    Rs::Vector{Float64} # gradient of SNP under null model
end

function autodiff_loglikelihood(
    null_β::Vector, # first p*d are β, next m are for vech(L), next s are for nuisance
    snp_β::Vector, # this should be zeros(d)
    z::Vector, # SNP values for each sample (length n*1)
    data::MultivariateCopulaData 
    )
    n, p, m, d = data.n, data.p, data.m, data.d

    # allocate everything for now
    B = zeros(p, d)
    copyto!(B, 1, null_β, 1, p * d)
    L = cholesky(Symmetric(zeros(d, d), :L), check=false)
    un_vech!(L, @view(null_β[p * d + 1:p * d + m]))
    ϕ = null_β[p * d + m + 1:end]
    std_res = zeros(d)
    η = zeros(n, d)
    storage_d = zeros(d)

    # update η
    mul!(η, data.X, B)
    for i in 1:n
        η[i, :] .+= z[i] .* snp_β
    end

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

function multivariateGWAS_adhoc_lrt(
    qc_model::MultivariateCopulaModel,
    G::SnpArray;
    check_grad::Bool = true,
    pval_cutoff::Float64 = 0.05 / (0.1*size(G, 2)), # terminates LRT when p-value is above this
    aggregate_function::Function = x -> sum(abs, x) / length(x), # function that operates on SNP betas
    alt_model_max_iter::Int = 10000,
    alt_model_backtrack_steps::Int = 50,
    )
    # some needed constants
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p, d = size(qc_model.B)    # dimension of fixed effects in each sample
    T = eltype(qc_model.data.X)
    n == qc_model.data.n || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-2, qc_model.grad) && 
        error("Null model gradient is not zero!")

    # full beta and logl under the null 
    nullβ = [vec(qc_model.B); qc_model.vechL; qc_model.ϕ]
    logl_H0 = loglikelihood!(nullβ, qc_model.data)

    # estimate grad of null for each SNP
    pvals = zeros(T, q)
    Rstore = zeros(T, d)
    z = zeros(T, n)
    Rs = zeros(T, q)
    @showprogress "Estimating grad under null" for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=true, impute=true)
        any(zi -> isnan(zi) || isinf(zi), z) && error("SNP $j has nan or inf!")

        # compute grad of SNP effect under null
        snp_β = zeros(d)
        Rstore .= 0
        GC.enable(false)
        Enzyme.autodiff(
            Reverse, autodiff_loglikelihood,
            Const(nullβ), Duplicated(snp_β, Rstore), 
            Const(z), Const(qc_model.data),
        )
        GC.enable(true)

        # store magnitude of grad under null
        Rs[j] = aggregate_function(Rstore)
    end

    # run LRT for top SNPs
    perm = sortperm(Rs, rev=true)
    Xfull = hcat(qc_model.data.X, zeros(n))
    pvals = ones(q)
    logl_Ha = fill(-Inf, q)
    prog = Progress(q, desc="Running LRT, termination when p > $pval_cutoff")
    for j in perm
        # append SNP to Xfull
        SnpArrays.copyto!(@view(Xfull[:, end]), @view(G[:, j]), center=true, 
            scale=true, impute=true)

        # fit alternative model
        logl_Ha[j] = refit(qc_model, Xfull, verbose=false,
            solver_config = 
            Dict("print_level"               => 0, 
                "tol"                        => 10^-3,
                "max_iter"                   => alt_model_max_iter,
                "accept_after_max_steps"     => alt_model_backtrack_steps,
                "warm_start_init_point"      => "yes", 
                "limited_memory_max_history" => 6, # default value
                "hessian_approximation"      => "limited-memory",
                )
        )

        # lrt
        ts = -2(logl_H0 - logl_Ha[j])
        pvals[j] = ccdf(Chisq(d), ts)
        0 ≤ pvals[j] ≤ 1 || error("SNP j has pval $(pvals[j]), shouldn't happen!")
        next!(prog)
        pvals[j] > pval_cutoff && break
    end
    finish!(prog)

    return gwas_result("Adhoc likelihood ratio tests",
        logl_H0, p, logl_Ha, pvals, Rs
    )
end

function refit(
    qc_model::MultivariateCopulaModel{T}, # null model
    X::Matrix{T}, # data with SNP augmented in last column
    solver :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config :: Dict = 
        Dict("print_level"                => 0, 
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
    qcm = MultivariateCopulaModel(
        qc_model.data.Y, X, qc_model.data.vecdist, qc_model.data.veclink)
    return fit!(qcm, solver, solver_config=solver_config, verbose=verbose)
end

function multivariateGWAS_autodiff(
    qc_model::MultivariateCopulaVCModel,
    G::SnpArray;
    check_grad::Bool = true
    )
    # some needed constants
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p, d = size(qc_model.B)    # dimension of fixed effects in each sample
    m = length(qc_model.θ)     # number of variance components in each sample
    s = count(x -> typeof(x) <: Normal, qc_model.vecdist) # number of nuisance parameters (only Gaussian for now)
    T = eltype(qc_model.X)
    n == length(qc_model.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-3, qc_model.∇vecB) && 
        error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-3, qc_model.∇θ) && 
        error("Null model gradient of variance components is not zero!")

    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(par) = loglikelihood(par, qc_model, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    ∇logl! = (grad, x) -> ForwardDiff.gradient!(grad, autodiff_loglikelihood, x)
    ∇²logl! = (hess, x) -> ForwardDiff.hessian!(hess, autodiff_loglikelihood, x)

    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 2]), center=true, scale=true, impute=true)
    fullβ = [vec(qc_model.B); qc_model.θ; qc_model.ϕ; zeros(d)]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-d, 1:end-d])

    # score test for each SNP
    pvals = zeros(T, q)
    grad_store = zeros(T, p*d + m + s + d)
    W = zeros(T, p*d + m + s, d)
    Q = zeros(T, d, d)
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=true, impute=true)

        # compute W/Q/R using in-place versions of ForwardDiff grad/hess
        ∇²logl!(Hfull, fullβ)
        copyto!(W, @view(Hfull[1:end-d, end-d+1:end]))
        W .*= -1
        copyto!(Q, @view(Hfull[end-d+1:end, end-d+1:end]))
        Q .*= -1
        ∇logl!(grad_store, fullβ)
        R = grad_store[end-d+1:end]

        # compute W/Q/R using not-inplace version of ForwardDiff grad/hess
        # Hfull = ∇²logl(fullβ)
        # W = -Hfull[1:p*d+m, end-d+1:end]
        # Q = -Hfull[end-d+1:end, end-d+1:end]
        # R = ∇logl(fullβ)[end-d+1:end]

        S = R' * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(d), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    return pvals
end

function loglikelihood(
    par::AbstractVector{T}, # length pd+m+s+d. m is num of VCs, s is num of nuisance params, d is SNP effect on d phenotypes
    qc_model::MultivariateCopulaVCModel, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    ) where T
    n = length(qc_model.data)
    n == length(z) || error("Expected n == length(z)")

    # parameters
    p, d = size(qc_model.B)
    m = qc_model.m                     # number of variance components
    s = qc_model.s                     # number of nuisance parameters 
    B = reshape(par[1:p*d], p, d)      # nongenetic covariates
    θ = par[p*d+1:p*d+m]               # vc parameters
    τ = par[p*d+m+1:p*d+m+s]           # nuisance parameters
    γ = par[end-d+1:end]               # genetic beta

    # storages friendly to autodiff
    ηstore = zeros(T, d)
    μstore = zeros(T, d)
    varμstore = zeros(T, d)
    resstore = zeros(T, d)
    std_resstore = zeros(T, d)
    storage_d = zeros(T, d)
    qstore = zeros(T, m)

    logl = 0.0
    for i in 1:n
        # data for sample i
        xi = @view(qc_model.X[i, :])
        yi = @view(qc_model.Y[i, :])
        # update η, μ, res, to include effect of SNP
        At_mul_b!(ηstore, B, xi)
        ηstore .+= γ .* z[i]
        μstore .= GLM.linkinv.(qc_model.veclink, ηstore)
        varμstore .= GLM.glmvar.(qc_model.vecdist, μstore)
        resstore .= yi .- μstore
        # update std_res (gaussian case needs separate treatment)
        nuisance_counter = 1
        for j in eachindex(std_resstore)
            if typeof(qc_model.vecdist[j]) <: Normal
                τj = abs(τ[nuisance_counter])
                std_resstore[j] = resstore[j] * sqrt(τj)
                nuisance_counter += 1
            else
                std_resstore[j] = resstore[j] / sqrt(varμstore[j])
            end
        end
        # GLM loglikelihood (term 2)
        nuisance_counter = 1
        for j in eachindex(yi)
            dist = qc_model.vecdist[j]
            if typeof(dist) <: Normal
                τj = inv(τ[nuisance_counter])
                logl += QuasiCopula.loglik_obs(dist, yi[j], μstore[j], one(T), τj)
                nuisance_counter += 1
            else
                logl += QuasiCopula.loglik_obs(dist, yi[j], μstore[j], one(T), one(T))
            end
        end
        # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
        tsum = dot(θ, qc_model.t) # tsum = 0.5tr(Γ)
        logl += -log(1 + tsum)
        # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
        @inbounds for k in 1:qc_model.m # loop over m variance components
            mul!(storage_d, qc_model.V[k], std_resstore) # storage_d = V[k] * r
            qstore[k] = dot(std_resstore, storage_d) / 2 # q[k] = 0.5 r * V[k] * r
        end
        qsum = dot(θ, qstore) # qsum = 0.5 r*Γ*r
        logl += log(1 + qsum)
    end

    return logl
end

function At_mul_b!(c::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    n, p = size(A)
    fill!(c, zero(T))
    for j in 1:p, i in 1:n
        c[j] += A[i, j] * b[i]
    end
    return c
end
