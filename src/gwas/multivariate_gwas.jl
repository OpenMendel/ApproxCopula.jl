"""
    GWASCopulaVCModel(qc_model::MixedCopuMultivariateCopulaVCModellaVCModel, G::SnpArray)

Performs score tests for each SNP in `G`, given a fitted (null) model on the 
non-genetic covariates.

# Inputs
+ `qc_model`: A fitted `GaussianCopulaVCModel`, `GLMCopulaVCModel` or `NBCopulaVCModel` 
    that includes `n` sample points and `p` non-genetic covariates.
+ `G`: A `SnpArray` (compressed `.bed/bim/fam` PLINK file) with `n` samples and 
    `q` SNPs

# Outputs
A length `q` vector of p-values storing the score statistics for each SNP
"""
function GWASCopulaVCModel(
    qc_model::MultivariateCopulaVCModel,
    G::SnpArray;
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p, d = size(B)    # dimension of fixed effects in each sample
    m = length(qc_model.θ) # number of variance components in each sample
    s = 0 # number of nuisance parameters (todo)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    # any(x -> abs(x) > 1e-2, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    # any(x -> abs(x) > 1e-2, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")

    # preallocated arrays for efficiency
    z = zeros(T, n)
    W = zeros(T, p*d + m + s)
    χ2 = Chisq(d)
    pvals = zeros(T, q)
    Wtime = 0.0
    Qtime = 0.0
    Rtime = 0.0
    Pinv = get_Pinv(qc_model)

    # score test for each SNP
    for j in 1:q
        # # sync vectors
        # SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        # Q, R = zero(T), zero(T)
        # fill!(W, zero(T))
        # # loop over each sample
        # for i in 1:n
        #     # variables for current sample
        #     qc = qc_model.data[i]
        #     d = qc.n # number of observations for current sample
        #     fill!(zi, z[i])
        #     # update gradient of residual with respect to β and γ
        #     ∇resγ = get_∇resγ(qc_model, i, @view(zi[1:d])) # d × 1
        #     ∇resβ = get_∇resβ(qc_model, i) # d × p
        #     # form some constants 
        #     Γ = zeros(T, d, d)
        #     for k in 1:qc.m # loop over variance components
        #         Γ .+= qc_model.θ[k] .* qc.V[k]
        #     end
        #     denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
        #     denom2 = abs2(denom)
        #     storage.denom[1] = denom
        #     storage.denom2[1] = denom2
        #     # calculate W
        #     Wtime += @elapsed update_W!(W, qc_model, i, @view(zi[1:d]), Γ, ∇resβ, ∇resγ, storage)
        #     # update Q
        #     Qtime += @elapsed Q += calculate_Qi(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom, denom2)
        #     # update R
        #     Rtime += @elapsed R += calculate_Ri(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom)
        # end
        # # score test
        # # @show R
        # # @show Q
        # # @show W
        # # if j == 1
        # #     fdsa
        # # end
        # S = R * inv(Q - dot(W, Pinv, W)) * R
        # pvals[j] = ccdf(χ2, S)
    end
    @show Wtime
    @show Qtime
    @show Rtime
    return pvals
end

function get_Pinv(qc_model::MultivariateCopulaVCModel)
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    return inv(P)
end

function get_Hββ(qc_model::MultivariateCopulaVCModel)
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)    
    # loop over samples
    for (i, qc) in enumerate(qc_model.data)
        d = qc.n # number of observations for current sample
        # GLM term
        H -= Transpose(qc.X) * Diagonal(qc.w2) * qc.X
        # 2nd term
        res = qc.res # d × 1 standardized residuals
        ∇resβ = qc.∇resβ # d × p
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = qc.η
        μ = qc.μ
        varμ = qc.varμ
        res = qc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = qc.X[j, :]
            dist = typeof(qc_model.d[i]) <: NegativeBinomial ? NegativeBinomial(qc_model.r[1]) : qc_model.d[i]
            link = qc_model.link[i]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
        # @show qc.w2
        # @show res
        # @show η
        # @show μ
        # @show varμ
        # j = 1
        # @show dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])
        # fdsa
    end
    return H
end

function get_Hθθ(qc_model::MultivariateCopulaVCModel)
    # use loglikelihood! function to get Hθθ. Commented out code gives the same answer
    loglikelihood!(qc_model, true, true)
    return qc_model.Hθ
    # m = length(qc_model.data[1].V) # number of variance components
    # hess_math = zeros(m, m)
    # for i in eachindex(qc_model.data)
    #     r = qc_model.data[i].res
    #     Ω = qc_model.data[i].V
    #     b = [0.5r' * Ω[k] * r for k in 1:m]
    #     c = [0.5tr(Ω[k]) for k in 1:m]
    #     hess_math += b*b' / (1 + qc_model.θ'*b)^2 - c*c' / (1 + qc_model.θ'*c)^2
    # end
    # return -hess_math
end

function get_Hβθ(qc_model::MultivariateCopulaVCModel)
    m = length(qc_model.data[1].V)  # number of variance components
    p = size(qc_model.data[1].X, 2) # number of fixed effects
    hess_math = zeros(p, m)
    for i in eachindex(qc_model.data)
        r = qc_model.data[i].res
        Ω = qc_model.data[i].V
        θ = qc_model.θ
        ∇resβ = qc_model.data[i].∇resβ
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end
