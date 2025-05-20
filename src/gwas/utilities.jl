"""
    ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector{T}, T}, μj::T, ηj::T)
    ∇²σ²_j!(storage::AbstractMatrix{T}, d::Distribution, l::Link, xj::AbstractVector{T}, μj::T, ηj::T, c::T)

Computes the Hessian of the σ^2 function with respect to β for sample i (Xi) 
at time j. The inplace versions is equivalent to `storage .+= c .* ∇²σ²_j` 

Note: sometimes `xj` is a vector (Hessian wrt to β) and sometimes a scalar 
(Hessian wrt to γ). 
"""
@inline function ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector{T}, T}, 
                μj::T, ηj::T) where T
    c = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    return c * xj * xj' # p × p or 1 × 1
end
@inline function ∇²σ²_j!(storage::AbstractMatrix{T}, d::Distribution, l::Link, 
                xj::AbstractVector{T}, μj::T, ηj::T, c::T) where T
    c2 = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    BLAS.gemm!('N', 'T', c * c2, xj, xj, one(T), storage) # storage = p × p
end

"""
    ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector{T}, T}, zj::T, μj::T, ηj::T)
    ∇²σ²_j!(storage::AbstractVector{T}, d::Distribution, l::Link, xj::AbstractVector{T}, zj::T, μj::T, ηj::T, c::T)

Computes the Hessian of the mean function with respect to β then γ. The inplace
versions is equivalent to `storage .+= c .* ∇²σ²_j` 

Note: sometimes `xj` is a vector (Hessian wrt to β) and sometimes a scalar 
(Hessian wrt to γ). 
"""
@inline function ∇²σ²_j(d::Distribution, l::Link, xj::Union{AbstractVector{T}, T}, 
    zj::T, μj::T, ηj::T) where T
    c = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    return c * zj * xj # p × 1
end
@inline function ∇²σ²_j!(storage::AbstractVector{T}, d::Distribution, l::Link, 
                xj::AbstractVector{T}, zj::T, μj::T, ηj::T, c::T) where T
    c2 = sigmamu2(d, μj)*GLM.mueta(l, ηj)^2 + sigmamu(d, μj)*mueta2(l, ηj)
    storage .+= (c * c2 * zj) .* xj # p × 1
end

"""
    ∇²μ_j(l::Link, ηj::T, xj::Union{AbstractVector, T})
    ∇²μ_j!(storage, l::Link, ηj::T, xj::AbstractVector, c::T)

Computes the Hessian of the mean function with respect to β for sample i (Xi) 
at time j. The inplace version is equivalent to `storage .+= c .* ∇²μ_j` 

Note: sometimes `xj` is a vector (Hessian wrt to β) and sometimes a scalar 
(Hessian wrt to γ). 
"""
@inline function ∇²μ_j(l::Link, ηj::T, xj::Union{AbstractVector{T}, T}) where T
    d²μdη² = mueta2(l, ηj)
    return d²μdη² * xj * xj' # p × p or 1 × 1 (depending on length of xj)
end
@inline function ∇²μ_j!(storage::AbstractMatrix{T}, l::Link, ηj::T, 
    xj::AbstractVector, c::T
    ) where T # call ∇²μ_j if xj is scalar
    d²μdη² = mueta2(l, ηj)
    BLAS.gemm!('N', 'T', c * d²μdη², xj, xj, one(T), storage) # storage = p × p
end

"""
    ∇²μ_j(l::Link, ηj::T, xj::Union{AbstractVector, T}, zj::T)
    ∇²μ_j!(storage, l::Link, ηj::T, xj::AbstractVector, zj::T, c::T)

Computes the Hessian of the mean function with respect to β then γ. The inplace
version is equivalent to `storage .+= c .* ∇²μ_j` 

Note: sometimes `xj` is a vector (in which case we compute Hβγ) and sometimes 
a scalar (in which case we compute Hγγ). 
"""
@inline function ∇²μ_j(l::Link, ηj::T, xj::Union{AbstractVector{T}, T}, zj::T) where T
    d²μdη² = mueta2(l, ηj)
    return d²μdη² * zj * xj # p × 1 or 1 × 1
end
@inline function ∇²μ_j!(storage::AbstractVector{T}, l::Link, ηj::T, 
                xj::AbstractVector{T}, zj::Number, c::T
                ) where T # call ∇²μ_j if xj::Number
    d²μdη² = mueta2(l, ηj)
    storage .+= (c * d²μdη² * zj) .* xj # p × 1
end

"""
    sigmaμ2(D::Distribution, μ::Real)

Computes d²σ²/dμ²
"""
function sigmamu2 end
@inline sigmamu2(::Normal, μ::Real) = zero(μ)
@inline sigmamu2(::Bernoulli, μ::Real) = -2
@inline sigmamu2(::Poisson, μ::Real) = zero(μ)
@inline sigmamu2(d::NegativeBinomial, μ::Real) = 2/d.r + 1

"""
    sigmaeta(D::Distribution, μ::Real)

Computes dσ²/dμ
"""
function sigmamu end
@inline sigmamu(::Normal, μ::Real) = zero(μ)
@inline sigmamu(::Bernoulli, μ::Real) = one(μ) - 2μ
@inline sigmamu(::Poisson, μ::Real) = one(μ)
@inline sigmamu(d::NegativeBinomial, μ::Real) = 2μ/d.r + 1

"""
    mueta2(l::Link, η::Real)

Second derivative of the inverse link function `d^2μ/dη^2`, for link `L` at linear predictor value `η`.
I.e. derivative of the mueta function in GLM.jl
"""
function mueta2 end
@inline mueta2(::IdentityLink, η::Real) = zero(η)
@inline function mueta2(::LogitLink, η::Real)
    # expabs = exp(-abs(η))
    expabs = exp(-η)
    denom = 1 + expabs
    return -expabs / denom^2 + 2expabs^2 / denom^3
end
@inline mueta2(::LogLink, η::Real) = exp(η)

"""
    dβdβ_res_ij(dist, link, xj, η_j, μ_j, varμ_j, res_j)

Computes the Hessian of the standardized residuals for sample i 
at the j'th measurement, the gradient is evaluate with respect to β (or γ) twice.
"""
@inline function dβdβ_res_ij(dist, link, xj, η_j, μ_j, varμ_j, res_j)
    invσ_j = inv(sqrt(varμ_j))
    ∇μ_ij  = GLM.mueta(link, η_j) * xj
    ∇σ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * xj

    # assemble 5 terms
    term1 = -invσ_j * ∇²μ_j(link, η_j, xj)
    term2 = 0.5invσ_j^3 * ∇σ²_ij * ∇μ_ij'
    term3 = -0.5 * res_j * inv(varμ_j) * ∇²σ²_j(dist, link, xj, μ_j, η_j)
    term4 = 0.5invσ_j^3 * ∇μ_ij * ∇σ²_ij'
    term5 = 0.75res_j * inv(varμ_j^2) * ∇σ²_ij * ∇σ²_ij'
    result = term1 + term2 + term3 + term4 + term5

    return result # p × p or 1 × 1 depending on dimension of xj
end

"""
    dγdβresβ_ij(dist, link, xj, η_j, μ_j, varμ_j, res_j)
    dγdβresβ_ij!(storage, dist, link, xj, η_j, μ_j, varμ_j, res_j)

Computes the Hessian of the standardized residuals for sample i 
at the j'th measurement, first gradient respect to β, then with respect to γ.
The inplace version is equivalent to `result_final .-= c * dγdβresβ_ij`
"""
@inline function dγdβresβ_ij(dist, link, xj, z, η_j, μ_j, varμ_j, res_j)
    invσ_j = inv(sqrt(varμ_j)) # 1 × 1
    ∇ᵧμ_ij  = GLM.mueta(link, η_j) * z # 1 × 1
    ∇ᵦμ_ij  = GLM.mueta(link, η_j) * xj # p × 1
    ∇ᵧσ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * z # 1 × 1
    ∇ᵦσ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * xj # p × 1

    # assemble 5 terms
    term1 = -invσ_j * ∇²μ_j(link, η_j, xj, z)
    term2 = 0.5invσ_j^3 * ∇ᵦσ²_ij * ∇ᵧμ_ij
    term3 = -0.5 * res_j * inv(varμ_j) * ∇²σ²_j(dist, link, xj, z, μ_j, η_j)
    term4 = 0.5invσ_j^3 * ∇ᵦμ_ij * ∇ᵧσ²_ij
    term5 = 0.75res_j * inv(varμ_j^2) * ∇ᵦσ²_ij * ∇ᵧσ²_ij
    result = term1 + term2 + term3 + term4 + term5

    return result # p × 1
end
@inline function dγdβresβ_ij!(result_final::AbstractVector{T}, dist::Distribution, link::Link,
    xj::AbstractVector{T}, z::T, η_j::T, μ_j::T, varμ_j::T, res_j::T, 
    c::T, storage::Storages) where T
    result = storage.p_storage
    ∇ᵦσ²_ij = storage.p_storage2
    fill!(result, 0)

    # needed quantities
    invσ_j = inv(sqrt(varμ_j)) # 1 × 1
    ∇ᵧμ_ij  = GLM.mueta(link, η_j) * z # 1 × 1
    ∇ᵧσ²_ij = sigmamu(dist, μ_j) * GLM.mueta(link, η_j) * z # 1 × 1
    # ∇ᵦμ_ij  = GLM.mueta(link, η_j) * xj # p × 1
    ∇ᵦσ²_ij .= (sigmamu(dist, μ_j) * GLM.mueta(link, η_j)) .* xj # p × 1

    # term1 (-invσ_j * ∇²μ_j(link, η_j, xj, z))
    ∇²μ_j!(result, link, η_j, xj, z, -invσ_j)
    # term2 (0.5invσ_j^3 * ∇ᵦσ²_ij * ∇ᵧμ_ij)
    result .+= (0.5invσ_j^3 * ∇ᵧμ_ij) .* ∇ᵦσ²_ij
    # term3 
    ∇²σ²_j!(result, dist, link, xj, z, μ_j, η_j, -0.5 * res_j * inv(varμ_j))
    # term 4 (0.5invσ_j^3 * ∇ᵦμ_ij * ∇ᵧσ²_ij)
    result .+= (0.5invσ_j^3 * ∇ᵧσ²_ij * GLM.mueta(link, η_j)) .* xj
    # term 5
    result .+= (0.75res_j * inv(varμ_j^2) * ∇ᵧσ²_ij) .* ∇ᵦσ²_ij

    # scale by c
    result_final .-= c .* result

    return result_final # p × 1
end

"""
    update_∇res_ij(d::Distribution, x_ji, res_j, μ_j, dμ_j, varμ_j)

Computes ∇resβ_ij, the gradient (wrt β) of the standardized residuals for sample i 
at the j'th measurement. Here res_j is the standardized residual
"""
function update_∇res_ij end
@inline update_∇res_ij(d::Normal, x_ji, res_j, μ_j, dμ_j, varμ_j) = -x_ji
@inline update_∇res_ij(d::Bernoulli, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -sqrt(varμ_j) * x_ji - (0.5 * res_j * (1 - 2μ_j) * x_ji)
@inline update_∇res_ij(d::Poisson, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    x_ji * (
    -(inv(sqrt(varμ_j)) + (0.5 * inv(varμ_j)) * res_j) * dμ_j
    )
@inline update_∇res_ij(d::NegativeBinomial, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -inv(sqrt(varμ_j)) * dμ_j * x_ji - (0.5 * inv(varμ_j)) *
    res_j * (μ_j * inv(d.r) + (1 + inv(d.r) * μ_j)) * dμ_j * x_ji

"""
    _get_null_distribution(gcm)

Tries to guess the base distribution for an abstract Copula Model.
"""
_get_null_distribution(gcm::GaussianCopulaVCModel) = Normal()
_get_null_distribution(gcm::NBCopulaVCModel) = NegativeBinomial()
function _get_null_distribution(gcm::GLMCopulaVCModel)
    T = eltype(gcm.data[1].X)
    if eltype(gcm.d) == Bernoulli{T}
        return Bernoulli()
    elseif eltype(gcm.d) == Poisson{T}
        return Poisson()
    else
        error("GLMCopulaVCModel should have marginal distributions Bernoulli or Poisson but was $(eltype(gcm.d))")
    end
end

function simulate_random_snparray(s::Union{String, UndefInitializer}, n::Int64,
    p::Int64; mafs::Vector{Float64}=zeros(Float64, p), min_ma::Int = 5)

    #first simulate a random {0, 1, 2} matrix with each SNP drawn from Binomial(2, r[i])
    A1 = BitArray(undef, n, p) 
    A2 = BitArray(undef, n, p) 
    for j in 1:p
        minor_alleles = 0
        maf = NaN
        while minor_alleles <= min_ma
            maf = mafs[j] == 0 ? 0.5rand() : mafs[j]
            for i in 1:n
                A1[i, j] = rand(Bernoulli(maf))
                A2[i, j] = rand(Bernoulli(maf))
            end
            minor_alleles = sum(view(A1, :, j)) + sum(view(A2, :, j))
        end
        mafs[j] = maf
    end

    #fill the SnpArray with the corresponding x_tmp entry
    return _make_snparray(s, A1, A2)
end

function _make_snparray(s::Union{String, UndefInitializer}, A1::BitArray, A2::BitArray)
    n, p = size(A1)
    x = SnpArray(s, n, p)
    for i in 1:(n*p)
        c = A1[i] + A2[i]
        if c == 0
            x[i] = 0x00
        elseif c == 1
            x[i] = 0x02
        elseif c == 2
            x[i] = 0x03
        else
            throw(MissingException("matrix shouldn't have missing values!"))
        end
    end
    return x
end

function simulate_multivariate_traits(;
    p = 5,    # number of fixed effects, including intercept
    n = 1000, # number of sample
    d = 3,    # number of phenotypes per sample
    q = 1000, # number of SNPs
    k = 0,    # number of causal SNPs
    Γ::Matrix{Float64} = Matrix(SymmetricToeplitz(0.5.^(0:(d-1)))),
    seed::Int = 2023,
    possible_distributions = [Bernoulli, Poisson, Normal],
    σ2true = 1.0, # true nuisance parameter used for Gaussian phenoypes (assumes all gaussian phenotype have same variance)
    Btrue = rand(Uniform(-0.5, 0.5), p, d), # true effect sizes for nongenetic covariates
    maf = 0.5rand(),
    use_VC_model::Bool = false,
    use_VC_AD_model::Bool = false
    )
    # sample d marginal distributions for each phenotype within samples
    Random.seed!(seed)
    vecdist = rand(possible_distributions, d)
    veclink = [canonicallink(vecdist[j]()) for j in 1:d]

    # simulate non-genetic design matrix
    Random.seed!(seed)
    X = [ones(n) randn(n, p - 1)]

    # set minor allele freq
    mafs = fill(maf, q)

    # simulate random SnpArray with q SNPs and randomly choose k SNPs to be causal
    Random.seed!(seed)
    G = simulate_random_snparray(undef, n, q, mafs=mafs)
    Gfloat = convert(Matrix{Float64}, G, center=true, scale=true)
    γtrue = zeros(q, d)
    causal_snps = sample(1:q, k, replace=false) |> sort
    γtrue[causal_snps, 1] .= rand([-0.5, 0.5], k)
    for j in 2:d
        γtrue[causal_snps, j] .= 0.5^j .* γtrue[causal_snps, 1]
    end

    # sample phenotypes
    Y = zeros(n, d)
    y = Vector{Float64}(undef, d)
    for i in 1:n
        Xi = X[i, :]
        Gi = Gfloat[i, :]
        η = Btrue' * Xi + γtrue' * Gi
        vecd_tmp = Vector{UnivariateDistribution}(undef, d)
        for j in 1:d
            dist = vecdist[j]
            μj = GLM.linkinv(canonicallink(dist()), η[j])
            if dist == Normal
                vecd_tmp[j] = Normal(μj, sqrt(σ2true))
            else
                vecd_tmp[j] = dist(μj)
            end
        end
        multivariate_dist = MultivariateMix(vecd_tmp, Γ)
        res = Vector{Float64}(undef, d)
        rand(multivariate_dist, y, res)
        Y[i, :] .= y
    end

    # form model
    qc_model = MultivariateCopulaModel(Y, X, vecdist, veclink)

    # test
    if use_VC_model
        V = [ones(d, d), Matrix(I, d, d)]
        qc_model = MultivariateCopulaVCModel(Y, X, V, vecdist, veclink);
    elseif use_VC_AD_model
        V = [ones(d, d), Matrix(I, d, d), cov(Y)]
        qc_model = MultivariateCopulaModel_AD(Y, X, V, vecdist, veclink);
    end
    initialize_model!(qc_model)

    return qc_model, G, Btrue, Γ, γtrue, σ2true
end

function simulate_longitudinal_traits(;
    n = 1000, # sample size
    d_min = 1, # min number of observations per sample
    d_max = 5, # max number of observations per sample
    p = 3, # number of nongenetic covariates, including intercept
    m = 1, # number of variance components
    q = 1000, # number of SNPs
    k = 10, # number of causal SNPs
    maf = 0.5rand(),
    causal_snp_β = 0.5rand(),
    τtrue = 0.01, # inverse variance for gaussian case (e.g. σ^2 = 100)
    seed = 2022,
    y_distribution = Bernoulli,
    T = Float64,
    plinkdir::Union{String, UndefInitializer} = undef,
    )
    Random.seed!(seed)
    m == 1 || m == 2 || error("m (number of VC) must be 1 or 2")
    
    # non-genetic effect sizes
    Random.seed!(seed)
    βtrue = [1.0; rand(-0.5:1:0.5, p-1)]
    dist = y_distribution()
    link = y_distribution == NegativeBinomial ? LogLink() : canonicallink(dist)
    Dist = typeof(dist)
    Link = typeof(link)

    # variance components
    θtrue = fill(0.5, m)

    # simulate (nongenetic) design matrices
    Random.seed!(seed)
    X_full = Matrix{Float64}[]
    for i in 1:n
        nobs = rand(d_min:d_max) # number of obs for this sample
        push!(X_full, hcat(ones(nobs), randn(nobs, p - 1)))
    end
    
    # simulate causal alleles
    Random.seed!(seed)
    γtrue = zeros(q)
    γtrue[1:k] .= causal_snp_β
    shuffle!(γtrue)
    
    # set minor allele freq
    mafs = fill(maf, q)

    # simulate random SnpArray with q SNPs with prespecified maf
    Random.seed!(seed)
    G = simulate_random_snparray(plinkdir, n, q, mafs=mafs)
    Gfloat = convert(Matrix{T}, G, center=true, scale=true)

    # effect of causal alleles
    η_G = Gfloat * γtrue

    # simulate phenotypes
    if y_distribution == Normal
        σ2 = inv(τtrue)
        σ = sqrt(σ2)
        obs = Vector{GaussianCopulaVCObs{T}}(undef, n)
        for i in 1:n
            # data matrix
            X = X_full[i]
            η = X * βtrue
            η .+= η_G[i] # add genetic effects
            μ = GLM.linkinv.(link, η)
            vecd = Vector{ContinuousUnivariateDistribution}(undef, size(X, 1))
            # VC matrices
            V1 = ones(size(X, 1), size(X, 1))
            V2 = Matrix(I, size(X, 1), size(X, 1))
            Γ = m == 1 ? θtrue[1] * V1 : θtrue[1] * V1 + θtrue[2] * V2
            for i in 1:size(X, 1)
                vecd[i] = y_distribution(μ[i], σ)
            end
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            # simuate single vector y
            y = Vector{T}(undef, size(X, 1))
            res = Vector{T}(undef, size(X, 1))
            rand(nonmixed_multivariate_dist, y, res)
            V = m == 1 ? [V1] : [V1, V2]
            obs[i] = GaussianCopulaVCObs(y, X, V)
        end
        qc_model = GaussianCopulaVCModel(obs)
    elseif y_distribution == NegativeBinomial
        rtrue = 10.0
        obs = Vector{NBCopulaVCObs{T, Dist, Link}}(undef, n)
        for i in 1:n
            # data matrix
            X = X_full[i]
            η = X * βtrue
            η .+= η_G[i] # add genetic effects
            μ = GLM.linkinv.(link, η)
            p = rtrue ./ (μ .+ rtrue)
            vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:size(X, 1)]
            # VC matrices
            V1 = ones(size(X, 1), size(X, 1))
            V2 = Matrix(I, size(X, 1), size(X, 1))
            Γ = m == 1 ? θtrue[1] * V1 : θtrue[1] * V1 + θtrue[2] * V2
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            # simuate single vector y
            y = Vector{Float64}(undef, size(X, 1))
            res = Vector{Float64}(undef, size(X, 1))
            rand(nonmixed_multivariate_dist, y, res)
            V = m == 1 ? [V1] : [V1, V2]
            obs[i] = NBCopulaVCObs(y, X, V, dist, link)
        end
        qc_model = NBCopulaVCModel(obs)
    else # Bernoulli or Poisson
        obs = Vector{GLMCopulaVCObs{T, Dist, Link}}(undef, n)
        for i in 1:n
            # data matrix
            X = X_full[i]
            η = X * βtrue
            η .+= η_G[i] # add genetic effects
            μ = GLM.linkinv.(link, η)
            # VC matrices
            V1 = ones(size(X, 1), size(X, 1))
            V2 = 0.1Matrix(I, size(X, 1), size(X, 1))
            Γ = m == 1 ? θtrue[1] * V1 : θtrue[1] * V1 + θtrue[2] * V2
            vecd = Vector{DiscreteUnivariateDistribution}(undef, size(X, 1))
            for i in 1:size(X, 1)
                vecd[i] = y_distribution(μ[i])
            end
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            # simuate single vector y
            y = Vector{T}(undef, size(X, 1))
            res = Vector{T}(undef, size(X, 1))
            rand(nonmixed_multivariate_dist, y, res)
            V = m == 1 ? [V1] : [V1, V2]
            obs[i] = GLMCopulaVCObs(y, X, V, dist, link)
        end
        qc_model = GLMCopulaVCModel(obs)
    end
    initialize_model!(qc_model)
    return qc_model, G, βtrue, θtrue, γtrue, τtrue
end

function zscore(p::T, beta::T) where T
    return sign(beta) * quantile(Normal(), p/2)
end
