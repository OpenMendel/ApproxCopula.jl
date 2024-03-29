"""
    std_res_differential!(gc)

compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta. For Normal it will be X.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T <: BlasReal, D<:Normal{T}, Link}
    gc.∇resβ .= gc.X
    gc.∇resβ .*= -one(T)
    nothing
end

"""
    std_res_differential!(gc)

compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Negative Binomial.
"""
function std_res_differential!(gc::Union{NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}
    ) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @inbounds for i in 1:gc.p
        @simd for j in 1:gc.n
            gc.∇resβ[j, i] = -inv(sqrt(gc.varμ[j])) * gc.dμ[j] * gc.X[j, i] - (0.5 * inv(gc.varμ[j])) * gc.res[j] * (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) *  gc.dμ[j] * gc.X[j, i]
        end
    end
    nothing
end

"""
    std_res_differential!(gc)

compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Poisson.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
    @inbounds for i in 1:gc.p
        @simd for j in 1:gc.n
            gc.∇resβ[j, i] = gc.X[j, i]
            gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
        end
    end
    nothing
end

"""
    std_res_differential!(gc)

compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Bernoulli.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Bernoulli{T}, Link}
    @inbounds for i in 1:gc.p
        @simd for j in 1:gc.n
            gc.∇resβ[j, i] = -sqrt(gc.varμ[j]) * gc.X[j, i] - (0.5 * gc.res[j] * (1 - (2 * gc.μ[j])) * gc.X[j, i])
        end
    end
    nothing
end

"""
    glm_gradient(gc, β)

Calculates the gradient with respect to beta for our the glm portion for one obs. Keeps the residuals standardized.
"""
function glm_gradient(gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaARObs, NBCopulaARObs, NBCopulaCSObs, GLMCopulaCSObs})
    gc.storage_n .= gc.w1 .* gc.res
    mul!(gc.storage_p1, transpose(gc.X), gc.storage_n)
    gc.storage_p1
end

"""
    glm_hessian(gc)

Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
"""
function glm_hessian(gc::Union{GLMCopulaVCObs, NBCopulaVCObs, GLMCopulaARObs, NBCopulaARObs, NBCopulaCSObs, GLMCopulaCSObs})
    mul!(gc.storage_np, Diagonal(gc.w2), gc.X)
    BLAS.gemm!('T', 'N', -1.0, gc.X, gc.storage_np, 0.0, gc.storage_pp)
end

"""
    update_∇θ!(gcm)

Update θ gradient for Newton's Algorithm, given β.
"""
function update_∇θ!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    @inbounds for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i])            # standardize the residuals GLM variance(μ)
        QuasiCopula.update_quadform!(gcm.data[i]) # with standardized residuals
        gcm.QF[i, :] .= gcm.data[i].q
    end
    BLAS.gemv!('N', T(1), gcm.QF,  gcm.θ, T(0), gcm.storage_n)
    gcm.storage_n .= inv.(1 .+ gcm.storage_n)
    BLAS.gemv!('N', T(1), gcm.TR,  gcm.θ, T(0), gcm.storage_n2)
    gcm.storage_n2 .= inv.(1 .+ gcm.storage_n2)
    BLAS.gemv!('T', T(1), gcm.QF, gcm.storage_n, T(0), gcm.∇θ)
    BLAS.gemv!('T', -T(1), gcm.TR, gcm.storage_n2, T(1), gcm.∇θ)
end

"""
    update_Hθ!(gcm)

Update θ Hessian for Newton's Algorithm, given β.
"""
function update_Hθ!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    fill!(gcm.Hθ, 0.0)
    @inbounds for j in 1:gcm.m
        @simd for i in 1:gcm.n
            gcm.hess1[j, i] = gcm.QF[i, j] * gcm.storage_n[i]
            gcm.hess2[j, i] = gcm.TR[i, j] * gcm.storage_n2[i]
        end
    end
    BLAS.gemm!('N', 'T', -T(1), gcm.hess1, gcm.hess1, T(0), gcm.Hθ)
    BLAS.gemm!('N', 'T', T(1), gcm.hess2, gcm.hess2, T(1), gcm.Hθ)
end
