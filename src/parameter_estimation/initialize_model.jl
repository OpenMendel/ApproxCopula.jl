"""
    initialize_beta!(gcm{GLMCopulaVCModel})

Initialize the linear regression parameters `β` using GLM.jl
"""
function initialize_beta!(gcm::Union{GLMCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}, GLMCopulaCSModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    # form df
    Xstack = []
    Ystack = []
    @inbounds for i in 1:length(gcm.data)
        push!(Xstack, gcm.data[i].X)
        push!(Ystack, gcm.data[i].y)
    end
    Xstack = [vcat(Xstack...)][1]
    Ystack = [vcat(Ystack...)][1]
    fit_glm = GLM.glm(Xstack, Ystack, gcm.d[1], gcm.link[1])
    copyto!(gcm.β, fit_glm.pp.beta0)
    nothing
end

"""
    initialize_model!(gcm)

Initialize the linear regression parameters `β` using Newton's Algorithm under Independence Assumption, update variance components using MM-Algorithm.
"""
function initialize_model!(
    gcm::GLMCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    fill!(gcm.τ, 1.0)
    fill!(gcm.θ, 1.0)
    println("initializing variance components using MM-Algorithm")
    update_θ!(gcm)
    copyto!(gcm.σ2, gcm.θ)
    copyto!(gcm.ρ, 0.2)
    nothing
end

function initialize_model!(
    gcm::GLMCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    copyto!(gcm.ρ, 0.2)
    nothing
end

"""
    initialize_model!(gcm{GLMCopulaVCModel NBCopulaVCModel})

Initialize the linear regression parameters `β` using GLM.jl, and update variance components using MM-Algorithm.
"""
function initialize_model!(
    gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    # println("initializing β using Newton's Algorithm under Independence Assumption")
    initialize_beta!(gcm)
    # @show gcm.β
    fill!(gcm.τ, 1.0)
    # println("initializing variance components using MM-Algorithm")
    fill!(gcm.θ, 1.0)
    update_θ!(gcm)
    if sum(gcm.θ) >= 20
      fill!(gcm.θ, 1.0)
    end
    # @show gcm.θ
    nothing
end

"""
    initialize_model!(gcm)
Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least
squares solution.
"""
function initialize_model!(gcm::GaussianCopulaVCModel{T}) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p)
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    # @show gcm.β
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    # println("initializing dispersion using residual sum of squares")
    gcm.τ[1] = gcm.ntotal / rss
    # @show gcm.τ
    # println("initializing variance components using MM-Algorithm")
    fill!(gcm.θ, 1.0)
    update_θ!(gcm)
    # @show gcm.θ
    nothing
end

# march 6 2022 try this initialize model instead so that r doesn't do weird things
function initialize_model!(
    gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    println("initializing β using GLM.jl")
    initialize_beta!(gcm) # initialize beta using negative binomial glm from GLM.jl
    @show gcm.β
    fill!(gcm.θ, 1.0)
    fill!(gcm.τ, 1.0)
    println("initializing variance components using MM-Algorithm")
    update_θ!(gcm)
    if sum(gcm.θ) >= 20
        fill!(gcm.θ, 1.0)
    end
    @show gcm.θ
    println("initializing r using Newton update")
    fill!(gcm.r, 1)
    ApproxCopula.update_r!(gcm)
    nothing
end

# # code inspired from https://github.com/JuliaStats/GLM.jl/blob/master/src/negbinfit.jl
function initialize_model!(gcm::NBCopulaCSModel{T, D, Link}) where {T <: BlasReal, D, Link}
    # initial guess for r = 1
    fill!(gcm.r, 1)

    # fit a Poisson regression model to estimate μ, η, β, τ
    println("Initializing NegBin r to Poisson regression values")
    nsample = length(gcm.data)
    gcsPoisson = Vector{GLMCopulaCSObs{T, Poisson{T}, LogLink}}(undef, nsample)
    for (i, gc) in enumerate(gcm.data)
        gcsPoisson[i] = GLMCopulaCSObs(gc.y, gc.X, Poisson(), LogLink())
    end
    gcmPoisson = GLMCopulaCSModel(gcsPoisson)
    initialize_model!(gcmPoisson) # initialize beta using poisson glm from GLM.jl
    copyto!(gcm.β, gcmPoisson.β)
    copyto!(gcm.σ2, gcmPoisson.σ2)

    # update r using maximum likelihood with Newton's method
    for gc in gcm.data
        fill!(gcm.τ, 1.0)
        fill!(gc.∇β, 0)
        fill!(gc.Hβ, 0)
        fill!(gc.varμ, 1)
        fill!(gc.res, 0)
    end
    println("initializing r using Newton update")
    ApproxCopula.update_r!(gcm)
    copyto!(gcm.ρ, 0.2)
    nothing
end

function initialize_model!(gcm::NBCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}
    # initial guess for r = 1
    fill!(gcm.r, 1)

    # fit a Poisson regression model to estimate μ, η, β, τ
    println("Initializing NegBin r to Poisson regression values")
    nsample = length(gcm.data)
    gcsPoisson = Vector{GLMCopulaARObs{T, Poisson{T}, LogLink}}(undef, nsample)
    for (i, gc) in enumerate(gcm.data)
        gcsPoisson[i] = GLMCopulaARObs(gc.y, gc.X, Poisson(), LogLink())
    end
    gcmPoisson = GLMCopulaARModel(gcsPoisson)
    initialize_model!(gcmPoisson) # initialize beta using poisson glm from GLM.jl
    copyto!(gcm.β, gcmPoisson.β)
    copyto!(gcm.σ2, gcmPoisson.σ2)

    # update r using maximum likelihood with Newton's method
    for gc in gcm.data
        fill!(gcm.τ, 1.0)
        fill!(gc.∇β, 0)
        fill!(gc.Hβ, 0)
        fill!(gc.varμ, 1)
        fill!(gc.res, 0)
    end
    println("initializing r using Newton update")
    ApproxCopula.update_r!(gcm)
    copyto!(gcm.ρ, 0.2)
    nothing
end
