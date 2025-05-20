using ApproxCopula
using BenchmarkTools
using GLM
using Distributions
using Test
using LinearAlgebra

@testset "GLM utilities" begin
    # ∇²μ_j vs ∇²μ_j! (p × p)
    p = 10
    storage = zeros(p, p)
    xj = randn(p)
    l = GLM.LogitLink()
    ηj = randn()
    c = rand()
    ApproxCopula.∇²μ_j!(storage, l, ηj, xj, c)
    @test all(c .* ApproxCopula.∇²μ_j(l, ηj, xj) .≈ storage)

    # ∇²μ_j vs ∇²μ_j! (p × 1)
    storage = zeros(p)
    xj = randn(p)
    zj = randn()
    l = GLM.LogitLink()
    ηj = randn()
    c = rand()
    ApproxCopula.∇²μ_j!(storage, l, ηj, xj, zj, c)
    @test all(c .* ApproxCopula.∇²μ_j(l, ηj, xj, zj) .≈ storage)

    # ∇²σ²_j vs ∇²σ²_j! (p × p)
    p = 10
    storage = zeros(p, p)
    xj = randn(p)
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    ηj = randn()
    μj = rand()
    c = rand()
    ApproxCopula.∇²σ²_j!(storage, d, l, xj, μj, ηj, c)
    @test all(c .* ApproxCopula.∇²σ²_j(d, l, xj, μj, ηj) .≈ storage)

    # ∇²σ²_j vs ∇²σ²_j! (p × 1)
    p = 10
    storage = zeros(p)
    xj = randn(p)
    zj = randn()
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    ηj = randn()
    μj = rand()
    c = randn()
    ApproxCopula.∇²σ²_j!(storage, d, l, xj, μj, ηj, zj, c)
    @test all(c .* ApproxCopula.∇²σ²_j(d, l, xj, μj, ηj, zj) .≈ storage)

    # dγdβresβ_ij vs dγdβresβ_ij! (later should be efficient)
    n = 100
    p = 100
    s = 0
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    xj = randn(p)
    z = randn()
    η_j = randn()
    μ_j = randn()
    varμ_j = rand()
    res_j = randn()
    maxd = 10
    storage = ApproxCopula.storages(p, maxd, 2, s)
    W = zeros(p)
    c = randn()
    ApproxCopula.dγdβresβ_ij!(W, d, l, xj, z, η_j, μ_j, varμ_j, res_j, c, storage)
    @test all(-c .* ApproxCopula.dγdβresβ_ij(d, l, xj, z, η_j, μ_j, varμ_j, res_j) .≈ W)
    b = @benchmark ApproxCopula.dγdβresβ_ij!($W, $d, $l, $xj, $z, $η_j, $μ_j, $varμ_j, $res_j, $c, $storage)
    @test b.allocs == 0
    @test b.memory == 0

    # get_Hβγ_i vs get_neg_Hβγ_i! (later should be efficient)
    p = 15
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        d_min=5, d_max=maxd, m=m, p=p, y_distribution=Bernoulli)
    i = 1
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    storages = ApproxCopula.storages(p, maxd, m, s)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)
    storages.denom[1] = denom
    storages.denom2[1] = denom2
    W = zeros(p)
    ApproxCopula.get_neg_Hβγ_i!(W, qc, Γ, qc.∇resβ, ∇resγ, z, storages)
    @test all(ApproxCopula.get_Hβγ_i(qc, Γ, qc.∇resβ, ∇resγ, z, storages) .≈ W)
    b = @benchmark ApproxCopula.get_neg_Hβγ_i!($W, $qc, $Γ, $(qc.∇resβ), $∇resγ, $z, $storages)
    @test b.allocs == 0
    @test b.memory == 0

    # get_neg_Hθγ_i! vs get_neg_Hθγ_i (later should be efficient)
    p = 15
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        d_min=5, d_max=maxd, m=m, p=p, y_distribution=Bernoulli)
    i = 1
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    storages = ApproxCopula.storages(p, maxd, m, s)
    W = zeros(m)
    ApproxCopula.get_neg_Hθγ_i!(W, qc, θtrue, ∇resγ, storages)
    @test all(ApproxCopula.get_neg_Hθγ_i(qc, θtrue, ∇resγ, storages) .≈ W)
    b = @benchmark ApproxCopula.get_neg_Hθγ_i!($W, $qc, $θtrue, $∇resγ, $storages)
    @test b.allocs == 0
    @test b.memory == 0

    # dβdβ_res_ij should be efficient when xj is scalar
    dist = GLM.Bernoulli()
    link = GLM.LogitLink()
    xj = randn()
    η_j = randn()
    μ_j = randn()
    varμ_j = rand()
    res_j = randn()
    b = @benchmark ApproxCopula.dβdβ_res_ij($dist, $link, $xj, $η_j, $μ_j, $varμ_j, $res_j)
    @test b.allocs == 0
    @test b.memory == 0
end

@testset "Gaussian utilities" begin
    # get_neg_Hβγ_i! (note: comparison with get_Hβγ_i is omitted here since
    #                   get_Hβγ_i required β as input, which is tedious to 
    #                   call here, as we need to update η, μ,... etc.)
    p = 15
    mind = 1
    maxd = 10
    m = 2
    s = 1
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        d_min=mind, d_max=maxd, m=m, p=p, y_distribution=Normal)
    i = 1
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    τ = rand()
    storages = ApproxCopula.storages(p, maxd, m, s)
    W = zeros(p)
    b = @benchmark ApproxCopula.get_neg_Hβγ_i!($W, $qc, $Γ, $z, $τ, $storages)
    @test b.allocs == 0
    @test b.memory == 0

    # get_Hτγ_i
    storages = ApproxCopula.storages(p, maxd, m, s)
    b = @benchmark ApproxCopula.get_Hτγ_i($qc, $z, $θtrue, $τ, $storages)
    @test b.allocs == 0
    @test b.memory == 0
end

@testset "GLM update_W!" begin
    n = 1000
    p = 15
    mind = 1
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        n=n, d_min=mind, d_max=maxd, m=m, p=p, y_distribution=Bernoulli)
    i = rand(1:n)
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    ∇resβ = ApproxCopula.get_∇resβ(qc_model, i) # d × p
    storages = ApproxCopula.storages(p, maxd, m, s)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)
    storages.denom[1] = denom
    storages.denom2[1] = denom2

    # correctness
    W = zeros(p + m)
    ApproxCopula.update_W!(W, qc_model, i, z, Γ, ∇resβ, ∇resγ, storages)
    W1true = ApproxCopula.get_Hβγ_i(qc, Γ, ∇resβ, ∇resγ, z, storages)
    W2true = ApproxCopula.get_neg_Hθγ_i(qc, qc_model.θ, ∇resγ, storages)
    @test all([W1true; W2true] .≈ W)
    
    # efficiency
    b = @benchmark ApproxCopula.update_W!($W, $qc_model, $i, $z, $Γ, $∇resβ, $∇resγ, $storages)
    @test b.allocs == 0
    @test b.memory == 0
end

@testset "Gaussian update_W!" begin
    n = 1000
    p = 15
    mind = 1
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        n=n, d_min=mind, d_max=maxd, m=m, p=p, y_distribution=Normal)
    i = rand(1:n)
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    ∇resβ = ApproxCopula.get_∇resβ(qc_model, i) # d × p
    storages = ApproxCopula.storages(p, maxd, m, s)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)
    storages.denom[1] = denom
    storages.denom2[1] = denom2
    τ = qc_model.τ
    W = zeros(p + m + 1)

    # efficiency
    b = @benchmark ApproxCopula.update_W!($W, $qc_model, $i, $z, $Γ, $∇resβ, $∇resγ, $storages)
    @test b.allocs == 0
    @test b.memory == 0
end

@testset "calculate_Qi" begin
    n = 1000
    p = 15
    mind = 1
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        n=n, d_min=mind, d_max=maxd, m=m, p=p, y_distribution=Bernoulli)
    i = rand(1:n)
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    ∇resβ = ApproxCopula.get_∇resβ(qc_model, i) # d × p
    storages = ApproxCopula.storages(p, maxd, m, s)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)

    # efficiency
    b = @benchmark ApproxCopula.calculate_Qi($qc_model, $i, $z, $Γ, $∇resγ, $denom, $denom2, $storages)
    @test b.allocs == 0
    @test b.memory == 0
end

@testset "calculate_Ri" begin
    n = 1000
    p = 15
    mind = 1
    maxd = 10
    m = 2
    s = 0
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        n=n, d_min=mind, d_max=maxd, m=m, p=p, y_distribution=Bernoulli)
    i = rand(1:n)
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = ApproxCopula.get_∇resγ(qc_model, i, z) # d × 1
    ∇resβ = ApproxCopula.get_∇resβ(qc_model, i) # d × p
    storages = ApproxCopula.storages(p, maxd, m, s)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)

    # efficiency
    b = @benchmark ApproxCopula.calculate_Ri($qc_model, $i, $z, $Γ, $∇resγ, $denom, $storages)
    @test b.allocs == 0
    @test b.memory == 0
end
