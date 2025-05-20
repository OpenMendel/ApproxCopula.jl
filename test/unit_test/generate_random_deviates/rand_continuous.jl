module RandTest

using ApproxCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

# Normal
@testset "Normal(0,1) * (1 + 0.5 x^2)" begin ## where x = mean(dist)
dist = Normal(0, 1)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_normal = pdf_constants(γ, dist)

# test if the CDF is a proper density (integrates to 1)
inv(d_normal.c0 + d_normal.c1 * d_normal.μ + d_normal.c2 * (d_normal.σ2 + abs2(d_normal.μ)))
@test d_normal.c * (d_normal.c0 + d_normal.c1  * d_normal.μ + d_normal.c2 * (d_normal.σ2 + abs2(d_normal.μ))) == 1.0
@test minimum(d_normal) == -Inf
@test maximum(d_normal) == Inf
@test cdf(d_normal, -Inf) == 0
@test cdf(d_normal, Inf) == 1

Random.seed!(1234)
nsample = 10_000
@info "sample $nsample points for the $dist distribution"
s = Vector{Float64}(undef, nsample)
rand!(d_normal, s) # compile
@time rand!(d_normal, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_normal))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_normal))")
end

# GAMMA
# when α = 2.5, 1.0 = θ
@testset "Gamma(α = 2.5, θ = 1.0) * (c0 + c1 * x + c2 * x^2)" begin ## where x = mean(dist)
Random.seed!(1234)
α, θ = (2.5, 1.0)
# res = Vector{Float64}(undef, 5)
# Γ = rand(5, 5)
γ = 1.0
dist = Gamma(α, θ)

# test if the proper c0, c1, c2 constants are stored.
d_gamma = pdf_constants(γ, dist)
@test d_gamma.c0 == 1 + (mean(d_gamma.d)^2 * inv(var(d_gamma.d)) * 0.5 * γ)
@test d_gamma.c1 == 0.5 * γ * (-2 * mean(d_gamma.d) * inv(var(d_gamma.d)))
@test d_gamma.c2 == 0.5 * γ * (inv(var(d_gamma.d)))

# test if the CDF is a proper density (integrates to 1)
α, θ = Distributions.params(dist)
normalizing_c1 = (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α))
normalizing_c2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))
@test d_gamma.c * (d_gamma.c0 + normalizing_c1 * d_gamma.c1 + normalizing_c2 * d_gamma.c2) == 1.0

# @test d_gamma.c ≈ inv(1+ 0.5 * tr(Γ))
@test minimum(d_gamma) == 0
@test maximum(d_gamma) == Inf
@test cdf(d_gamma, 0) == 0
@test cdf(d_gamma, Inf) == 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution"
s = Vector{Float64}(undef, nsample)
rand!(d_gamma, s) # compile
@time rand!(d_gamma, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_gamma))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_gamma))")
end

# n = 10,0000 ::: time = 0.028 seconds
# sample mean = 3.1510221305244226; theoretical mean = 3.1666666666666665
# sample var = 5.67119904573935; theoretical var = 5.722222222222223

# ## when there is a lot of noise there is some deviation from truth
# @testset "Gamma(α = 3.0, θ = 2.0) * (c0 + c1 * x + c2 * x^2); Lots of nosie ==> deviation" begin ## where x = mean(dist)
# α, θ = (3.0, 2.0)
# Random.seed!(1234)
# dist = Gamma(α, θ)

# # test if the proper c0, c1, c2 constants are stored.
# d_gamma = pdf_constants(γ, dist)
# @test d_gamma.c0 == 1 + (mean(d_gamma.d)^2 * inv(var(d_gamma.d)) * 0.5 * γ)
# @test d_gamma.c1 == 0.5 * γ * (-2 * mean(d_gamma.d) * inv(var(d_gamma.d)))
# @test d_gamma.c2 == 0.5 * γ * (inv(var(d_gamma.d)))

# # test if the CDF is a proper density (integrates to 1)
# α, θ = params(dist)
# normalizing_c1 = (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α))
# normalizing_c2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))
# @test d_gamma.c * (d_gamma.c0 + d_gamma.c1  * d_gamma.μ + d_gamma.c2 * (d_gamma.σ2 + abs2(d_gamma.μ))) == 1.0


# @test d_gamma.c ≈ inv(1+ 0.5 * γ)
# @test minimum(d_gamma) == 0
# @test maximum(d_gamma) == Inf
# @test cdf(d_gamma, 0) == 0
# @test cdf(d_gamma, Inf) == 1

# ###
# Random.seed!(1234)
# nsample = 10_000 #
# @info "sample $nsample points for the $dist distribution"
# s = Vector{Float64}(undef, nsample)
# rand!(d_gamma, s) # compile
# @time rand!(d_gamma, s)
# println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_gamma))")
# println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_gamma))")
# end
# # n = 10,0000 ::: time = 0.049717 seconds
# # sample mean = 5.663051756167577; theoretical mean = 6.632654124304445
# # sample var = 11.845482796775215; theoretical var = 19.191598250653946

### EXPONENTIAL test
@testset "Exponential(θ = 3) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(1234)
dist = Exponential(3)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_exp = pdf_constants(γ, dist)
@test d_exp.c0 == 1  + (mean(d_exp.d)^2 * inv(var(d_exp.d)) * 0.5 * γ)
@test d_exp.c1 == 0.5 * γ * (-2 * mean(d_exp.d) * inv(var(d_exp.d)))
@test d_exp.c2 == 0.5 * γ * (inv(var(d_exp.d)))

# test if the CDF is a proper density (integrates to 1)
θ = Distributions.params(d_exp.d)[1]
normalizing_c1 = θ * (StatsFuns.gamma(2)/ StatsFuns.gamma(1))
normalizing_c2 = θ^2 * (StatsFuns.gamma(3)/ StatsFuns.gamma(1))
@test d_exp.c * (d_exp.c0 + normalizing_c1 * d_exp.c1 + normalizing_c2 * d_exp.c2) == 1.0

@test d_exp.c ≈ inv(1+ 0.5 * γ)
@test minimum(d_exp) == 0
@test maximum(d_exp) == Inf
@test cdf(d_exp, 0) == 0
@test cdf(d_exp, Inf) == 1

###
Random.seed!(1234)
nsample = 10_000 #_000
@info "sample $nsample points for the $dist distribution"
s = Vector{Float64}(undef, nsample)
rand!(d_exp, s) # compile
@time rand!(d_exp, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_exp))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_exp))")
end
# n = 10,0000 ::: time = 0.011543 seconds
# sample mean = 4.9647655815252465; theoretical mean = 5.0
# sample var = 28.68846969375196; theoretical var = 29.0

# ### EXPONENTIAL test
# @testset "Exponential(θ = 20) * (c0 + c1 * x + c2 * x^2);" begin
# Random.seed!(1234)
# Γ = rand(5, 5)
# res = Vector{Float64}(undef, 5)
# dist = Exponential(20)

# # test if the proper c0, c1, c2 constants are stored.
# d_exp = pdf_constants(Γ, res, 1, dist)
# @test d_exp.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_exp.d)^2 * inv(var(d_exp.d)) * 0.5 * Γ[1,1])
# @test d_exp.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_exp.d) * inv(var(d_exp.d)))
# @test d_exp.c2 == 0.5 * Γ[1, 1] * (inv(var(d_exp.d)))

# # test if the CDF is a proper density (integrates to 1)
# θ = params(d_exp.d)[1]
# normalizing_c1 = θ * (StatsFuns.gamma(2)/ StatsFuns.gamma(1))
# normalizing_c2 = θ^2 * (StatsFuns.gamma(3)/ StatsFuns.gamma(1))
# @test d_exp.c * (d_exp.c0 + normalizing_c1 * d_exp.c1 + normalizing_c2 * d_exp.c2) == 1.0

# @test d_exp.c ≈ inv(1+ 0.5 * tr(Γ))
# @test minimum(d_exp) == 0
# @test maximum(d_exp) == Inf
# @test cdf(d_exp, 0) == 0
# @test cdf(d_exp, Inf) == 1

# ###
# Random.seed!(1234)
# nsample = 10_000 #_000
# @info "sample $nsample points for the $dist distribution"
# s = Vector{Float64}(undef, nsample)
# rand!(d_exp, s) # compile
# @time rand!(d_exp, s)
# println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_exp))")
# println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_exp))")
# end
# # n = 10,0000 ::: time = 0.012240 seconds
# # sample mean = 26.110450150884134; theoretical mean = 26.326541243044463
# # sample var = 856.9798254204945; theoretical var = 866.0981753436141

####
#########  Beta test using bisection ############
@testset "Beta(α = 2.0, β = 3.0) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
α, β = (2.0, 3.0)
dist = Beta(α, β)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_beta = pdf_constants(γ, dist)
@test d_beta.c ≈ inv(1 + 0.5 * γ)
@test d_beta.c0 == 1 + (mean(d_beta.d)^2 * inv(var(d_beta.d)) * 0.5 * γ)
@test d_beta.c1 == 0.5 * γ * (-2 * mean(d_beta.d) * inv(var(d_beta.d)))
@test d_beta.c2 == 0.5 * γ * (inv(var(d_beta.d)))

# test if the CDF is a proper density (integrates to 1)
α, β = Distributions.params(d_beta.d)
normalizing_c1 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 1)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 1))
normalizing_c2 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 2)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 2))
@test d_beta.c * (d_beta.c0 + normalizing_c1 * d_beta.c1 + normalizing_c2 * d_beta.c2) ≈  1.0
@test minimum(d_beta) == 0
@test maximum(d_beta) ≈ 1
@test cdf(d_beta, 0) == 0
@test cdf(d_beta, 1) ≈ 1

###
# Random.seed!(1234)
# nsample = 10_000 #
# @info "sample $nsample points for the $dist distribution using the Bisection method."
# s = Vector{Float64}(undef, nsample)
# rand!(d_beta, s) # compile
# @time rand!(d_beta, s)
# println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_beta))")
# println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_beta))")
end

#########  Beta test ############
# @testset "Beta(α = 0.5, β = 3.0) * (c0 + c1 * x + c2 * x^2);" begin
# Random.seed!(123)
# α, β = (0.5, 3.0)
# dist = Beta(α, β)
#
# # test if the proper c0, c1, c2 constants are stored.
# γ = 1.0
# d_beta = pdf_constants(γ, dist)
# @test d_beta.c ≈ inv(1 + 0.5 * γ)
# @test d_beta.c0 == 1 + (mean(d_beta.d)^2 * inv(var(d_beta.d)) * 0.5 * γ)
# @test d_beta.c1 == 0.5 * γ * (-2 * mean(d_beta.d) * inv(var(d_beta.d)))
# @test d_beta.c2 == 0.5 * γ * (inv(var(d_beta.d)))
#
# # test if the CDF is a proper density (integrates to 1)
# α, β = Distributions.params(d_beta.d)
# normalizing_c1 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 1)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 1))
# normalizing_c2 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 2)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 2))
# @test d_beta.c * (d_beta.c0 + normalizing_c1 * d_beta.c1 + normalizing_c2 * d_beta.c2) == 1.0
# @test minimum(d_beta) == 0
# @test maximum(d_beta) ≈ 1
# @test cdf(d_beta, 0) == 0
# @test cdf(d_beta, 1) ≈ 1
#
# ###
# Random.seed!(1234)
# nsample = 10_000 #
# @info "sample $nsample points for the $dist distribution using the Bisection method."
# s = Vector{Float64}(undef, nsample)
# rand!(d_beta, s) # compile
# @time rand!(d_beta, s)
# println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_beta))")
# println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_beta))")
end
