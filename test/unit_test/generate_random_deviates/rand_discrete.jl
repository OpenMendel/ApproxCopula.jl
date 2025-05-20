using ApproxCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

### Poisson ###
@testset "Poisson(θ) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(1234)
dist = Poisson(5)
# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_pois = pdf_constants(γ, dist)

@test d_pois.c0 == 1 + (mean(d_pois.d)^2 * inv(var(d_pois.d)) * 0.5 * γ)
@test d_pois.c1 == 0.5 * γ * (-2 * mean(d_pois.d) * inv(var(d_pois.d)))
@test d_pois.c2 == 0.5 * γ * (inv(var(d_pois.d)))

pmf_copula!(d_pois)
reordered_k, reordered_pmf = reorder_pmf(d_pois.pmf_vec, d_pois.μ)
sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000
@info "sample $nsample points for the $dist distribution"
s = Vector{Int64}(undef, nsample)
rand!(d_pois, s) # compile
@time rand!(d_pois, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(d_pois))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(d_pois))")
end
# n = 10,0000 ::: time = 0.019234 seconds
# sample mean = 5.315; theoretical mean = 5.333333333333333
# sample var = 8.48922392239224; theoretical var = 8.555555555555557

# ### Binomial ###
@testset "Binomial(n, p) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
n = 30
p = 0.1
dist = Binomial(n, p)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_binomial = pdf_constants(γ, dist)

@test d_binomial.c ≈ inv(1 + 0.5 * γ)
@test d_binomial.c0 == 1 + (mean(d_binomial.d)^2 * inv(var(d_binomial.d)) * 0.5 * γ)
@test d_binomial.c1 == 0.5 * γ * (-2 * mean(d_binomial.d) * inv(var(d_binomial.d)))
@test d_binomial.c2 == 0.5 * γ * (inv(var(d_binomial.d)))

pmf_copula!(d_binomial)
reordered_k, reordered_pmf = reorder_pmf(d_binomial.pmf_vec, d_binomial.μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution using the Bisection method."
s = Vector{Int64}(undef, nsample)
rand!(d_binomial, s) # compile
@time rand!(d_binomial, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_binomial))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_binomial))")
end

# n = 10,0000 ::: time = 0.038302 seconds (40.00 k allocations: 12.817 MiB, 23.83% gc time)
# sample mean = 3.129; theoretical mean = 3.122671204329908
# sample var = 3.561715171517152; theoretical var = 3.583518347344844

# ### Geometric p = 0.2 ###
@testset "Geometric(p = 0.2) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
p = 0.2
dist = Geometric(p)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_geometric = pdf_constants(γ, dist)

@test d_geometric.c ≈ inv(1 + 0.5 * γ)
@test d_geometric.c0 == 1 + (mean(d_geometric.d)^2 * inv(var(d_geometric.d)) * 0.5 * γ)
@test d_geometric.c1 == 0.5 * γ * (-2 * mean(d_geometric.d) * inv(var(d_geometric.d)))
@test d_geometric.c2 == 0.5 * γ * (inv(var(d_geometric.d)))

pmf_copula!(d_geometric)
reordered_k, reordered_pmf = reorder_pmf(d_geometric.pmf_vec, d_geometric.μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Int64}(undef, nsample)
rand!(d_geometric, s) # compile
@time rand!(d_geometric, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_geometric))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_geometric))")
end

# n = 10,0000 ::: time = 0.032615 seconds (40.00 k allocations: 34.180 MiB, 21.35% gc time)

# ### Geometric p = 0.5###
@testset "Geometric(p = 0.5) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)

p = 0.5
dist = Geometric(p)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_geometric = pdf_constants(γ, dist)

@test d_geometric.c ≈ inv(1 + 0.5 * γ)
@test d_geometric.c0 == 1 + (mean(d_geometric.d)^2 * inv(var(d_geometric.d)) * 0.5 * γ)
@test d_geometric.c1 == 0.5 * γ * (-2 * mean(d_geometric.d) * inv(var(d_geometric.d)))
@test d_geometric.c2 == 0.5 * γ * (inv(var(d_geometric.d)))

pmf_copula!(d_geometric)
reordered_k, reordered_pmf = reorder_pmf(d_geometric.pmf_vec, d_geometric.μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Int64}(undef, nsample)
rand!(d_geometric, s) # compile
@time rand!(d_geometric, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_geometric))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_geometric))")
end

# n = 10,0000 ::: time = 0.025624 seconds (40.00 k allocations: 34.180 MiB, 33.37% gc time)

# ### Negative Binomial r = 5, p = 0.5 ###
@testset "NegativeBinomial(r = 5, p = 0.5) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
r, p = 5, 0.5
dist = NegativeBinomial(r, p)

# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_nb = pdf_constants(γ, dist)

@test d_nb.c ≈ inv(1 + 0.5 * γ)
@test d_nb.c0 == 1  + (mean(d_nb.d)^2 * inv(var(d_nb.d)) * 0.5 * γ)
@test d_nb.c1 == 0.5 * γ * (-2 * mean(d_nb.d) * inv(var(d_nb.d)))
@test d_nb.c2 == 0.5 * γ * (inv(var(d_nb.d)))

pmf_copula!(d_nb)
reordered_k, reordered_pmf = reorder_pmf(d_nb.pmf_vec, d_nb.μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Int64}(undef, nsample)
rand!(d_nb, s) # compile
@time rand!(d_nb, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(ApproxCopula.mean(d_nb))")
println("sample var = $(Statistics.var(s)); theoretical var = $(ApproxCopula.var(d_nb))")
end

# n = 10,0000 ::: time = 0.142361 seconds (40.00 k allocations: 18.921 MiB, 57.42% gc time)
### Bernoulli ###
@testset "Bernoulli(θ) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(1234)
dist = Bernoulli(0.8)
# test if the proper c0, c1, c2 constants are stored.
γ = 1.0
d_bernoulli = pdf_constants(γ, dist)

@test d_bernoulli.c0 == 1 + (mean(d_bernoulli.d)^2 * inv(var(d_bernoulli.d)) * 0.5 * γ)
@test d_bernoulli.c1 == 0.5 * γ * (-2 * mean(d_bernoulli.d) * inv(var(d_bernoulli.d)))
@test d_bernoulli.c2 == 0.5 * γ * (inv(var(d_bernoulli.d)))

pmf_copula!(d_bernoulli)
reordered_k, reordered_pmf = reorder_pmf(d_bernoulli.pmf_vec, d_bernoulli.μ)
sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000
@info "sample $nsample points for the $dist distribution"
s = Vector{Int64}(undef, nsample)
rand!(d_bernoulli, s) # compile
@time rand!(d_bernoulli, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(d_bernoulli))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(d_bernoulli))")
end
# n = 10,0000 ::: time = 0.019234 seconds
# sample mean = 5.315; theoretical mean = 5.333333333333333
# sample var = 8.48922392239224; theoretical var = 8.555555555555557
