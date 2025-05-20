using ApproxCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns
using LinearAlgebra: BlasReal, copytri!
### MVN only
@testset "Generate a 3 element random vector each with Y1, Y2, Y3 ~ Normal(5, 0.2). First we check the conditional mean and variance of Y2 | Y1 and then for Y3 | Y1, Y2. " begin
Random.seed!(12345)
mean_normal = 5
var_normal = 0.2
d1 = Normal(mean_normal, var_normal)
d2 = Normal(mean_normal, var_normal)
d3 = Normal(mean_normal, var_normal)

vecd = [d1, d2, d3]
variance_component_1 = 0.1
Γ = variance_component_1 * ones(3, 3)

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

#### Testing if the second element in the bivariate poisson follows correct mean and variance
# preallocate Y and res to simulate and store
n = 3
Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)

# simulate Y1 first
i = 1
# form constants for Y1 pdf
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# simulate Y1 and store it to Y[1]
Random.seed!(12345)
Y[i] = rand(nonmixed_multivariate_dist.gc_obs[i])
# update residuals res[1] using Y[1] and the mean and variance (lambda for Poisson) using GLM package
res[i] = update_res!(Y[i], res[i], nonmixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 2
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])

# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Normal distribution"
s = Vector{Float64}(undef, nsample)
Random.seed!(12345)
rand!(nonmixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(nonmixed_multivariate_dist.gc_obs[i], s) # get time
# 0.032922 seconds (1 allocation: 32 bytes)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(nonmixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(nonmixed_multivariate_dist.gc_obs[i]))")



# conditional mean and variance of Y2 given Y1
# sample mean = 5.005900458369502; theoretical mean = 5.002996732265867
# sample var = 0.04316391761735573; theoretical var = 0.04362288747145726

## now testing conditional Y3 given Y1, Y2
# simulate Y2 first
i = 2
# form constants for Y1 pdf
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# simulate Y1 and store it to Y[1]
Random.seed!(12345)
Y[i] = rand(nonmixed_multivariate_dist.gc_obs[i])
# update residuals res[2] using Y[2] and the mean and variance (lambda for Poisson) using GLM package
res[i] = update_res!(Y[i], res[i], nonmixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 3
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Normal distribution"
s = Vector{Float64}(undef, nsample)
Random.seed!(12345)
rand!(nonmixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(nonmixed_multivariate_dist.gc_obs[i], s) # get time
# 0.040553 seconds (1 allocation: 32 bytes)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(nonmixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(nonmixed_multivariate_dist.gc_obs[i]))")
# conditional mean and variance of Y3 given Y1, Y2
# sample mean = 5.006034080584214; theoretical mean = 5.003125498930692
# sample var = 0.04331760554438279; theoretical var = 0.043778156954903835
# rand(mvn_dist, Y, res)
end

### Poisson only
@testset "Generate a 3 element random vector each with Y1, Y2, Y3 ~ Poisson(5). First we check the conditional mean and variance of Y2 | Y1 and then for Y3 | Y1, Y2. " begin
# variance components
Random.seed!(12345)
# simulate desired variance components (coefficients of covariance matrices V[k] for k in 1:m)
mean1 = 5
n = 3
d1 = Poisson(mean1)
d2 = Poisson(mean1)
d3 = Poisson(mean1)

vecd = [d1, d2, d3]
random_intercept_1 = 0.1
Γ = random_intercept_1 * ones(n, n)

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

#### Testing if the second element in the bivariate poisson follows correct mean and variance
# preallocate Y and res to simulate and store
Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)

# simulate Y1 first
i = 1
# form constants for Y1 pdf
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# simulate Y1 and store it to Y[1]
Random.seed!(12345)
Y[i] = rand(nonmixed_multivariate_dist.gc_obs[i])
# update residuals res[1] using Y[1] and the mean and variance (lambda for Poisson) using GLM package
res[i] = update_res!(Y[i], res[i], nonmixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 2
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])

# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Poisson distribution"
s = Vector{Int64}(undef, nsample)
Random.seed!(12345)
rand!(nonmixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(nonmixed_multivariate_dist.gc_obs[i], s) # get time
# 0.019684 seconds (50.00 k allocations: 12.207 MiB)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(nonmixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(nonmixed_multivariate_dist.gc_obs[i]))")
# conditional mean and variance of Y2 given Y1
# sample mean = 5.2485; theoretical mean = 5.219298245614036
# sample var = 5.689716721672168; theoretical var = 5.6098030163127035

## now testing conditional Y3 given Y1, Y2
# simulate Y2 first
i = 2
# form constants for Y1 pdf
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# simulate Y1 and store it to Y[1]
Random.seed!(12345)
Y[i] = rand(nonmixed_multivariate_dist.gc_obs[i])
# update residuals res[2] using Y[2] and the mean and variance (lambda for Poisson) using GLM package
res[i] = update_res!(Y[i], res[i], nonmixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 3
nonmixed_multivariate_dist.gc_obs[i] = pdf_constants(nonmixed_multivariate_dist.Γ, res, i, nonmixed_multivariate_dist.vecd[i])
# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Poisson distribution"
s = Vector{Int64}(undef, nsample)
Random.seed!(12345)
rand!(nonmixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(nonmixed_multivariate_dist.gc_obs[i], s) # get time
# 0.024654 seconds (50.00 k allocations: 12.207 MiB)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(nonmixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(nonmixed_multivariate_dist.gc_obs[i]))")
# conditional mean and variance of Y3 given Y1, Y2
# sample mean = 5.2381; theoretical mean = 5.206611570247935
# sample var = 5.658774267426744; theoretical var = 5.577146369783474
end

@testset "Form mixtures of different discrete and cts distributions. Generate 3 element vector: Y1 ~ Normal(5, 0.2), Y2 ~ Exponential(1/5), Y3 ~ Poisson(5) " begin
Random.seed!(12345)
mean_normal = -5
var_normal = 0.2

d1 = Normal(mean_normal, var_normal)
d2 = Exponential(-1/mean_normal)
d3 = Poisson(exp(mean_normal))

vc = 0.1
Γ = vc * ones(3, 3)

vecd = [d1, d2, d3]
mixed_multivariate_dist = MultivariateMix(vecd, Γ)

# Y = Vector{Float64}(undef, n)
# res = Vector{Float64}(undef, n)
# rand(mixed_multivariate_dist, Y, res)
#### Testing if the second element in the bivariate poisson follows correct mean and variance
# preallocate Y and res to simulate and store
n = 3
Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)

# simulate Y1 first
i = 1
# form constants for Y1 pdf
mixed_multivariate_dist.gc_obs[i] = pdf_constants(mixed_multivariate_dist.Γ, res, i, mixed_multivariate_dist.vecd[i])
# simulate Y1 and store it to Y[1]
Random.seed!(12345)
Y[i] = rand(mixed_multivariate_dist.gc_obs[i])
# update residuals res[1] using Y[1] and the mean and variance (lambda for Poisson) using GLM package
res[i] = update_res!(Y[i], res[i], mixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 2
mixed_multivariate_dist.gc_obs[i] = pdf_constants(mixed_multivariate_dist.Γ, res, i, mixed_multivariate_dist.vecd[i])

# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Exponential distribution"
s = Vector{Float64}(undef, nsample)
Random.seed!(12345)
rand!(mixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(mixed_multivariate_dist.gc_obs[i], s) # get time
# 0.030101 seconds (50.00 k allocations: 12.207 MiB)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(mixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(mixed_multivariate_dist.gc_obs[i]))")
# conditional mean and variance of Y2 given Y1
# sample mean = 0.2235739008006706; theoretical mean = 0.22115607164570328
# sample var = 0.056128853458491966; theoretical var = 0.05527858504273324

## now testing conditional Y3 given Y1, Y2
# simulate Y2 first
i = 2
# form constants for Y1 pdf
mixed_multivariate_dist.gc_obs[i] = pdf_constants(mixed_multivariate_dist.Γ, res, i, mixed_multivariate_dist.vecd[i])
# simulate Y2 and store it to Y[2]
Random.seed!(12345)
Y[i] = rand(mixed_multivariate_dist.gc_obs[i])
# update residuals res[2] using Y[2] and the mean and variance (1/5 for Exponential) using GLM package
res[i] = update_res!(Y[i], res[i], mixed_multivariate_dist.gc_obs[i])

# next, for conditional density of Y2 | Y1, form the constants for pdf
i = 3
mixed_multivariate_dist.gc_obs[i] = pdf_constants(mixed_multivariate_dist.Γ, res, i, mixed_multivariate_dist.vecd[i])
# pre allocate for sampling nsample times from conditional density
Random.seed!(12345)
nsample = 10_000
@info "sample $nsample points for the conditional Poisson distribution"
s = Vector{Int64}(undef, nsample)
Random.seed!(12345)
rand!(mixed_multivariate_dist.gc_obs[i], s) # compile
Random.seed!(12345)
@time rand!(mixed_multivariate_dist.gc_obs[i], s) # get time
# 0.024654 seconds (50.00 k allocations: 12.207 MiB)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(mixed_multivariate_dist.gc_obs[i]))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(mixed_multivariate_dist.gc_obs[i]))")
# conditional mean and variance of Y3 given Y1, Y2
# sample mean = 0.0562; theoretical mean = 0.05564357782888464
# sample var = 0.053646924692469226; theoretical var = 0.0538934802364384
#####


### bivariate Poisson correlation and covariance
mean1 = 5
mean2 = 5
d1 = Poisson(mean1)
d2 = Poisson(mean2)

vecd = [d1, d2]
Γ1 = 0.1 * ones(2, 2)
Γ2 = [0.1 -0.1; -0.1 0.1]

gc_vec1 = NonMixedMultivariateDistribution(vecd, Γ1)
gc_vec2 = NonMixedMultivariateDistribution(vecd, Γ2)


nsample = 10_000
@info "sample $nsample independent vectors for the bivariate Poisson distribution"
# compile
Y_nsample1 = simulate_nobs_independent_vectors(gc_vec1, nsample)
Y_nsample2 = simulate_nobs_independent_vectors(gc_vec2, nsample)

Y_11 = [Y_nsample1[i, 1][1] for i in 1:nsample]
Y_12 = [Y_nsample1[i, 1][2] for i in 1:nsample]
Statistics.cov(Y_11, Y_12)

@show Statistics.cor(Y_11, Y_12)

Y_21 = [Y_nsample2[i, 1][1] for i in 1:nsample]
Y_22 = [Y_nsample2[i, 1][2] for i in 1:nsample]
Statistics.cov(Y_21, Y_22)

@show Statistics.cor(Y_21, Y_22)

@test ApproxCopula.cov(gc_vec1, 1, 2) == -ApproxCopula.cov(gc_vec2, 1, 2)
@test ApproxCopula.cor(gc_vec1, 1, 2) == -ApproxCopula.cor(gc_vec2, 1, 2)

###
# bivariate mixed normal and poisson with theoretical correlation ~
##
mean_normal = 5
var_normal = 0.2

mean_poisson = 7

d1 = Normal(mean_normal, var_normal)
d2 = Poisson(mean_poisson)

vc = 0.1
Γ1 = [0.1 0.1; 0.1 0.1]
Γ2 = [0.1 -0.1; -0.1 0.1]

vecd = [d1, d2]
gc_vec1 = MultivariateMix(vecd, Γ1)
gc_vec2 = MultivariateMix(vecd, Γ2)

nsample = 10_000
@info "sample $nsample independent vectors for the normal and poisson"
# compile
Y_nsample1 = simulate_nobs_independent_vectors(gc_vec1, nsample)
Y_nsample2 = simulate_nobs_independent_vectors(gc_vec2, nsample)

Y_11 = [Y_nsample1[i, 1][1] for i in 1:nsample]
Y_12 = [Y_nsample1[i, 1][2] for i in 1:nsample]
Statistics.cov(Y_11, Y_12)

@show Statistics.cor(Y_11, Y_12)
ApproxCopula.cov(gc_vec1, 1, 2)
ApproxCopula.cor(gc_vec1, 1, 2)
println("sample cor = $(Statistics.cor(Y_11, Y_12)); theoretical cor = $(ApproxCopula.cor(gc_vec1, 1, 2))")
println("sample cov = $(Statistics.cov(Y_11, Y_12)); theoretical var = $(ApproxCopula.cov(gc_vec1, 1, 2))")


Y_21 = [Y_nsample2[i, 1][1] for i in 1:nsample]
Y_22 = [Y_nsample2[i, 1][2] for i in 1:nsample]
Statistics.cov(Y_21, Y_22)

@show Statistics.cor(Y_21, Y_22)

ApproxCopula.cov(gc_vec2, 1, 2)
ApproxCopula.cor(gc_vec2, 1, 2)
println("sample cor = $(Statistics.cor(Y_21, Y_22)); theoretical cor = $(ApproxCopula.cor(gc_vec2, 1, 2))")
println("sample cov = $(Statistics.cov(Y_21, Y_22)); theoretical var = $(ApproxCopula.cov(gc_vec2, 1, 2))")


@test ApproxCopula.cov(gc_vec1, 1, 2) == -ApproxCopula.cov(gc_vec2, 1, 2)
@test ApproxCopula.cor(gc_vec1, 1, 2) == -ApproxCopula.cor(gc_vec2, 1, 2)
####
# mixed with normal and 0, 1
mean_normal = 5
var_normal = 0.2

mean_bernoulli = 0.5

d1 = Normal(mean_normal, var_normal)
d2 = Bernoulli(mean_bernoulli)

vc = 0.1
Γ1 = [0.1 0.1; 0.1 0.1]
Γ2 = [0.1 -0.1; -0.1 0.1]

vecd = [d1, d2]
gc_vec1 = MultivariateMix(vecd, Γ1)
gc_vec2 = MultivariateMix(vecd, Γ2)

nsample = 10_000
@info "sample $nsample independent vectors for the mixed normal and binary bivariate"
# compile
Y_nsample1 = simulate_nobs_independent_vectors(gc_vec1, nsample)
Y_nsample2 = simulate_nobs_independent_vectors(gc_vec2, nsample)

Y_11 = [Y_nsample1[i, 1][1] for i in 1:nsample]
Y_12 = [Y_nsample1[i, 1][2] for i in 1:nsample]
# @show Statistics.cov(Y_11, Y_12)

# @show Statistics.cor(Y_11, Y_12)
# @show ApproxCopula.cov(gc_vec1, 1, 2)
# @show ApproxCopula.cor(gc_vec1, 1, 2)
println("sample cor = $(Statistics.cor(Y_11, Y_12)); theoretical cor = $(ApproxCopula.cor(gc_vec1, 1, 2))")
println("sample cov = $(Statistics.cov(Y_11, Y_12)); theoretical var = $(ApproxCopula.cov(gc_vec1, 1, 2))")


Y_21 = [Y_nsample2[i, 1][1] for i in 1:nsample]
Y_22 = [Y_nsample2[i, 1][2] for i in 1:nsample]
# @show Statistics.cov(Y_21, Y_22)

# @show Statistics.cor(Y_21, Y_22)

# @show ApproxCopula.cov(gc_vec2, 1, 2)
# @show ApproxCopula.cor(gc_vec2, 1, 2)
println("sample cor = $(Statistics.cor(Y_21, Y_22)); theoretical cor = $(ApproxCopula.cor(gc_vec2, 1, 2))")
println("sample cov = $(Statistics.cov(Y_21, Y_22)); theoretical var = $(ApproxCopula.cov(gc_vec2, 1, 2))")
#

@test ApproxCopula.cov(gc_vec1, 1, 2) == -ApproxCopula.cov(gc_vec2, 1, 2)
@test ApproxCopula.cor(gc_vec1, 1, 2) == -ApproxCopula.cor(gc_vec2, 1, 2)


# mixed with normal and 0, 1 with more noise in normal
mean_normal = 5
sd_normal = 0.2

mean_bernoulli = 0.5

d1 = Bernoulli(mean_bernoulli)
d2 = Normal(mean_normal, sd_normal)

vc = 0.1
Γ1 = [0.1 0.1; 0.1 0.1]
Γ2 = [0.1 -0.1; -0.1 0.1]

vecd = [d1, d2]
gc_vec1 = MultivariateMix(vecd, Γ1)
gc_vec2 = MultivariateMix(vecd, Γ2)

nsample = 10_000
@info "sample $nsample independent vectors for the mixed normal and binary bivariate"
# compile
Y_nsample1 = simulate_nobs_independent_vectors(gc_vec1, nsample)
Y_nsample2 = simulate_nobs_independent_vectors(gc_vec2, nsample)

Y_11 = [Y_nsample1[i, 1][1] for i in 1:nsample]
Y_12 = [Y_nsample1[i, 1][2] for i in 1:nsample]
# @show Statistics.cov(Y_11, Y_12)

# @show Statistics.cor(Y_11, Y_12)
# @show ApproxCopula.cov(gc_vec1, 1, 2)
# @show ApproxCopula.cor(gc_vec1, 1, 2)
println("sample cor = $(Statistics.cor(Y_11, Y_12)); theoretical cor = $(ApproxCopula.cor(gc_vec1, 1, 2))")
println("sample cov = $(Statistics.cov(Y_11, Y_12)); theoretical cov = $(ApproxCopula.cov(gc_vec1, 1, 2))")


Y_21 = [Y_nsample2[i, 1][1] for i in 1:nsample]
Y_22 = [Y_nsample2[i, 1][2] for i in 1:nsample]
# @show Statistics.cov(Y_21, Y_22)

# @show Statistics.cor(Y_21, Y_22)

# @show ApproxCopula.cov(gc_vec2, 1, 2)
# @show ApproxCopula.cor(gc_vec2, 1, 2)
println("sample cor = $(Statistics.cor(Y_21, Y_22)); theoretical cor = $(ApproxCopula.cor(gc_vec2, 1, 2))")
println("sample cov = $(Statistics.cov(Y_21, Y_22)); theoretical cov = $(ApproxCopula.cov(gc_vec2, 1, 2))")
#

@test ApproxCopula.cov(gc_vec1, 1, 2) == -ApproxCopula.cov(gc_vec2, 1, 2)
@test ApproxCopula.cor(gc_vec1, 1, 2) == -ApproxCopula.cor(gc_vec2, 1, 2)

end
