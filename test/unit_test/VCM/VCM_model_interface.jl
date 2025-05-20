using DataFrames, ApproxCopula, LinearAlgebra, GLM, RDatasets
using Test

Mmmec = dataset("mlmRev", "Mmmec");
df = Mmmec
y = :Deaths
grouping = :Region
covariates = [:UVB]

### Now with two variance components V1 = ones(di, di), V2 = Identity
function make_Vs(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol)
    groups = unique(df[!, grouping])
    n = length(groups)
    V = Vector{Vector{Matrix{Float64}}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        V[i] = [ones(ni, ni), Matrix(I, ni, ni)]
    end
    V
end

V = make_Vs(df, y, grouping);

#### POISSON ####
d = Poisson()
link = LogLink()
Poisson_VC_model = VC_model(df, y, grouping, covariates, d, link)

@test typeof(Poisson_VC_model) == GLMCopulaVCModel{Float64, Poisson{Float64}, LogLink}
# test that we have all ones for random intercept covariance structure by default
@test Poisson_VC_model.data[1].V[1][1] == 1.0

# two vc
Poisson_2VC_model = VC_model(df, y, grouping, covariates, V, d, link)
@test typeof(Poisson_2VC_model) == GLMCopulaVCModel{Float64, Poisson{Float64}, LogLink}
@test length(Poisson_2VC_model.data[1].V) == 2

# intercept only model
Poisson_VC_model_intercept_only = VC_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(Poisson_VC_model_intercept_only.data[1].X) == Poisson_VC_model_intercept_only.data[1].n

# intercept only model with V and penalty
Poisson_2VC_model_intercept_only = VC_model(df, y, grouping, V, d, link; penalized = true)
# check if default to intercept only
@test Poisson_2VC_model_intercept_only.penalized == true
@test sum(Poisson_2VC_model_intercept_only.data[1].X) == Poisson_2VC_model_intercept_only.data[1].n

#### nb ####
d = NegativeBinomial()
link = LogLink()
nb_VC_model = VC_model(df, y, grouping, covariates, d, link)

@test typeof(nb_VC_model) == NBCopulaVCModel{Float64, NegativeBinomial{Float64}, LogLink}
# test that we have all ones for random intercept covariance structure by default
@test nb_VC_model.data[1].V[1][1] == 1.0

# two vc
nb_2VC_model = VC_model(df, y, grouping, covariates, V, d, link)
@test typeof(nb_2VC_model) == NBCopulaVCModel{Float64, NegativeBinomial{Float64}, LogLink}
@test length(nb_2VC_model.data[1].V) == 2

# intercept only model
nb_VC_model_intercept_only = VC_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(nb_VC_model_intercept_only.data[1].X) == nb_VC_model_intercept_only.data[1].n

# intercept only model with V and penalty
nb_2VC_model_intercept_only = VC_model(df, y, grouping, V, d, link; penalized = true)
# check if default to intercept only
@test nb_2VC_model_intercept_only.penalized == true
@test sum(nb_2VC_model_intercept_only.data[1].X) == nb_2VC_model_intercept_only.data[1].n

#### gaussian ####
d = Normal()
link = IdentityLink()
gaussian_VC_model = VC_model(df, y, grouping, covariates, d, link)

@test typeof(gaussian_VC_model) == GaussianCopulaVCModel{Float64}
# test that we have all ones for random intercept covariance structure by default
@test gaussian_VC_model.data[1].V[1][1] == 1.0

# two vc
gaussian_2VC_model = VC_model(df, y, grouping, covariates, V, d, link)
@test typeof(gaussian_2VC_model) == GaussianCopulaVCModel{Float64}
@test length(gaussian_2VC_model.data[1].V) == 2

# intercept only model
gaussian_VC_model_intercept_only = VC_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(gaussian_VC_model_intercept_only.data[1].X) == gaussian_VC_model_intercept_only.data[1].n

# intercept only model with V and penalty
gaussian_2VC_model_intercept_only = VC_model(df, y, grouping, V, d, link; penalized = true)
# check if default to intercept only
@test gaussian_2VC_model_intercept_only.penalized == true
@test sum(gaussian_2VC_model_intercept_only.data[1].X) == gaussian_2VC_model_intercept_only.data[1].n
