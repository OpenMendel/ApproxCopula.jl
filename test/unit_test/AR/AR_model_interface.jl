using DataFrames, ApproxCopula, LinearAlgebra, GLM, RDatasets
using Test

Mmmec = dataset("mlmRev", "Mmmec");
df = Mmmec
y = :Deaths
grouping = :Region
covariates = [:UVB]

d = Poisson()
link = LogLink()

#### POISSON ####
# forming AR(1) model with Poisson base
Poisson_AR_model = AR_model(df, y, grouping, covariates, d, link)

@test typeof(Poisson_AR_model) == GLMCopulaARModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_AR_model.penalized == false

# forming AR(1) model with Poisson base with L2 ridge penalty
Poisson_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(Poisson_AR_model_penalized) == GLMCopulaARModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_AR_model_penalized.penalized == true

# intercept only model
Poisson_AR_model_intercept_only = AR_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(Poisson_AR_model_intercept_only.data[1].X) == Poisson_AR_model_intercept_only.data[1].n

#### nb ####
d = NegativeBinomial()
link = LogLink()
nb_AR_model = AR_model(df, y, grouping, covariates, d, link)

@test typeof(nb_AR_model) == NBCopulaARModel{Float64, NegativeBinomial{Float64}, LogLink}
@test Poisson_AR_model.penalized == false

# forming AR(1) model with nb base with L2 ridge penalty
nb_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(nb_AR_model_penalized) == NBCopulaARModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_AR_model_penalized.penalized == true

# intercept only model
nb_AR_model_intercept_only = AR_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(nb_AR_model_intercept_only.data[1].X) == nb_AR_model_intercept_only.data[1].n

#### gaussian ####
d = Normal()
link = IdentityLink()
gaussian_AR_model = AR_model(df, y, grouping, covariates, d, link)

@test typeof(gaussian_AR_model) == GaussianCopulaARModel{Float64}
@test gaussian_AR_model.penalized == false

# forming AR(1) model with gaussian base with L2 ridge penalty
gaussian_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(gaussian_AR_model_penalized) == GaussianCopulaARModel{Float64}
@test gaussian_AR_model_penalized.penalized == true

# intercept only model
gaussian_AR_model_intercept_only = AR_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(gaussian_AR_model_intercept_only.data[1].X) == gaussian_AR_model_intercept_only.data[1].n
