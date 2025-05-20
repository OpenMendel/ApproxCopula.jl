using DataFrames, ApproxCopula, LinearAlgebra, GLM, RDatasets
using Test

Mmmec = dataset("mlmRev", "Mmmec");
df = Mmmec
y = :Deaths
grouping = :Region
covariates = [:UVB]

d = Poisson()
link = LogLink()

# forming CS model with Poisson base
Poisson_CS_model = CS_model(df, y, grouping, covariates, d, link)

@test typeof(Poisson_CS_model) == GLMCopulaCSModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_CS_model.penalized == false

# forming CS model with Poisson base with L2 ridge penalty
Poisson_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(Poisson_CS_model_penalized) == GLMCopulaCSModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_CS_model_penalized.penalized == true

# intercept only model
Poisson_CS_model_intercept_only = CS_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(Poisson_CS_model_intercept_only.data[1].X) == Poisson_CS_model_intercept_only.data[1].n

#### nb ####
d = NegativeBinomial()
link = LogLink()
nb_CS_model = CS_model(df, y, grouping, covariates, d, link)

@test typeof(nb_CS_model) == NBCopulaCSModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_CS_model.penalized == false

# forming CS model with nb base with L2 ridge penalty
nb_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(nb_CS_model_penalized) == NBCopulaCSModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_CS_model_penalized.penalized == true

# intercept only model
nb_CS_model_intercept_only = CS_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(nb_CS_model_intercept_only.data[1].X) == nb_CS_model_intercept_only.data[1].n

#### gaussian ####
d = Normal()
link = IdentityLink()
gaussian_CS_model = CS_model(df, y, grouping, covariates, d, link)

@test typeof(gaussian_CS_model) == GaussianCopulaCSModel{Float64}
@test gaussian_CS_model.penalized == false

# forming CS model with gaussian base with L2 ridge penalty
gaussian_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true)

@test typeof(gaussian_CS_model_penalized) == GaussianCopulaCSModel{Float64}
@test gaussian_CS_model_penalized.penalized == true

# intercept only model
gaussian_CS_model_intercept_only = CS_model(df, y, grouping, d, link)
# check if default to intercept only
@test sum(gaussian_CS_model_intercept_only.data[1].X) == gaussian_CS_model_intercept_only.data[1].n
