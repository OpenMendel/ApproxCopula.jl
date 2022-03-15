"Check the number of clusters in the sample."
nclusters(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel, Poisson_Bernoulli_VCModel, GLMCopulaARModel, NBCopulaARModel, GaussianCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel, GaussianCopulaCSModel}) = length(model.data)

"Check minimum cluster size."
minclustersize(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel, GLMCopulaARModel, NBCopulaARModel, GaussianCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel, GaussianCopulaCSModel}) = minimum(model.data[i].n for i in 1:nclusters(model))

"Check maximum cluster size."
maxclustersize(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel, GLMCopulaARModel, NBCopulaARModel, GaussianCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel, GaussianCopulaCSModel}) = maximum(model.data[i].n for i in 1:nclusters(model))

"Check the number of variance components."
nvc(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel, Poisson_Bernoulli_VCModel}) = model.m

"Check the number of fixed effects to estimate."
nfixedeffects(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel, Poisson_Bernoulli_VCModel, GLMCopulaARModel, NBCopulaARModel, GaussianCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel, GaussianCopulaCSModel}) = model.p

"Check the base distribution."
basedistribution(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GLMCopulaARModel, NBCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel}) = Base.typename(typeof(unique(model.d)[1])).wrapper

"Check the link function."
linkfunction(model::Union{GLMCopulaVCModel, NBCopulaVCModel, GLMCopulaARModel, NBCopulaARModel, GLMCopulaCSModel, NBCopulaCSModel}) = Base.typename(typeof(unique(model.link)[1])).wrapper

# gaussian base
"Check the base distribution."
basedistribution(model::Union{GaussianCopulaVCModel, GaussianCopulaARModel, GaussianCopulaCSModel}) = Base.typename(typeof(Normal())).wrapper

"Check the link function."
linkfunction(model::Union{GaussianCopulaVCModel, GaussianCopulaARModel, GaussianCopulaCSModel}) = Base.typename(typeof(IdentityLink())).wrapper

# better printing; customize how a type is summarized in a REPL
function Base.show(io::IO, x::Union{GLMCopulaVCModel, NBCopulaVCModel, GaussianCopulaVCModel})
    print(io, "Quasi-Copula Variance Component Model\n")
    print(io, "  * base distribution: $(basedistribution(x))\n")
    print(io, "  * link function: $(linkfunction(x))\n")
    print(io, "  * number of clusters: $(nclusters(x))\n")
    print(io, "  * cluster size min, max: $(minclustersize(x)), $(maxclustersize(x))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
    print(io, "  * number of fixed effects: $(nfixedeffects(x))\n")
    if x.penalized == true
        print(io, "  * L2 ridge penalty on variance components: $(x.penalized)")
    end
end

function Base.show(io::IO, x::Union{GLMCopulaARModel, NBCopulaARModel, GaussianCopulaARModel})
    print(io, "Quasi-Copula Autoregressive AR(1) Model\n")
    print(io, "  * base distribution: $(basedistribution(x))\n")
    print(io, "  * link function: $(linkfunction(x))\n")
    print(io, "  * number of clusters: $(nclusters(x))\n")
    print(io, "  * cluster size min, max: $(minclustersize(x)), $(maxclustersize(x))\n")
    print(io, "  * number of fixed effects: $(nfixedeffects(x))\n")
    if x.penalized == true
        print(io, "  * L2 ridge penalty on AR(1) variance parameter: $(x.penalized)")
    end
end

function Base.show(io::IO, x::Union{GLMCopulaCSModel, NBCopulaCSModel, GaussianCopulaCSModel})
    print(io, "Quasi-Copula Compound Symmetric CS Model\n")
    print(io, "  * base distribution: $(basedistribution(x))\n")
    print(io, "  * link function: $(linkfunction(x))\n")
    print(io, "  * number of clusters: $(nclusters(x))\n")
    print(io, "  * cluster size min, max: $(minclustersize(x)), $(maxclustersize(x))\n")
    print(io, "  * number of fixed effects: $(nfixedeffects(x))\n")
    if x.penalized == true
        print(io, "  * L2 ridge penalty on CS variance parameter: $(x.penalized)")
    end
end

# bivariate mixed vc
"Check cluster size."
clustersize(model::Poisson_Bernoulli_VCModel) = unique(model.data[i].n for i in 1:nclusters(model))[1]

"Check all the base distributions."
basedistributions(model::Poisson_Bernoulli_VCModel) = [Base.typename.(typeof.(unique(model.vecd)[1][i])).wrapper for i in 1:length(unique(model.vecd[1]))]

"Check all the link functions."
linkfunctions(model::Poisson_Bernoulli_VCModel) = [Base.typename.(typeof.(unique(model.veclink)[1][i])).wrapper for i in 1:length(unique(model.veclink[1]))]

# better printing; customize how a type is summarized in a REPL
function Base.show(io::IO, x::Poisson_Bernoulli_VCModel)
    print(io, "Bivariate Mixed Quasi-Copula Variance Component Model\n")
    print(io, "  * base Distributions for (y_1, y_2): ($(basedistributions(x)[1]), $(basedistributions(x)[2]))\n")
    print(io, "  * link functions for (y_1, y_2): ($(linkfunctions(x)[1]), $(linkfunctions(x)[2]))\n")
    print(io, "  * number of clusters: $(nclusters(x))\n")
    print(io, "  * cluster size: $(clustersize(x))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
    print(io, "  * number of fixed effects: $(nfixedeffects(x))")
end
