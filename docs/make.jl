using Documenter
using ApproxCopula

makedocs(
    sitename = "ApproxCopula",
    format = Documenter.HTML(),
    modules = [ApproxCopula],
    pages = [
        "Home" => "index.md",
        "API" => "man/api.md",
        "AR(1) Covariance" => "man/AR_Examples.md",
        "CS Covariance" => "man/CS_Examples.md",
        "VC Covariance" => "man/VC_Examples.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo   = "github.com/OpenMendel/ApproxCopula.jl.git",
    target = "build"
)
