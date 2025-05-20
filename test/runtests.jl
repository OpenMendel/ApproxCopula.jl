module PkgTest
using Test
@testset "Generating Random Deviates" begin
    include("unit_test/generate_random_deviates/rand_continuous.jl")
    include("unit_test/generate_random_deviates/rand_discrete.jl")
    include("unit_test/generate_random_deviates/rand_multivariate.jl")
end

## VCM
@testset "VCM Covariance" begin
    include("unit_test/VCM/VCM_model_interface.jl")
    include("unit_test/VCM/singlerun_nbVCM.jl")
    include("unit_test/VCM/singlerun_bernoulliVCM.jl")
    include("unit_test/VCM/singlerun_normalVCM.jl")
    include("unit_test/VCM/singlerun_poissonVCM.jl")
end

## AR
@testset "AR Covariance" begin
    include("unit_test/AR/AR_model_interface.jl")
    include("unit_test/AR/singlerun_nbAR.jl")
    include("unit_test/AR/singlerun_bernoulliAR.jl")
    include("unit_test/AR/singlerun_normalAR.jl")
    include("unit_test/AR/singlerun_poissonAR.jl")
end

## CS
@testset "CS Covariance" begin
    include("unit_test/CS/CS_model_interface.jl")
    include("unit_test/CS/singlerun_nbCS.jl")
    include("unit_test/CS/singlerun_bernoulliCS.jl")
    include("unit_test/CS/singlerun_normalCS.jl")
    include("unit_test/CS/singlerun_poissonCS.jl")
end

## GWAS
@testset "GWAS" begin
    include("unit_test/GWAS/tests.jl")
end

end
