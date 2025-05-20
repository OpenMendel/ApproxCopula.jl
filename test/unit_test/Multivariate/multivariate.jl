@testset "utilities" begin
    # vech!(v::AbstractVector, L::Cholesky)
    n = 1000
    x = randn(n, n)
    sigma = Symmetric(x'*x, :L)
    L = cholesky(sigma)
    v1 = zeros((n * (n+1)) >> 1)
    vech!(v1, L)
    v1 = zeros((n * (n+1)) >> 1)
    vech!(v1, L)
    @test all(L.L*Transpose(L.L) .≈ sigma)
    @test all(v1 .== vech(L.L))
    b = @allocated vech!(v1, L)
    @test b == 0

    # un_vech!(A::AbstractMatrix, v::AbstractVector)
    n = 1000
    x = randn(n, n)
    v = vech(x)
    xtest = zeros(n, n)
    un_vech!(xtest, v)
    @test all(LowerTriangular(xtest) .≈ LowerTriangular(x))
    b = @allocated un_vech!(xtest, v)
    @test b == 0

    # un_vech!(L::Cholesky, v::AbstractVector)
    n = 1000
    x = randn(n, n)
    sigma = Symmetric(x'*x, :L)
    L = cholesky(sigma)
    v = vech(L.L)
    L.factors .= 0
    un_vech!(L, v)
    @test all(L.L*L.L' .≈ sigma)
    b = @allocated un_vech!(L, v)
    @test b == 0

    # copyto!(L::Cholesky, C::Cholesky)
    n = 1000
    x = randn(n, n)
    y = randn(n, n)
    sigma1 = Symmetric(x'*x, :L)
    sigma2 = Symmetric(y'*y, :L)
    L = cholesky(sigma1)
    C = cholesky(sigma2)
    copyto!(L, C)
    @test all(L.L*Transpose(L.L) .≈ sigma2)
    b = @allocated copyto!(L, C)
    @test b == 0
end
