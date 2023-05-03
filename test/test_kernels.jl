using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, Test


@testset "Kernels" begin
    P = Legendre()
    T = Chebyshev()
    P² = RectPolynomial(Fill(P, 2))
    T² = RectPolynomial(Fill(T, 2))

    k = (x,y) -> exp(x*cos(y))
    K = P * transform(P², splat(k)).array * P'
    @test (K * expand(P, exp))[0.1] ≈ (K*expand(T, exp))[0.1] ≈ 2.5521805183347417

    K = T * transform(T², splat(k)).array * T'
    @test (K * expand(P, exp))[0.1] ≈ (K*expand(T, exp))[0.1] ≈ 2.5521805183347417


    x = axes(P,1)
    @test (k.(x,x') * expand(P, exp))[0.1] ≈ (k.(x,x') *expand(T, exp))[0.1] ≈ 2.5521805183347417

    y = Inclusion(2..3)
    @test (k.(y, x') * expand(P, exp))[2.1]  ≈ (k.(y, x') * expand(T, exp))[2.1] ≈ 13.80651898993351
end