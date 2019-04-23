using ApproxFun, MultivariateOrthogonalPolynomials, Test, FastTransforms
    import MultivariateOrthogonalPolynomials: isphericalcoordinates, DoubleWrappedSphere



S = DoubleWrappedSphere()
@test domain(S) == Sphere()



S = ProductSphericalHarmonics()

@test domain(S) == Sphere()
@test norm.(ApproxFun.checkpoints(Sphere())) ≈ [1,1,1]
@test norm.(ApproxFun.checkpoints(S)) ≈ [1,1,1]
@test norm.(Vec.(points(ProductSphericalHarmonics(), 10,10)...)) ≈ ones(10,10)

f = ProductFun((x,y,z) -> 1, S, 10,10)
@test ApproxFun.coefficients(f) == ones(1,1)

x,y,z = qrfact(randn(3,3))[:Q][:,1] # random point on sphere
@test f(x,y,z) ≈ 1

f = ProductFun((x,y,z) -> x, S, 10,10)
@test ApproxFun.coefficients(f) ≈ [0 0.5 0.5]
@test f(x,y,z) ≈ x

f = ProductFun((x,y,z) -> y, S, 10,10)
@test ApproxFun.coefficients(f) ≈ [0 0.5im -0.5im]
@test f(x,y,z) ≈ y

f = ProductFun((x,y,z) -> z, S, 10,10)
@test ApproxFun.coefficients(f) ≈ [0,1]''
@test f(x,y,z) ≈ z

#
# using FastTransforms
#
#
# f = (x,y,z) -> 1+x+2y+3z+4x^2+5x*y+7x*z+8y^2+9y*z
# pf = ProductFun((θ,φ) -> f(isphericalcoordinates(θ,φ)...) , Fourier()*Laurent())
# @test pf(sphericalcoordinates(x,y,z)...) ≈ x
#
# coefficients(pf) |> chopm
#
#
# cfs = fourier2sph(pad(coefficients(pf),5,5))
