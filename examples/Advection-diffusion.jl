using ApproxFun, MultivariateOrthogonalPolynomials, FixedSizeArrays, Plots, GLVisualize
    import MultivariateOrthogonalPolynomials: plan_evaluate


PT = ProductTriangle(1,1,1)
    X, Y = points(PT,10,10)
    X = Float32.(X)
    Y = Float32.(Y)

S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))
u0=10Fun(S,randn(10));
    P = plan_evaluate(u0)
    Z = Matrix{Float32}(P.(X,Y))



window = glscreen()
_view(visualize((X,Y,Z), :surface))
@async renderloop(window)

plot(pad(u0,100))

f=Fun((x,y)->exp(-x^2-y^2*sin(y)),Chebyshev()^2,100)
plot(f)
ncoefficients(f)

f=pad(f,105)
pts = points(f)
plot(first.(pts),last.(pts),values(f))


ApproxFun.nblocks(f)

pts = points(f)
values(f)
ApproxFun.coefficientmatrix(f)
surface(first.(pts),last.(pts),va

f.(pts)

@which points(space(f),10)

@time ApproxFun.itransform(space(f),coefficients(f))


surface(u0)

@profile values(u0)

Profile.print()

pts=points(u0)

P = MultivariateOrthogonalPolynomials.plan_evaluate(u0)


@time P.plan.(pts)



@time

ApproxFun.weight.(S,[Vec(0.1,0.2),Vec(0.1,0.2)])

P(pts)


Δ = Laplacian(S)
Dx = Derivative(S,[1,0])
Dy = Derivative(S,[0,1])

h=0.001
ε=0.01


A=I-h*(ε*Δ+Dx)

QR=qrfact(A)
