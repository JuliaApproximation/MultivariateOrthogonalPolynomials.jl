using ApproxFun, MultivariateOrthogonalPolynomials, FixedSizeArrays, Plots, GLVisualize
    import MultivariateOrthogonalPolynomials: plan_evaluate


window = glscreen()
    @async GLWindow.waiting_renderloop(window)


S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))

Δ = Laplacian(S)
Dx = Derivative(S,[1,0])
Dy = Derivative(S,[0,1])

h=0.001
ε=0.01


A=I-h*(ε*Δ+Dx)

QR=qrfact(A)
C=cache(eye(A))  # conversion operator


# Plotting grid
PT = ProductTriangle(1,1,1)
    X, Y = points(PT,30,30)
    X = Float32.(X)
    Y = Float32.(Y)

X = linspace(0f0,1f0,30)*ones(Float32,30)'
Y = X'
Y = (1-X).*Y

using Plots

scatter(X,Y)

# Initial condition


using ApproxFun


u=u0=Fun(S,4randn(10));
    P = plan_evaluate(u)
    Z = Matrix{Float32}(P.(X,Y))
    vis = visualize((X,Y,Z), :surface)
    _view(vis)

u=u0=Fun(S,4randn(10));
    P = plan_evaluate(u)
    Z = Matrix{Float32}(P.(X,Y))
    vis = visualize((X,Y,Z), :surface)
    GLAbstraction.set_arg!(vis, :position_z, Z)
    yield()



for k=1:1000
    @time u=\(QR,C*u;tolerance=1E-5)
    chop!(u,1E-7)
    if mod(k,10) == 0
        P = plan_evaluate(u)
        @time Z = Matrix{Float32}(P.(X,Y))
        GLAbstraction.set_arg!(vis, :position_z, Z)
    end
    yield()
end


f = Fun((x,y)->exp(-x^2*cos(y)),KoornwinderTriangle(1,1,1))

f(0.1,0.2)-exp(-0.1^2*cos(0.2))

@time f(0.1,0.2)

ncoefficients(f)

Conversion(KoornwinderTriangle(1,1,1),KoornwinderTriangle(1,1,2))


Conversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))[Block(10),Block(11)]

Conversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))[Block(10),Block(11)]




import ApproxFun:Block
MultivariateOrthogonalPolynomials.Recurrence(2,KoornwinderTriangle(1,1,1))[Block(3),Block(3)]




MultivariateOrthogonalPolynomials.Recurrence(2,KoornwinderTriangle(1,1,1))[Block(4),Block(3)]
