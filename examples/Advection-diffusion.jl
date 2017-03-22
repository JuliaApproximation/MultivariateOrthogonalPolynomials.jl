using ApproxFun, MultivariateOrthogonalPolynomials, Plots, GLVisualize
    import MultivariateOrthogonalPolynomials: plan_evaluate


window = glscreen()
    @async GLWindow.waiting_renderloop(window)


S = WeightedTriangle(1,1,1)

Δ = Laplacian(S)
Dx = Derivative(S,[1,0])
Dy = Derivative(S,[0,1])


h=0.0001
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


# Initial condition

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


Z


for k=1:500
    u=\(QR,C*u;tolerance=1E-5)
    chop!(u,1E-7)
    if mod(k,2) == 0
        P = plan_evaluate(u)
        Z = Matrix{Float32}(P.(X,Y))
        GLAbstraction.set_arg!(vis, :position_z, Z)
    end
    yield()
end
