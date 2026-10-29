using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, ContinuumArrays, BlockArrays
using BlockArrays: blockcolsupport, colsupport

W = Weighted(Ultraspherical(3/2))
W² = KronPolynomial(W, W)
Δ = -weaklaplacian(W²)

F = qr(Δ)

# f = (1+y)/2 * 

P = Legendre()
P² = KronPolynomial(P,P)
R = W²'P²
@test -laplacian(u)[SVector(0.1,0.2)] ≈ f((0.1,0.2))

D2N = zeros(500,500)
for J = 1:100
    @show J
    f = splat((x,y) -> (1+y)/2 * Ultraspherical(3/2)[x,J])
    b = R * transform(P² , f)
    c = \(F, [b[colsupport(b)]; zeros(∞)]; tolerance=1E-11)
    u = W² * c
    d = coefficients(diff(u, (0,1)))
    bcs = blockcolsupport(d)
    N = Int(last(bcs))
    D2N[1:min(N,size(D2N,1)),J] = [sum(d[Block(k)[j]] for k in j:N) for j in 1:min(N,size(D2N,1))]
end

[W \ [basis(diff(u, (0,1)))[SVector(x,1),k] for x in -1..1] for k = 1:10]

[1 0 1 0 0 1
 0 1 0 0 
 0 0 0 0 1
 0 0 0 1
]