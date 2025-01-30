using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test

P = Legendre()
P² = RectPolynomial(P,P)
p₀ = expand(P², 𝐱 -> 1)
x,y = coordinates(P²)
w = P²/P²\ (x-y).^2

w .* p₀


sum(p₀)
using LazyBandedMatrices
KronTrav(sum(P;dims=1), sum(P;dims=1))

T = ChebyshevT()
U = ChebyshevU()
KronTrav(sum(T;dims=1), sum(Jacobi(1,1);dims=1))



x = axes(P,1)
X = zeros(∞,∞); X[1,1] = 1;
p₀ = P*X*P'
@test sum(p₀) == 4

sum(x.^2 .* p₀)

P\F/P'