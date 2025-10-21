using ClassicalOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StatsBase, Plots

x,y = coordinates(ChebyshevInterval() ^ 2)
qr([one(x) x y])