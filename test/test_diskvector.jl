using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test, ForwardDiff, StaticArrays
using ForwardDiff: gradient


k = 0; m = 0; n = 2

Z_x = (n,m) -> (𝐱 -> gradient(𝐱 -> zernikez(n,m,𝐱), 𝐱)[1])
Z_y = (n,m) -> (𝐱 -> gradient(𝐱 -> zernikez(n,m,𝐱), 𝐱)[2])


𝐱 = SVector(0.1,0.2)
r = norm(𝐱); θ = atan(𝐱[2], 𝐱[1])
z = 2r^2 - 1
zernikez(n,m,𝐱)

W = (n,a,b) -> 2^(a+b+1)/(2n+a+b+1) * gamma(n+a+1)gamma(n+b+1)/(gamma(n+a+b+1)factorial(n))

@test jacobip(n,k, m,z) / sqrt(W(n,k,m)) ≈ normalizedjacobip(n,k,m,z)

r^m * cos(m*θ) * jacobip(n,k, m,z) / sqrt(W(n,k,m) / 2^(2+k+m))

sqrt(π) * zernikez(n,m,𝐱)

@time expand(Zernike(), Z_x(3,2))

# vector OPs
o = expand(Zernike(), _ -> 1)

v = [[o,0*o], [0*o,o]]
ip = (v,w) -> dot(v[1],w[1]) + dot(v[2],w[2])
ip(v[1],v[2])

expand(Zernike(), Z_x(3,2))


W_x = (n,m) -> (𝐱 -> gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(n,m,1,𝐱), 𝐱)[1])
W_y = (n,m) -> (𝐱 -> gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(n,m,1,𝐱), 𝐱)[2])

∇W = (n,m) -> [expand(Zernike(), W_x(n,m)),expand(Zernike(), W_y(n,m))]

ip(∇W(2,3), [expand(Zernike(), splat((x,y) -> 1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3)),expand(Zernike(), splat((x,y) -> 1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3))])

ip(,[expand(Zernike(), W_x(3,2)),expand(Zernike(), W_y(3,2))])

w = [expand(Zernike(), splat((x,y)->1-y^2)) expand(Zernike(), splat((x,y)->x*y)); expand(Zernike(), splat((x,y)->x*y)) expand(Zernike(), splat((x,y)->1-x^2))]

wiW1 = (n,m) -> expand(Zernike()[:,Block.(1:20)], splat((x,y) -> [1-x^2,-x*y]' * gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(n,m,1,𝐱), SVector(x,y))/(1-x^2-y^2)))
wiW2 = (n,m) -> expand(Zernike()[:,Block.(1:20)], splat((x,y) -> [-x*y,1-y^2]' * gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(n,m,1,𝐱), SVector(x,y))/(1-x^2-y^2)))

[wiW1(3,4),wiW2(3,4)], [expand(Zernike(), splat((x,y) -> 1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3)),expand(Zernike(), splat((x,y) -> 1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3))]

v = [wiW1(3,4),wiW2(3,4)]
[ip(v, ∇W(n,m)) for n=0:5, m=0:5]
dot(∇W(0,6)[1], ∇W(4,6)[1])
dot(∇W(0,6)[2], ∇W(4,6)[2])


∇W(0,6)[1][SVector(0.1,0.2)]
gradient(𝐱 -> (1-norm(𝐱)^2)*zernikez(0,6,1,𝐱), SVector(0.1,0.2))
ip(∇W(8,4), ∇W(9,4))
[ip(∇W(8,4), ∇W(n,m)) for n=0:10, m=0:6]
v = [wiW1(3,4),wiW2(3,4)]
[ip(v, ∇W(n,m)) for n=0:10, m=0:6]

zernikez(4,6,1,0.1*SVector(cos(0.2),sin(0.2)))
v[1][SVector(0.1,0.2)]

(1-x^2) *P^(1,1) * (1-x^2) *P^(1,1)
(𝐱 -> (1-norm(𝐱)^2)*zernikez(3,4,1,𝐱))(0.1*SVector(cos(0.2),sin(0.2)))