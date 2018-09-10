using ApproxFun, MultivariateOrthogonalPolynomials, Plots
 import MultivariateOrthogonalPolynomials: DirichletTriangle, Vec


# Neumann
d = Triangle()
S = DirichletTriangle{1,1,1}(d)
Dx = Derivative(S,[1,0])
Dy = Derivative(S,[0,1])
B₁ = I : S → Legendre(Vec(0,0) .. Vec(0,1))
B₂ = I : S → Legendre(Vec(0,0) .. Vec(1,0))
B₃ = I : S → Legendre(Vec(0,1) .. Vec(1,0))

L = [0 B₁ 0;
 0  0  B₂;
 0  B₃ B₃;
 Dx -eye(S) 0;
  Dy 0 -eye(S);
   0.1eye(S) Dx Dy]


f = Fun((x,y) -> exp(x*cos(y)), d)

u = \(L , [0;0; 0; 0;0; f]; tolerance=1E-5)
plot(u[1])



d = Triangle() , Triangle(Vec(1,1),Vec(1,0),Vec(0,1))
S₁,S₂ = S = DirichletTriangle{1,1,1}.(d)
Dx₁ = Derivative(S₁,[1,0])
Dy₁ = Derivative(S₁,[0,1])
Dx₂ = Derivative(S₂,[1,0])
Dy₂ = Derivative(S₂,[0,1])
B₁₁ = I : S[1] → Legendre(Vec(0,0) .. Vec(0,1))
B₁₂ = I : S[1] → Legendre(Vec(0,0) .. Vec(1,0))
B₁₃ = I : S[1] → Legendre(Vec(0,1) .. Vec(1,0))
B₂₁ = I : S[2] → Legendre(Vec(1,1) .. Vec(0,1))
B₂₂ = I : S[2] → Legendre(Vec(1,1) .. Vec(1,0))
B₂₃ = I : S[2] → Legendre(Vec(0,1) .. Vec(1,0))
I₁ = eye(S₁)
I₂ = eye(S₂)

L = [B₁₁  0    0    0    0    0
     B₁₂  0    0    0    0    0
     0    0    0    B₂₁  0    0
     0    0    0    B₂₂  0    0
     B₁₃  0    0   -B₂₃  0    0
     0    B₁₃  B₁₃  0   -B₂₃ -B₂₃
     Dx₁ -I₁   0    0    0    0
     Dy₁  0   -I₁   0    0    0
     0    0    0    Dx₂ -I₂   0
     0    0    0    Dy₂  0   -I₂
     0    Dx₁  Dy₁  0    0    0
     0    0    0    0    Dx₂  Dy₂]

rs = rangespace(L)
f = vcat(map(r->Fun((x,y) -> real(exp(x+im*y)), r,20), rs[1:4])...)

u = \(L, [f; zeros(8)...]; tolerance=1E-5)


plot(u[1])
 plot!(u[4])

plot(u[4])
@which ApproxFun.plotptsvals(u[4])

ncoefficients(u[4])
@which points(u[4].space,210)
points(u[4].space,ncoefficients(u[4]))
pts=points(JacobiTriangle(0,0,0),210)
fromcanonical(S,pts[1])
K = S
n = 210
map(Vec,map(vec,points(ProductTriangle(K),round(Int,sqrt(n)),round(Int,sqrt(n))))...)
S = JacobiTriangle(0,0,0,d[2])
fromcanonical(S, Vec(0.1,0.2))
u[1](0.1,0.2)-real(exp(0.1+0.2im)) # ~ 4.5E-16
u[4](0.6,0.7)-real(exp(0.6+0.7im)) # ~ 6.7E-16
