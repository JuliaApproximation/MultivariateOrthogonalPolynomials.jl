using ApproxFun, MultivariateOrthogonalPolynomials, Plots, BlockArrays
    import MultivariateOrthogonalPolynomials: Vec, DirichletTriangle


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
\

h=0.01;
 (u[1](0.5,0.5)-u[1](0.5-h,0.5-h))/h
plot(Dx*u[1])
plot(Dy*u[1])
plot((Dx+Dy)*u[1])
plot(u[3])

rhs = Fun([0;0; 0; 0;0; f], rangespace(L))

A = Matrix(L[Block.(1:10), Block.(1:10)])

u = svdfact(A) \ pad(rhs.coefficients, size(A,1))

Laplacian()*Fun(domainspace(L),u)[1] - f |> coefficients


Dx*Fun(domainspace(L),u)[1]  - Fun(domainspace(L),u)[2] |> coefficients

Dx*Fun(domainspace(L),u)[2] + Dy*Fun(domainspace(L),u)[3] - f |> coefficients



plot(Fun(domainspace(L),u)[1])

u
[Dx eye(S)]

L = []

[Dx[

# D to N
d = Triangle() , Triangle(Vec(1,1),Vec(0,1),Vec(1,0))
S = DirichletTriangle{1,1,1}.(d)
B₁₁ = I : S[1] → Legendre(Vec(0,0) .. Vec(0,1))
B₁₂ = I : S[1] → Legendre(Vec(0,0) .. Vec(1,0))
B₂₁ = I : S[2] → Legendre(Vec(1,1) .. Vec(0,1))
B₂₂ = I : S[2] → Legendre(Vec(1,1) .. Vec(1,0))
