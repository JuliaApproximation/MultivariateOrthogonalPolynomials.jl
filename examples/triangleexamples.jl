using ContinuumArrays, MultivariateOrthogonalPolynomials, BlockArrays, SpecialFunctions, FillArrays, StaticArrays, Plots


#######
# Example 1 Poisson
#######

P = JacobiTriangle()
W = WeightedTriangle(1,1,1)


𝐱 = axes(P,1)
∂ˣ = PartialDerivative{1}(𝐱)
∂ʸ = PartialDerivative{2}(𝐱)

u = expand(P, splat((x,y) -> exp(x*cos(y))))
@test (∂ˣ*u)[SVector(0.1,0.2)] ≈ cos(0.2)exp(0.1*cos(0.2))
@test (∂ʸ*u)[SVector(0.1,0.2)] ≈ -0.1*sin(0.2)exp(0.1*cos(0.2))


W = WeightedTriangle(1,1,1)
u = expand(W, splat((x,y) -> x*y*(1-x-y) * cos(x*y)))
let (x,y) = (0.1,0.2)
      @test (∂ʸ*u)[SVector(x,y)] ≈ x*(1 - x - y)*cos(x*y) - x*y*cos(x*y) - (x^2)*(1 - x - y)*y*sin(x*y) 
end


# Δ = Laplacian(𝐱)

N = 500
M = sparse(Δ[Block.(1:N), Block.(1:N)])
F = lu(M)
u_1 = F \ pad(coefficients(Fun(1, rangespace(Δ))), size(M,1))
u_2 = F \ pad(coefficients(Fun((x,y) -> x^2 + y^2, rangespace(Δ))), size(M,1))
u_3 = F \ pad(coefficients(Fun((x,y) -> x*y*(1-x-y), rangespace(Δ))), size(M,1))
u_4 = F \ pad(coefficients(Fun((x,y) -> x*y*(1-x-y) *cos(300x*y), rangespace(Δ))), size(M,1))
u_5 = F \ pad(coefficients(Fun((x,y) -> exp(-25((x-0.2)^2+(y-0.2)^2)), rangespace(Δ))), size(M,1))

gr()

plot(abs.(u_1)[1:10_000]; yscale=:log10, xscale=:log10, label="1")
plot!(abs.(u_2)[1:10_000]; yscale=:log10, xscale=:log10, label="x^2 + y^2")
plot!(abs.(u_3)[1:10_000]; yscale=:log10, xscale=:log10, label="xy(1-x-y)")
plot!(abs.(u_4)[1:10_000]; yscale=:log10, xscale=:log10, label="xy(1-x-y)cos(300xy)")

#######
# Example 2 Variable coefficient Helmholts
######

S = TriangleWeight(1,1,1,JacobiTriangle(1,1,1))
x,y = Fun(Triangle())

V = 1-(3*(x-1)^4 + 5*y^4)


f = Fun((x,y) -> x*y*exp(x) , JacobiTriangle(1.,1.,1.))

k = 100
L = Laplacian(S) + (k^2 * V)
N = ceil(Int,2k)
@time M̃ = L[Block.(1:N), Block.(1:N)]
@time M = sparse(M̃)
@time F = lu(M)
@time u_c = F \ pad(coefficients(f), size(M,1))
u = Fun(domainspace(L), u_c)


#######
# Example 3 Bi-harmonic
#######

S = TriangleWeight(2,2,2, JacobiTriangle(2,2,2))
Δ = Laplacian(S)
L = Δ^2

N = 1001
M = sparse(L[Block.(1:N), Block.(1:N)])
F = lu(M)


u_1 = F \ pad(coefficients(Fun(1, rangespace(L))), size(M,1))
u_2 = F \ pad(coefficients(Fun((x,y) -> x^2 + y^2, rangespace(L))), size(M,1))
u_3 = F \ pad(coefficients(Fun((x,y) -> x*y*(1-x-y), rangespace(L))), size(M,1))
# u_4 = F \ pad(coefficients(Fun((x,y) -> x*y*(1-x-y) *cos(300x*y), rangespace(L))), size(M,1))
u_4 = F \ pad(coefficients(Fun((x,y) -> exp(-25((x-0.2)^2+(y-0.2)^2)), rangespace(L))), size(M,1))


bnrms = u -> [norm(PseudoBlockArray(u, 1:N)[Block(K)],Inf) for K=1:N]

plot(bnrms(u_1); xscale=:log10, yscale=:log10, linewidth=3, label="1")
plot!(bnrms(u_2); xscale=:log10, yscale=:log10, linewidth=3, linestyle=:dot, label="x^2 + y^2")
plot!(bnrms(u_3); xscale=:log10, yscale=:log10, linewidth=3, linestyle=:dot, label="x^2 + y^2")
plot!(bnrms(u_4); xscale=:log10, yscale=:log10, linewidth=3, linestyle=:dot, label="x^2 + y^2")


plot(abs.(coefficients(Fun((x,y) -> exp(-25((x-0.2)^2+(y-0.2)^2)), rangespace(L)))); scale=:log10, yscale=:log10)

#######
# Example 4 Laplace's equation
#######

S = DirichletTriangle{1,1,1}()
B = Dirichlet() : S
Δ = Laplacian() : S
# L = [B; Δ]

fs = ((x,y) -> exp(x)*cos(y),
      (x,y) -> x^2,
      (x,y) ->  x^3*(1-x)^3*(1-y)^3,
      (x,y) -> x^3*(1-x)^3*(1-y)^3 * cos(100x*y))

N = 1000; 
@time BM = sparse(B[Block.(1:N), Block.(1:N)]);
@time ΔM = sparse(Δ[Block.(1:N), Block.(1:N)]);
@time M = [BM; ΔM];
M = [sparse(I, size(M,1), 2) M] # add tau term
@time F = factorize(M);

# RHS
k = 1
∂u = Fun(fs[k], rangespace(B))
r = pad(coefficients(∂u, rangespace(B)), size(Fs[N],1))
@time u_cfs = F \ r
    

#######
# Example 5 Transport equation
#######

S = DirichletTriangle{0,1,0}()
∂S = Legendre(Segment(Vec(0.0,0), Vec(1.0,0)))
B = I : S → ∂S

Dx = Derivative([1,0]) : S
Dt = Derivative([0,1]) : S


x,_ = Fun(∂S)
u₀ = x*(1-x)*exp(x)

# u_t = u_x
L = [B; Dx - Dt]
N = 20; M = sparse(L[Block.(1:N), Block.(1:N)]);
u_c = M \ pad(coefficients([u₀; 0], rangespace(L)), size(M,1))
u = Fun(domainspace(L), u_c)

# 0.5u_t = u_x
S = DirichletTriangle{0,1,1}()
∂S = Legendre(Segment(Vec(0.0,0), Vec(1.0,0))) , Legendre(Segment(Vec(0.0,1.0),Vec(1.0,0)))
B_x = I : S → ∂S[1]
B_z = I : S → ∂S[2]
Dx = Derivative([1,0]) : S
Dt = Derivative([0,1]) : S

L = [B_x; B_z; Dx - 0.5Dt]
N = 100; M = sparse(L[Block.(1:N), Block.(1:N)])
u₀_x = Fun((x,_) -> x*exp(x-1), rangespace(B_x))
u₀_z = Fun((x,y) -> (1-y), rangespace(B_z))
u_c = M \ pad(coefficients([u₀_x; u₀_z; 0], rangespace(L)), size(M,1))
u = Fun(domainspace(L), u_c)

# u_t + u_x = 0
S = DirichletTriangle{1,1,0}()
∂S = Legendre(Segment(Vec(0.0,0), Vec(1.0,0))) , Legendre(Segment(Vec(0.0,0.0),Vec(0.0,1.0)))
B_x = I : S → ∂S[1]
B_y = I : S → ∂S[2]
Dx = Derivative([1,0]) : S
Dt = Derivative([0,1]) : S

L = [B_x; B_y; Dt + Dx]
N = 100; M = sparse(L[Block.(1:N), Block.(1:N)])
u₀_x = Fun((x,_) -> (1-x)*exp(x), rangespace(B_x))
u₀_y = Fun((x,y) -> (1-y), rangespace(B_y))

u_c = M \ pad(coefficients([u₀_x; u₀_y; 0], rangespace(L)), size(M,1))
u = Fun(domainspace(L), u_c)


#######
# Example 7 Helmholtz in a polygon
#######
d = Triangle() ∪ Triangle(Vec(1,1),Vec(1,0),Vec(0,1)) ∪
                 Triangle(Vec(1,1),Vec(0,1),Vec(0,2)) ∪
                 Triangle(Vec(0,0),Vec(0,1),Vec(-1,1.5))


f = Fun((x,y) -> cos(x*sin(y)), d)

∂d = PiecewiseSegment([Vec(0.,0), Vec(1.,0), Vec(1,1), Vec(0,2), Vec(0,1), Vec(-1,1.5), Vec(0,0)])

S = DirichletTriangle{1,1,1}.(components(d))
∂S = Legendre.(components(∂d))

∂S12 = Legendre(Segment(Vec(1.,0),Vec(0,1)))
∂S23 = Legendre(Segment(Vec(1.,1),Vec(0,1)))
∂S14 = Legendre(Segment(Vec(0.,0),Vec(0,1)))

Dx = Derivative([1,0])
Dy = Derivative([0,1])

k = 20

L = [(I : S[1] → ∂S[1]) 0                 0                     0                  0                0                       0                    0                  0                   0                    0                  0;
     0                  0                 0                    (I : S[2] → ∂S[2])  0                0                       0                    0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       (I : S[3] → ∂S[3])   0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       (I : S[3] → ∂S[4])   0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       0                    0                  0                   (I : S[4] → ∂S[5])   0                  0;
     0                  0                 0                     0                  0                0                       0                    0                  0                   (I : S[4] → ∂S[6])   0                  0;
     (I : S[1] → ∂S12)  0                 0                   -(I : S[2] → ∂S12)   0                0                       0                    0                  0                    0                   0                  0;
     0                  (I : S[1] → ∂S12) (I : S[1] → ∂S12)     0                -(I : S[2] → ∂S12) -(I : S[2] → ∂S12)      0                    0                  0                    0                   0                  0;
     0                  0                 0                    (I : S[2] → ∂S23)   0                0                      -(I : S[3] → ∂S23)    0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                (I : S[2] → ∂S23)       0                    0                  -(I : S[3] → ∂S23)   0                   0                  0;
     (I : S[1] → ∂S14)  0                 0                     0                  0                0                       0                    0                  0                   -(I : S[4] → ∂S14)   0                  0;
     0                  (I : S[1] → ∂S14) 0                     0                  0                0                       0                    0                  0                    0                  -(I : S[4] → ∂S14)   0;
     Dx                -I                 0                     0                  0                0                       0                    0                  0                    0                   0                  0;
     Dy                 0                -I                     0                  0                0                       0                    0                  0                    0                   0                  0;
     0                  0                 0                     Dx                -I                0                       0                    0                  0                    0                   0                  0;
     0                  0                 0                     Dy                 0               -I                       0                    0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       (Dx : S[3])         -(I : S[3])         0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       Dy                   0                 -I                    0                   0                  0;
     0                  0                 0                     0                  0                0                       0                    0                  0                    Dx                 -I                  0;
     0                  0                 0                     0                  0                0                       0                    0                  0                    (Dy : S[4])         0                 -(I : S[4])
    k^2*I              Dx                Dy                    0                  0                0                       0                    0                  0                    0                   0                  0;
     0                  0                 0                     k^2*I                  Dx               Dy                      0                    0                  0                    0                   0                  0;
     0                  0                 0                     0                  0                0                       k^2*I                    Dx                 Dy                   0                   0                  0;
     0                  0                 0                     0                  0                0                       0                    0                  0                    k^2*I                  Dx                 Dy
     ]


f = ones(∂d)

N = 101
@time M = sparse(L[Block.(1:N), Block.(1:N)]);
@time F = qr(M);
@time u_c = F \ pad(coefficients(vcat(components(f)..., Zeros(18)...), rangespace(L)), size(M,1));
u = Fun(domainspace(L), u_c)
