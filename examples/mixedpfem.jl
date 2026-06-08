using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, BlockArrays, StaticArrays

#####
# Poisson
#####

#  -Δu = f
#   𝐮 := ∇u
# Strong form:
#     <𝐯, 𝐮> - <𝐯, ∇u> = 0
#    <v, ∇⋅𝐮> = -<v,f>
# Weak form: for 𝐮,𝐯 ∈ H_div, u,v ∈ L_2, find 𝐮,u such that:
#    <𝐯, 𝐮> + <∇⋅𝐯,u> = 0
#    <v, ∇⋅𝐮>         = -<v,f>

# A natural basis for H_div on a square is,
# for integrated Legendre Cₙ := Cₙ^(-1/2)
# and Legendre $Pₙ$
#   {[Cₖ(x)Pⱼ(y),0]},  {[0,Pₖ(x)Cⱼ(y)}}

C = Ultraspherical(-1/2)
P = Legendre()
P² = KronPolynomial(P,P)
CP = KronPolynomial(C,P)
PC = KronPolynomial(P,C)


N = 20
M₁ = sparse((CP'CP)[Block.(1:N), Block.(1:N)])
M₂ = sparse((PC'PC)[Block.(1:N), Block.(1:N)])
D₁ = sparse((P²'diff(CP,(1,0)))[Block.(1:N-1), Block.(1:N)])
D₂ = sparse((P²'diff(PC,(0,1)))[Block.(1:N-1), Block.(1:N)])

Z₁ = Zeros(axes(M₁))
Z₂ = Zeros((axes(D₁,1),axes(D₁,1)))


A = [M₁ Z₁ D₁';
     Z₁ M₂ D₂';
     D₁ D₂ Z₂]



f = expand(P², splat((x,y) -> -(1+x)^2*cos(1-y)^2*cos((1+x)sin(1-y))+sin(1-y)*(-cos((1+x)sin(1-y))sin(1-y)+(1+x)*sin((1+x)sin(1-y)))))
𝐳 = Zeros(axes(M₁,1))
𝐜 = A \ [𝐳; 𝐳; -(P²'f)[Block.(1:N-1)]]

n₁ = size(M₁,1)
n₂ = size(D₁,1)
c₁,c₂,c = blocks(BlockArray(𝐜, [n₁,n₁,n₂]))


∇u₁ = CP[:,Block.(1:N)]c₁
∇u₂ = PC[:,Block.(1:N)]c₂
u = P²[:,Block.(1:N-1)]c


@test c₁'M₁*c₁ ≈ ∇u₁'∇u₁
@test c₂'M₂*c₂ ≈ ∇u₂'∇u₂
@test c₁'D₁'c ≈ diff(∇u₁,(1,0))'u
@test c'D₁*c₁ ≈ u'diff(∇u₁,(1,0))
@test c'D₂*c₂ ≈ u'diff(∇u₂,(0,1))


𝐱 = x,y = SVector(0.1,0.2)

@test diff(u,(1,0))[𝐱] ≈ ∇u₁[𝐱]
@test diff(u,(0,1))[𝐱] ≈ ∇u₂[𝐱]

@test u[𝐱] ≈ cos((x + 1)sin(y - 1))



# exact solution
N = 30
u = expand(P², splat((x,y) -> cos((x + 1)sin(y - 1))))
∇u₁ = CP/CP\diff(u,(1,0))
∇u₂ = CP/CP\diff(u,(0,1))

@test ∇u₁[𝐱] ≈ -sin(y - 1)*sin((x + 1)sin(y - 1))
@test ∇u₂[𝐱] ≈ -(x+1)*cos(y - 1)*sin((x + 1)sin(y - 1))
@test (diff(∇u₁,(1,0)) + diff(∇u₂,(0,1)))[𝐱] ≈ ((Q/Q\diff(∇u₁,(1,0))) + Q/Q\diff(∇u₂,(0,1)))[𝐱] ≈ f[𝐱]

fs = diff(∇u₁,(1,0)) , diff(∇u₂,(0,1))
Q\fs[1]


fs[1]

using LazyBandedMatrices
P = KronPolynomial(Ultraspherical(1/2), Ultraspherical(1/2))
Q = KronPolynomial(Ultraspherical(1/2), Ultraspherical(3/2))
R = Q \ P

Q \ (P*[1:10; zeros(∞)]) ≈ (Q \ P)*[1:10; zeros(∞)]




Q = (diff(∇u₁,(1,0)) + diff(∇u₂,(0,1))).args[1]


@test diff(∇u₁,(1,0))[𝐱] ≈ (Q/Q\diff(∇u₁,(1,0)))[𝐱] ≈  -sin(y - 1)^2*cos((x + 1)sin(y - 1))
@test diff(∇u₂,(0,1))[𝐱] ≈ (Q/Q\diff(∇u₂,(0,1)))[𝐱] ≈  (x+1)*sin(y - 1)*sin((x + 1)sin(y - 1))-(x+1)^2*cos(y - 1)^2*cos((x + 1)sin(y - 1))

(diff(∇u₁,(1,0)) + diff(∇u₂,(0,1)))


c = (P²\u)[Block.(1:N-1)]
c₁ = (CP\∇u₁)[Block.(1:N)]
c₂ = (PC\∇u₂)[Block.(1:N)]

@test ∇u₁[𝐱] ≈ CP[𝐱,Block.(1:N)]'*c₁

@test c₁'M₁*c₁ ≈ ∇u₁'∇u₁
@test c₂'M₂*c₂ ≈ ∇u₂'∇u₂
@test c₁'D₁'c ≈ diff(∇u₁,(1,0))'u
@test c'D₁*c₁ ≈ u'diff(∇u₁,(1,0))
@test c'D₂*c₂ ≈ u'diff(∇u₂,(0,1))
