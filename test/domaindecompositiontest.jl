
d = Triangle() , Triangle(Vec(1,1),Vec(0,1),Vec(1,0))
S = DirichletTriangle{1,1,1}.(d)
B₁₁ = I : S[1] → Legendre(Vec(0,0) .. Vec(0,1))
B₁₂ = I : S[1] → Legendre(Vec(0,0) .. Vec(1,0))
B₂₁ = I : S[2] → Legendre(Vec(1,1) .. Vec(0,1))
B₂₂ = I : S[2] → Legendre(Vec(1,1) .. Vec(1,0))
