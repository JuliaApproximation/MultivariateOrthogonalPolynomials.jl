using FastTransforms
import Base: *


#=
N = 2.^(7:10)

for n in N
    A = sphones(Float64, n, n)
    B = deepcopy(A)

    @time PC = c_plan_sph2fourier(n)
    @time c_sph2fourier(PC, B)

    @time PJ = SlowSphericalHarmonicPlan(A)
    @time A = PJ*A;

    @show vecnorm(A-B)/vecnorm(A)

    println()
    println()
end
=#

#=
α, β, γ = 0.0, -0.5, -0.5
#α, β, γ = -0.5, -0.5, -0.5
#α, β, γ = 0.0, 0.0, 0.0

for n in N
    A = trirand(Float64, n, n)
    B = deepcopy(A)

    @time PC = c_plan_tri2cheb(n, α, β, γ)
    @time c_tri2cheb(PC, B)

    @time PJ = SlowTriangularHarmonicPlan(A, α, β, γ);
    @time A = PJ*A;

    @show vecnorm(A-B)/vecnorm(B)

    println()
    println()
end

CF = zero(F); CF[1,2] = 1.0; CF[2,2] = 1.0; CF[3,2] = 1.0;
c_tri2cheb(cp, CF)
CF
c_cheb2tri(cp, CF)
CF

cprot = c_plan_rottriangle(size(F, 1), α, β, γ)

CF = zero(F); CF[1,2] = 1.0; CF[2,2] = 1.0; CF[3,2] = 1.0;
c_execute_tri_hi2lo(cprot, CF)
CF
c_execute_tri_lo2hi(cprot, CF)
CF
=#

using ApproxFun, FastTransforms

import ApproxFun: jacobip

jacobinorm(n,a,b) = if n ≠ 0
    sqrt((2n+a+b+1))*exp((lgamma(n+a+b+1)+lgamma(n+1)-log(2)*(a+b+1)-lgamma(n+a+1)-lgamma(n+b+1))/2)
else
    sqrt(exp(lgamma(a+b+2)-log(2)*(a+b+1)-lgamma(a+1)-lgamma(b+1)))
end

njacobip(n,a,b,x) = jacobinorm(n,a,b) * jacobip(n,a,b,x)

α = 0.0#0.25
β = 0.0#-0.25
γ = 0.0#-0.125

P = (ℓ,m,x,y) -> (2*(1-x))^m*njacobip(ℓ-m,2m+β+γ+1,α,2x-1)*njacobip(m,γ,β,2y/(1-x)-1)

p_T = chebyshevpoints(10)

f = (x,y) -> P(0,0,x,y) + P(1,1,x,y) + P(2,2,x,y) + P(3,3,x,y) + P(4,4,x,y) +
             P(1,0,x,y) + P(2,1,x,y) + P(3,2,x,y) +
             P(2,0,x,y) + P(3,1,x,y) +
             P(3,0,x,y)

f̃ = (s,t) -> f((s+1)/2, (t+1)/2*(1-(s+1)/2))

F = f̃.(p_T, p_T')

for j = 1:size(F,2)
    F[:,j] = chebyshevtransform(F[:,j])
end

for k = 1:size(F,1)
    F[k,:] = chebyshevtransform(F[k,:])
end

P = Tri2Cheb(

cp = c_plan_tri2cheb(size(F, 1), α, β, γ)

CF̌ = copy(F)

c_cheb2tri(cp, CF̌)

CF̌
