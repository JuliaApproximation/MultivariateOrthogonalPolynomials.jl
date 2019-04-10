using Revise, ApproxFun, MultivariateOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: checkerboard, icheckerboard, CDisk2CxfPlan
import ApproxFunBase: totensor
import ApproxFunOrthogonalPolynomials: jacobip, chebyshevt





f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; 
ff = Fun(f, ChebyshevDisk(), 1000)
@test ff(0.1,0.2) ≈ f(0.1,0.2)

ff = Fun(f, ChebyshevDisk())
@test ff(0.1,0.2) ≈ f(0.1,0.2)

function chebydiskeval(c::AbstractMatrix{T}, r, θ) where T
    ret = zero(T)
    for j = 1:2:size(c,2), k=1:size(c,1)
        m,ℓ = j ÷ 2, k-1
        ret += c[k,j] * cos(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    for j = 2:2:size(c,2), k=1:size(c,1)
        m = j ÷ 2; ℓ = k-1
        ret += c[k,j] * sin(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    end
    ret
end


for (m,ℓ) in ((0,1),(0,2),(0,3), (1,0), (1,1), (1,2), (2,0), (2,1),(3,0),(3,4))
    p1 = (r,θ) -> cos(m*θ) * cos((2ℓ+isodd(m))*acos(r))
    f = Fun((x,y) -> p1(sqrt(x^2+y^2), atan(y,x)), ChebyshevDisk())
    c = totensor(f.space, chop(f.coefficients,1E-12))    
    @test c[ℓ+1, 2m+1] ≈ 1
    @test f(0.1cos(0.2),0.1sin(0.2)) ≈ chebydiskeval(c, 0.1, 0.2)

    if m > 0
        p2 = (r,θ) -> sin(m*θ) * cos((2ℓ+isodd(m))*acos(r))
        f = Fun((x,y) -> p2(sqrt(x^2+y^2), atan(y,x)), ChebyshevDisk())
        c = totensor(f.space, chop(f.coefficients,1E-12))    
        @test c[ℓ+1, 2m] ≈ 1
        @test f(0.1cos(0.2),0.1sin(0.2)) ≈ chebydiskeval(c, 0.1, 0.2)
    end
end

f = Fun((x,y) -> 1, ChebyshevDisk()); n = 1;
    c = totensor(f.space, f.coefficients)
    P = CDisk2CxfPlan(n)
    d = P \ c
    @test d ≈ [1/sqrt(2)]
f = Fun((x,y) -> y, ChebyshevDisk()); n = 2;
    c = totensor(f.space, f.coefficients)
    c = pad(c, n, 4n-3)
    P = CDisk2CxfPlan(n)
    d = P \ c
    @test d ≈  [0.0  0.5  0.0  0.0  0.0; 0.0  0.0  0.0  0.0  0.0]

f = Fun((x,y) -> x, ChebyshevDisk()); n = 2;
    c = totensor(f.space, f.coefficients)
    c = pad(c, n, 4n-3)
    P = CDisk2CxfPlan(n)
    d = P \ c
    @test d ≈  [0.0  0.0  0.5  0.0  0.0; 0.0  0.0  0.0  0.0  0.0]


# m,ℓ = (0,2);
# m,ℓ = (1,1);
# m,ℓ = (2,2);


m,ℓ = (2,4);
p1 = (r,θ) -> sqrt(2ℓ+2)*r^m*jacobip((ℓ-m)÷2, m, 0, 2r^2-1)*cos(m*θ)/sqrt(2π)
f = Fun((x,y) -> p1(sqrt(x^2+y^2), atan(y,x)), ChebyshevDisk());
    @test f(0.1*cos(0.2),0.1*sin(0.2)) ≈ p1(0.1,0.2)
    c = totensor(f.space, chop(f.coefficients,1E-12))
    @test chebydiskeval(c, 0.1, 0.2) ≈ p1(0.1,0.2)
    

    n = size(c,1); c = sqrt(2π)*pad(c,n,4n-3)
    P = CDisk2CxfPlan(n)
    d = P \ c
    @test d[(ℓ-m)÷2, 2m+1] ≈ 0
    @test d[(ℓ-m)÷2+1, 2m+1] ≈ 1

    



    



