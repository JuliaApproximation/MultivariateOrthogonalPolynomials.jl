using Revise
using  ApproxFun, MultivariateOrthogonalPolynomials, InfiniteArrays, FillArrays, BlockArrays
import MultivariateOrthogonalPolynomials: checkerboard, icheckerboard, CDisk2CxfPlan, DiskTensorizer, _coefficients, ipolar
import ApproxFunBase: totensor, blocklengths, fromtensor, tensorizer, plan_transform!, plan_transform
import ApproxFunOrthogonalPolynomials: jacobip


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

@testset "Disk" begin
    @testset "tensorizer derivation" begin
        A = [1 2  3  4  5 7 8 11 12;
             6 9 10 13 14 0 0  0  0;
             15 0 0  0  0 0 0  0  0] 
        B = PseudoBlockArray(A, Ones{Int}(3), [1; Fill(2,4)])

        a = Vector{eltype(A)}()
        for N = 1:nblocks(B,2), K=1:(N+1)÷2
            append!(a, vec(B[Block(K,N-2K+2)]))
        end

        @test a == fromtensor(DiskTensorizer(),A) == fromtensor(ZernikeDisk(),A) == 1:15

        for N = 1:10
            M = 4*(N-1)+1; 
            n = length(fromtensor(DiskTensorizer(),Array{Float64}(undef,N,M)))
            @test round(Int,1/4*(1 + sqrt(1 + 8n)),RoundUp) == N
        end

        n = length(a)
        N = round(Int,1/4*(1 + sqrt(1 + 8n)),RoundUp)
        M = 4*(N-1)+1
        Ã = zeros(eltype(a), N, M)
        B = PseudoBlockArray(Ã, Ones{Int}(3), [1; Fill(2,4)])
        k = 1
        for N = 1:nblocks(B,2), K=1:(N+1)÷2
            V = view(B, Block(K,N-2K+2))
            for j = 1:length(V)
                V[j] = a[k]
                k += 1
            end
        end
        @test Ã == totensor(DiskTensorizer(),a) == totensor(ZernikeDisk(),a) == A
    end

    @testset "ChebyshevDisk" begin
        A = [1 2 3 4 5; 6 7 8 9 10]
        @test_broken totensor(ChebyshevDisk(),fromtensor(ChebyshevDisk(), A))[1:2,:] ≈ A

        f = (x,y) -> x*y+cos(y-0.1)+sin(x)+1; 
        ff = Fun(f, ChebyshevDisk(), 1000)
        @test ff(0.1,0.2) ≈ f(0.1,0.2)

        ff = Fun(f, ChebyshevDisk())
        @test ff(0.1,0.2) ≈ f(0.1,0.2)


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
    end


    @testset "disk transform" begin
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
        @test d ≈  [0.0  0.5  0.0 0.0 0.0 ; 0.0  0.0  0.0 0.0 0.0]

        f = Fun((x,y) -> x, ChebyshevDisk()); n = 2;
        c = totensor(f.space, f.coefficients)
        c = pad(c, n, 4n-3)
        P = CDisk2CxfPlan(n)
        d = P \ c
        @test d ≈  [0.0  0.0  0.5 0.0 0.0; 0.0  0.0  0.0 0.0 0.0]


        @testset "explicit polynomial" begin
            p1 = (r,θ) -> sqrt(10)*r^2*(4r^2-3)*cos(2θ)/sqrt(π)
            f = Fun((x,y) -> p1(sqrt(x^2+y^2), atan(y,x)), ChebyshevDisk());
            @test f(0.1*cos(0.2),0.1*sin(0.2)) ≈ p1(0.1,0.2)
            c = totensor(f.space, chop(f.coefficients,1E-12))
            @test chebydiskeval(c, 0.1, 0.2) ≈ p1(0.1,0.2)
            n = size(c,1); c = sqrt(π)*pad(c,n,4n-3)
            P = CDisk2CxfPlan(n)
            d = P \ c
            @test d[1,4] ≈ 0 atol=1E-13
            @test d[2,5] ≈ 1
        end
        @testset "different m and ℓ" begin
            for (m,ℓ) in ((0,0), (0,2), (1,1), (2,2), (2,6), (3,3))
                p1 = (r,θ) -> sqrt(2ℓ+2)*r^m*jacobip((ℓ-m)÷2, 0, m, 2r^2-1)*cos(m*θ)/sqrt(π)
                f = Fun((x,y) -> p1(sqrt(x^2+y^2), atan(y,x)), ChebyshevDisk());
                @test f(0.1*cos(0.2),0.1*sin(0.2)) ≈ p1(0.1,0.2)
                c = totensor(f.space, chop(f.coefficients,1E-12))
                @test chebydiskeval(c, 0.1, 0.2) ≈ p1(0.1,0.2)

                n = size(c,1); c = sqrt(π)*pad(c,n,4n-3)
                P = CDisk2CxfPlan(n)
                d = P \ c
                @test norm(d[:, 1:2m]) ≈ 0 atol = 1E-13
                @test norm(d[1:(ℓ-m)÷2, 2m+1]) ≈ 0 atol = 1E-13
                @test d[(ℓ-m)÷2+1, 2m+1] ≈ 1
                @test norm(d[(ℓ-m)÷2+2:end, 2m+1]) ≈ 0 atol = 1E-13
                @test norm(d[:, 2m+2:end]) ≈ 0 atol = 1E-13
            end
        end
    end

    @testset "transform derivation" begin
        p = points(ZernikeDisk(),10)
        @test length(p) == 6
        v = fill(1.0,length(p))
        n = length(v)
        N = (1 + isqrt(1+8n)) ÷ 4
        M = 2N-1
        D = plan_transform!(rectspace(ChebyshevDisk()), reshape(v,N,M))
        N,M = D.plan[1][2],D.plan[2][2]
        V=reshape(v,N,M)
        @test D*V ≈ [[1 zeros(1,2)]; zeros(1,3)]
        C = checkerboard(V)
        disk2cxf = CDisk2CxfPlan(size(C,1))
        C = disk2cxf\C
        @test C ≈ [[1/sqrt(2) zeros(1,4)]; zeros(1,5)]
        @test fromtensor(ZernikeDisk(), C) ≈ [1/sqrt(2); Zeros(5)]

        v = fill(1.0,length(p))
        @test plan_transform(ZernikeDisk(),v)*v ≈ [1/sqrt(2); Zeros(5)]
        @test v == fill(1.0,length(p))
    end

    @testset "ChebyshevDisk-ZernikeDisk" begin
        @test coefficients([sqrt(2)], ChebyshevDisk(), ZernikeDisk()) ≈ [1.0]
        @test coefficients([1.0], ZernikeDisk(), ChebyshevDisk()) ≈ [sqrt(2)]
        v = coefficients(coefficients(Float64.(1:5), ChebyshevDisk(), ZernikeDisk()), ZernikeDisk(), ChebyshevDisk())
        @test v ≈ [1:5;zeros(length(v)-5)]
        v = coefficients(coefficients(Float64.(1:10), ZernikeDisk(), ChebyshevDisk()), ChebyshevDisk(), ZernikeDisk())
        @test v ≈ [1:10;zeros(length(v)-10)]
        v = coefficients(coefficients(Float64.(1:5), ZernikeDisk(), ChebyshevDisk()), ChebyshevDisk(), ZernikeDisk())
        @test v ≈ [1:5;zeros(length(v)-5)]
    end

    @testset "transform" begin
        p= points(ZernikeDisk(), 6)
        ff = x->x[1]
        v = ff.(p)
        @test transform(ZernikeDisk(),v) ≈ [zeros(2); 0.5;zeros(3)]

        p= points(ZernikeDisk(), 20)
        f = Fun((x,y) -> x^2, ZernikeDisk(), 20)
        Fun((x,y) -> x^2, ChebyshevDisk())
        ff = x -> x[1]^2
        v = ff.(p)
        P = plan_transform(ZernikeDisk(),v)
        N,M = P.cxfplan.plan[1][2],P.cxfplan.plan[2][2] 
        V = P.cxfplan*reshape(copy(v),N,M)
        C = checkerboard(V)
        @test totensor(ChebyshevDisk(), Fun(ff, ChebyshevDisk()).coefficients)[1:2,1:5] ≈ C
    end

    @testset "ZernikeDisk evaluate" begin
        f = Fun((x,y) -> x^2, ZernikeDisk())
        C = totensor(ZernikeDisk(), f.coefficients)
        F = _coefficients(CDisk2CxfPlan(size(C,1)), C, ZernikeDisk(), ChebyshevDisk())
        @test ProductFun(icheckerboard(F), rectspace(ChebyshevDisk()))(ipolar(0.1,0.2)...) ≈ 0.1^2
    end

    @testset "ZernikeDisk" begin
        f = Fun((x,y) -> 1, ZernikeDisk(), 10)
        @test f(0.1,0.2) ≈ 1.0
        f = Fun((x,y) -> 1, ZernikeDisk())
        @test f(0.1,0.2) ≈ 1.0
        f = Fun((x,y) -> x, ZernikeDisk(),10)
        @test f(0.1,0.2) ≈ 0.1
        f = Fun((x,y) -> y, ZernikeDisk(),10)
        @test f(0.1,0.2) ≈ 0.2
        f = Fun((x,y) -> x^2, ZernikeDisk())
        @test f(0.1,0.2) ≈ 0.1^2
        f = Fun((x,y) -> x*y, ZernikeDisk())
        @test f(0.1,0.2) ≈ 0.1*0.2
        f = Fun((x,y) -> y^2, ZernikeDisk())
        @test f(0.1,0.2) ≈ 0.2^2
        f = Fun((x,y) -> x^2*y^3, ZernikeDisk(),100)
        @test f(0.1,0.2) ≈ 0.1^2*0.2^3
        ff = (x,y) -> x^2*y^3
        @test Fun(ff, ChebyshevDisk())(0.1,0.2) ≈ 0.1^2*0.2^3
        f = Fun((x,y) -> exp(x*cos(y-0.1)), ZernikeDisk(), 10000)
        @test f(0.1,0.2) ≈ exp(0.1*cos(0.1))

        f = Fun((x,y) -> exp(x*cos(y-0.1)), ZernikeDisk())
        @test f(0.1,0.2) ≈ exp(0.1*cos(0.1))

        @test blocklengths(ZernikeDisk()) == Base.OneTo(∞)
    end
end