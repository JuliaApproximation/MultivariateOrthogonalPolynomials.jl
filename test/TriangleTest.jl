using StaticArrays, Plots, BandedMatrices, FastTransforms,
        ApproxFun, MultivariateOrthogonalPolynomials, Test
    import MultivariateOrthogonalPolynomials: Lowering, ProductTriangle, DuffyTriangle,
                            clenshaw, block, TriangleWeight,plan_evaluate, weight
    import ApproxFun: testbandedblockbandedoperator, Block, BandedBlockBandedMatrix, blockcolrange, blocksize,
                    Vec, jacobip, plan_transform


@testset "Triangle domain" begin
    d = Triangle()
    @test fromcanonical(d, 1,0) == fromcanonical(d, Vec(1,0)) == tocanonical(d, Vec(1,0)) == Vec(1,0)
    @test fromcanonical(d, 0,1) == fromcanonical(d, Vec(0,1)) == tocanonical(d, Vec(0,1)) == Vec(0,1)
    @test fromcanonical(d, 0,0) == fromcanonical(d, Vec(0,0)) == tocanonical(d, Vec(0,0)) == Vec(0,0)

    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    @test fromcanonical(d,0,0) == d.a == Vec(2,3)
    @test fromcanonical(d,1,0) == d.b == Vec(3,4)
    @test fromcanonical(d,0,1) == d.c == Vec(1,6)
    @test tocanonical(d,d.a) == Vec(0,0)
    @test tocanonical(d,d.b) == Vec(1,0)
    @test tocanonical(d,d.c) == Vec(0,1)
end

let P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)
    function cfseval(a,b,c,r,x,y)
        j = 1; ret = 0.0
        for n=0:length(r),k=0:n
            ret += r[j]*P(n,k,a,b,c,x,y)
            j += 1
            j > length(r) && break
        end
        ret
    end
    @testset "Degenerate conversion" begin
        C = Conversion(KoornwinderTriangle(0.0,-0.5,-0.5),KoornwinderTriangle(0.0,0.5,-0.5))
        testbandedblockbandedoperator(C)
        r = randn(10)
        x,y = 0.1, 0.2
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,0.5,-0.5,C[1:10,1:10]*r,x,y)

        C = Conversion(KoornwinderTriangle(0.0,-0.5,-0.5),KoornwinderTriangle(0.0,-0.5,0.5))
        testbandedblockbandedoperator(C)
        r = randn(10)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,[C[k,j] for k=1:10,j=1:10]*r,x,y)
        @test cfseval(0.0,-0.5,-0.5,r,x,y) ≈ cfseval(0.0,-0.5,0.5,C[1:10,1:10]*r,x,y)
    end
end

N = 2000
    @time f = Fun((x,y) -> cos(1000x*y), Triangle(), sum(1:N));
    D_x = Derivative(space(f), [1,0])
    @time M_x = D_x[Block.(1:N), Block.(1:N)]
    @time (M_x*f.coefficients)

D_x

@time Fun(rangespace(D_x), (M_x*f.coefficients))(0.1,0.2)
(M_x*f.coefficients)
Fun(rangespace(D_x), (M_x*f.coefficients))(0.1,0.2) - (-1000*0.2*sin(1000*0.1*0.2))

struct FastLap
    D_x::BandedBlockBandedMatrix{Float64}
    D̃_x::BandedBlockBandedMatrix{Float64}
    R_x::BandedBlockBandedMatrix{Float64}
    R̃_x::BandedBlockBandedMatrix{Float64}

    D_y::BandedBlockBandedMatrix{Float64}
    D̃_y::BandedBlockBandedMatrix{Float64}
    R_y::BandedBlockBandedMatrix{Float64}
    R̃_y::BandedBlockBandedMatrix{Float64}
end


function FastLap(N)
    S = KoornwinderTriangle(0.,0.,0.)
    D_x = Derivative(S, [1,0])
    M_x = D_x[Block.(1:N), Block.(1:N)]
    D̃_x = Derivative(rangespace(D_x), [1,0])
    M̃_x = D̃_x[Block.(1:N), Block.(1:N)]
    C_x = Conversion(rangespace(D̃_x) , KoornwinderTriangle(2.0, 1.0, 2.0))
    R_x = C_x[Block.(1:N), Block.(1:N)]
    C̃_x = Conversion( rangespace(C_x) , KoornwinderTriangle(2.0, 2.0, 2.0))
    R̃_x = C̃_x[Block.(1:N), Block.(1:N)]

    D_y = Derivative(S, [0,1])
    M_y = D_y[Block.(1:N), Block.(1:N)]
    D̃_y = Derivative(rangespace(D_y), [0,1])
    M̃_y = D̃_y[Block.(1:N), Block.(1:N)]
    C_y = Conversion(rangespace(D̃_y) , KoornwinderTriangle(1.0, 2.0, 2.0))
    R_y = C_y[Block.(1:N), Block.(1:N)]
    C̃_y = Conversion( rangespace(C_y) , KoornwinderTriangle(2.0, 2.0, 2.0))
    R̃_y = C̃_y[Block.(1:N), Block.(1:N)]

    FastLap(M_x, M̃_x, R_x, R̃_x, M_y, M̃_y, R_y, R̃_y)
end

function Base.:*(L::FastLap, f)
    Fun(KoornwinderTriangle(2.,2.,2.), L.R̃_x*(L.R_x*(L.D̃_x*(L.D_x*f.coefficients))) + L.R̃_y*(L.R_y*(L.D̃_y*(L.D_y*f.coefficients))))
end


N = 50
    n = sum(1:N)
    ω = n/40
    @time f = Fun((x,y) -> cos(ω*x*y), Triangle(), n);
    plot(abs.(f.coefficients) .+ eps(); yscale=:log10)
@time L = FastLap(N)
@time L*f

x,y = 0.1,0.2
    (-ω^2*(x^2+y^2)*cos(ω*x*y) - (L*f)(0.1,0.2))/(L*f)(0.1,0.2)

err = Dict{Int,Float64}()
tim_tran = Dict{Int,Float64}()
tim_lapset = Dict{Int,Float64}()
tim_lap = Dict{Int,Float64}()


Ns = 100:100:2000
for N in Ns
    @show N
    n = sum(1:N)
    ω = N/10
    tim_tran[N] = @elapsed(f = Fun((x,y) -> cos(ω*x*y), Triangle(), n))
    tim_lapset[N] = @elapsed(L = FastLap(N))
    tim_lap[N] = @elapsed(v = L*f)
    err[N] = abs((-ω^2*(x^2+y^2)*cos(ω*x*y) - v(x,y))/(-ω^2*(x^2+y^2)*cos(ω*x*y)))
end


plot([sum(1:N) for N in Ns], [err[N] for N in Ns]; legend=false, title="Relative error of laplacian evaluated at (0.1,0.2)",
            linewidth=2.0, xscale=:log10, yscale=:log10)
        savefig("trianglelaplacianerror.pdf")

plot([sum(1:N) for N in Ns], [tim_tran[N] for N in Ns]; legend=:bottomright, title = "Time to calculate Laplacian coefficients",
            label="transform",
            linewidth=2.0, xscale=:log10, yscale=:log10)
    plot!([sum(1:N) for N in Ns], [tim_lapset[N] for N in Ns]; label="setup",
                linewidth=2.0, xscale=:log10, yscale=:log10)
    plot!([sum(1:N) for N in Ns], [tim_lap[N] for N in Ns]; label="apply",
                linewidth=2.0, xscale=:log10, yscale=:log10)
    savefig("trianglelaplaciantimes.pdf")


sum(1:2000)

v(0.1,0.2)
plot(abs.(v.coefficients).+eps(); yscale=:log10)



sum(1:2000)






f.coefficients


D_x = Derivative(space(f), [1,0])
    M_x = D_x[Block.(1:N), Block.(1:N)]
    D̃_x = Derivative(rangespace(D_x), [1,0])
    M̃_x = D̃_x[Block.(1:N), Block.(1:N)]
    C_x = Conversion(rangespace(D̃_x) , KoornwinderTriangle(2.0, 1.0, 2.0))
    R_x = C_x[Block.(1:N), Block.(1:N)]
    C̃_x = Conversion( rangespace(C_x) , KoornwinderTriangle(2.0, 2.0, 2.0))
    R̃_x = C̃_x[Block.(1:N), Block.(1:N)]
    @time R̃_x*(R_x*(M̃_x*(M_x*f.coefficients)))

typeof(M_x)

rangespace(D̃_x)


L = Laplacian(space(f))

@time L.op.ops[1][Block.(1:100), Block.(1:100)]


@time  points(Triangle(), 4_000_000);

@time p = points(Chebyshev()^2, 1_000_000);
S = Chebyshev()^2
N = 1_000_000
    T = Float64
    d = domain(S)
    @time pts = ApproxFun.squarepoints(T, N)
    @time for k=1:length(pts)
        pts[k] = fromcanonical(d,pts[k])
    end

@which fromcanonical(S,pts[1])

@time points(S, 1_000_000)


(1,2) .+ Vec(1,2)
d.domains
fromcanonical.(d.domains, Vec(0.1,0.2))

@time pts .= iduffy.(pts)
    # fromcanonical.(Ref(S.domain), )


P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)

        # f = Fun((x,y) -> P(0,0,0.,-0.5,-0.5,x,y), KoornwinderTriangle(0.,0.5,-0.5) )
    b = -0.5; c=-0.5
    ff = (xy) -> P(3,2,0.,b,c,xy...)
    S = KoornwinderTriangle(0.,b,c)
    @time f = Fun((x,y) -> cos(1000x*y), S)

@time Fun((x,y) -> cos(1000x*y), KoornwinderTriangle(0.,0.,0.))
ncoefficients(f)
using SO
P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)

        # f = Fun((x,y) -> P(0,0,0.,-0.5,-0.5,x,y), KoornwinderTriangle(0.,0.5,-0.5) )
    b = 0.0; c=0.0
    ff = (xy) -> P(0,0,0.,b,c,xy...) + P(1,0,0.,b,c,xy...) + P(1,1,0.,b,c,xy...) +
         P(2,0,0.,b,c,xy...) + P(2,1,0.,b,c,xy...) + P(2,2,0.,b,c,xy...)
    S = KoornwinderTriangle(0.,b,c)
    Fun(ff, S)
p = points(S ,20)
    v = ff.(p)
    P = plan_transform(S,v)
    v̂ = P.duffyplan*v
    F = tridevec_trans(v̂)
    F̌ = P.tri2cheb \ F
    # F̌ |> chopm



@which P.tri2cheb \ F

using SO
p = points(S ,20)
    v = ff.(p)
    P = plan_transform(S,v)
    v̂ = P.duffyplan*v
    F = tridevec_trans(v̂)
    F |> chopm

trivec(tridenormalize!(F̌))
using SO
@time f = Fun((x,y) -> cos(500x*y), KoornwinderTriangle(0.,0.5,-0.5) )
    f(0.1,0.2) , cos(500*0.1*0.2)

DuffyTriangle

ncoefficients(f)

cjt

f = Fun((x,y) -> cos(100x*y), KoornwinderTriangle(0.0,0.0,0.0))
        plot(f)

ncoefficients(f)


@time f = Fun((x,y) -> cos(100x*y), Triangle(),100)
    @time g = Laplacian()*f
    34
ncoefficients(f)

@which Derivative([1,0])*f

@which ApproxFun.mul_coefficients(view(L,1:ncoefficients(f), 1:ncoefficients(f)),
    f.coefficients)

using BlockBandedMatrices
V = view(L,1:ncoefficients(f), 1:ncoefficients(f))
    @which AbstractMatrix(V)


L = Derivative([1,0]) : space(f); N = 100
    @time L[Block.(1:N), Block.(1:N)]

ncoefficients(f)
plot(g)




@time f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,-0.5,-0.5)); # 1.15s
@time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,-0.5),40_000); # 0.2
@test f(0.1,0.2) ≈ cos(500*0.1*0.2)

@time f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,0.5,-0.5)); # 1.15s
@time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,0.5,-0.5),40_000); # 0.2
@test f(0.1,0.2) ≈ cos(500*0.1*0.2)

f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,0.5,0.5)); # 1.15s
f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,0.5,0.5),40_000); # 0.2
@test f(0.1,0.2) ≈ cos(500*0.1*0.2)

f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,-0.5,0.5)); # 1.15s
f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,0.5),40_000); # 0.2
@test f(0.1,0.2) ≈ cos(500*0.1*0.2)

C = Conversion(KoornwinderTriangle(0.,-0.5,-0.5),KoornwinderTriangle(0.,0.5,0.5))

C = Conversion(KoornwinderTriangle(0.0,-0.5,-0.5),KoornwinderTriangle(0.0,-0.5,0.5))

C[Block.(1:5),Block.(1:5)]


>>>>>>> 2764c54b20f46b8664b33cbd3008f44808ab104a
@testset "ProductTriangle constructors" begin
    S = ProductTriangle(1,1,1)
    @test fromcanonical(S, 0,0) == Vec(0,0)
    @test fromcanonical(S, 1,0) == fromcanonical(S, 1,1) == Vec(1,0)
    @test fromcanonical(S, 0,1) == Vec(0,1)
    pf = ProductFun((x,y)->exp(x*cos(y)), S, 40, 40)
    @test pf(0.1,0.2) ≈ exp(0.1*cos(0.2))

    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    S = ProductTriangle(1,1,1,d)
    @test fromcanonical(S, 0,0) == d.a
    @test fromcanonical(S, 1,0) == fromcanonical(S, 1,1) == d.b
    @test fromcanonical(S, 0,1) == d.c
    pf = ProductFun((x,y)->exp(x*cos(y)), S, 50, 50)
    @test pf(2,4) ≈ exp(2*cos(4))

    pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
    @test pyf(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
end

@testset "KoornwinderTriangle constructors" begin
    @time f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,-0.5,-0.5)); # 0.08s
    @time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,-0.5),40_000); # 0.2
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)
    @test values(f) ≈ f.(points(f))

    @time f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,0.5,-0.5)); # 0.08s
    @time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,0.5,-0.5),40_000); # 0.2
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,0.5,0.5)); # 1.15s
    f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,0.5,0.5),40_000); # 0.2
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    f = Fun((x,y)->cos(100x*y),KoornwinderTriangle(0.0,-0.5,0.5)); # 1.15s
    f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,0.5),40_000); # 0.2
    @test f(0.1,0.2) ≈ cos(500*0.1*0.2)

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    @time f = Fun((x,y)->cos(x*y),KoornwinderTriangle(0.0,-0.5,-0.5,d)); # 0.08s
    @test f(2,4) ≈ cos(2*4)
    @time f = Fun((x,y)->cos(x*y),KoornwinderTriangle(0.0,-0.5,0.5,d)); # 0.08s
    @test f(2,4) ≈ cos(2*4)
end

@time f = Fun((x,y)->cos(10x*y),KoornwinderTriangle(0.0,-0.5,-0.5)); # 0.08s
    @test f(0.1,0.2) ≈ cos(10*0.1*0.2)
    ncoefficients(f)
@time f = Fun((x,y) -> begin
        # ret = 0.0
        # for n = 18:18, k=8:8
        #     ret += P(n,k,0.,-0.5,-0.5,x,y)
        # end
        # ret
        P(18,8,0.,-0.5,-0.5,x,y)
    end,
    KoornwinderTriangle(0.0,-0.5,-0.5),600); # 0.08s
    sum(f.coefficients)

n = 600
    S = KoornwinderTriangle(0.0,-0.5,-0.5)
    ff = (xy) -> P(18,8,0.,-0.5,-0.5,xy...)
    p = points(S,n)
    v = ff.(p)
    PP = plan_transform(S,v)
    v̂ = PP.duffyplan*v
    F = tridevec_trans(v̂)
    F̌ = FastTransforms.cheb2tri(F,0.,-0.5,-0.5)
    F̌₂ = PP.tri2cheb \ F
    norm(F̌ - F̌₂)
size(F)
F̃ = copy(F)
c_cheb2tri(c_plan_tri2cheb(size(F,1),0.0,-0.5,-0.5),F̃)
F̃ |> chopm

@which PP.tri2cheb \ F
trivec(tridenormalize!(F̌,P.a,P.b,P.c))

sum(1:18)


using SO
tridevec(f.coefficients) |> chopm


sum(1:16)

@test f(0.1,0.2) ≈ cos(0.1*0.2)

p = points(space(f), 150)

ff = (xy) -> cos(xy[1]*xy[2])
ff.(p)




@time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,-0.5),40_000); # 0.2
@time f = Fun((x,y)->cos(500x*y),KoornwinderTriangle(0.0,-0.5,-0.5)); # 0.2

plot(f)
ncoefficients(f)

P = (n,k,a,b,c,x,y) -> x == 1.0 ? ((1-x))^k*jacobip(n-k,2k+b+c+1,a,1.0)*jacobip(k,c,b,-1.0) :
        ((1-x))^k*jacobip(n-k,2k+b+c+1,a,2x-1)*jacobip(k,c,b,2y/(1-x)-1)

f = Fun((x,y) -> P(5,1,0.,0.,0.,x,y), KoornwinderTriangle(0.,-0.5,-0.5) )
    F = tridevec(f.coefficients)
    F[:,2] = jjt(F[:,2], 2.0, 0.0, 3.0, 0.0) # correct
    jjt(F[2,:], -0.5, -0.5, 0.0, 0.0)
F
jjt(F[:,1], 0.0, 0.0, 1.0, 0.0) # correct





jjt(jjt(F[:,1], 0.0, -0.5, 0.0, 0.0),0.0,0.0,1.0,0.0)

jac2jac(Fun(Fun(Legendre(),[0,0,1.0]),Jacobi(0.0,0.5)).coefficients,
    0.0, 0.0, 0.5, 0.0)

jac2jac(Fun(x -> Fun(Jacobi(0.0,0.5),[0,0,1.0])(x),Legendre()).coefficients,
    0.0, 0.0, 0.5, 0.0)

jjt(Fun(x -> jacobip(2,0.5,0.0,x),Legendre()).coefficients,
    0.0, 0.0, 0.5, 0.0)

using SO
Fun(Fun(Jacobi(-0.5,0.0), [0,0,0,0,1.0]), Jacobi(0.0,0.0))


Fun(Fun(Jacobi(0.0,-0.5), [0,0,0,0,1.0]), Jacobi(0.0,0.0))


v = f.coefficients
    N = floor(Integer,sqrt(2length(v)) + 1/2)
    ret = zeros(Float64, N, N)
    j = 1
    for n=1:N,k=1:n
        j > length(v) && return
        ret[n-k+1,k] = v[j]
        j += 1
    end
    ret


v


@testset "old constructors" begin
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
    @test Fun(f,ProductTriangle(1,1,1))(0.1,0.2) ≈ exp(0.1*cos(0.2))
    f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,0,0))
    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))
    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1,d))
    @test f(2,4) ≈ exp(2*cos(4))


    K=KoornwinderTriangle(1,1,1)
    f=Fun(K,[1.])
    @test f(0.1,0.2) ≈ 1
    f=Fun(K,[0.,1.])
    @test f(0.1,0.2) ≈ -1.4
    f=Fun(K,[0.,0.,1.])
    @test f(0.1,0.2) ≈ -1
    f=Fun(K,[0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 1.18
    f=Fun(K,[0.,0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 1.2

    K=KoornwinderTriangle(2,1,1)
    f=Fun(K,[1.])
    @test f(0.1,0.2) ≈ 1
    f=Fun(K,[0.,1.])
    @test f(0.1,0.2) ≈ -2.3
    f=Fun(K,[0.,0.,1.])
    @test f(0.1,0.2) ≈ -1
    f=Fun(K,[0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 3.16
    f=Fun(K,[0.,0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 2.1

    K=KoornwinderTriangle(1,2,1)
    f=Fun(K,[1.])
    @test f(0.1,0.2) ≈ 1
    f=Fun(K,[0.,1.])
    @test f(0.1,0.2) ≈ -1.3
    f=Fun(K,[0.,0.,1.])
    @test f(0.1,0.2) ≈ -1.7
    f=Fun(K,[0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 0.96
    f=Fun(K,[0.,0.,0.,0.,1.])
    @test f(0.1,0.2) ≈ 1.87
end


@testset "Triangle Jacobi" begin
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
    Jx = MultivariateOrthogonalPolynomials.Lowering{1}(space(f))
    testbandedblockbandedoperator(Jx)
    @test Fun(Jx*f,ProductTriangle(0,1,1))(0.1,0.2) ≈ 0.1exp(0.1*cos(0.2))

    Jy = MultivariateOrthogonalPolynomials.Lowering{2}(space(f))
    testbandedblockbandedoperator(Jy)

    @test Jy[3,1] ≈ 1/3

    @test ApproxFun.colstop(Jy,1) == 3
    @test ApproxFun.colstop(Jy,2) == 5
    @test ApproxFun.colstop(Jy,3) == 6
    @test ApproxFun.colstop(Jy,4) == 8


    @test Fun(Jy*f,ProductTriangle(1,0,1))(0.1,0.2) ≈ 0.2exp(0.1*cos(0.2))
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

    Jz = MultivariateOrthogonalPolynomials.Lowering{3}(space(f))
    testbandedblockbandedoperator(Jz)
    @test Fun(Jz*f,ProductTriangle(1,1,0))(0.1,0.2) ≈ (1-0.1-0.2)exp(0.1*cos(0.2))

    @test f(0.1,0.2) ≈ exp(0.1*cos(0.2))

    Jx=Lowering{1}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jx)
    Jy=Lowering{2}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jy)
    Jz=Lowering{3}(KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(Jz)

    Jx=MultivariateOrthogonalPolynomials.Lowering{1}(space(f))→space(f)
    testbandedblockbandedoperator(Jx)
    @test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10


    Jy=MultivariateOrthogonalPolynomials.Lowering{2}(space(f))→space(f)
    testbandedblockbandedoperator(Jy)
    @test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10

    Jz=MultivariateOrthogonalPolynomials.Lowering{3}(space(f))→space(f)
    testbandedblockbandedoperator(Jz)
    @test norm((Jz*f-Fun((x,y)->(1-x-y)*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10

    for K in (KoornwinderTriangle(2,1,1),KoornwinderTriangle(1,2,1))
        testbandedblockbandedoperator(Lowering{1}(K))
        Jx=(Lowering{1}(K)→K)
        testbandedblockbandedoperator(Jx)
    end

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    f = Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1,d))
    Jx,Jy = MultivariateOrthogonalPolynomials.jacobioperators(space(f))
    x,y = fromcanonical(d,0.1,0.2)
    @test (Jx*f)(x,y) ≈ x*f(x,y)
    @test (Jy*f)(x,y) ≈ y*f(x,y)
end

@testset "Triangle Conversion" begin
    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,0,0))
    testbandedblockbandedoperator(C)

    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,1,0))
    testbandedblockbandedoperator(C)

    C = Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(0,0,1))
    testbandedblockbandedoperator(C)
    C=Conversion(KoornwinderTriangle(0,0,0),KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(C)

    Cx = I:KoornwinderTriangle(-1,0,0)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cx)

    Cy = I:KoornwinderTriangle(0,-1,0)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cy)

    Cz = I:KoornwinderTriangle(0,0,-1)→KoornwinderTriangle(0,0,0)
    testbandedblockbandedoperator(Cz)

    C=Conversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    @test eltype(C)==Float64

    f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
    norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,0,1))-f).coefficients) < 1E-11
    C=Conversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,0))-f).coefficients) < 1E-11
    C=Conversion(KoornwinderTriangle(0,1,1),KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(C)
    norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,1,1))-f).coefficients) < 1E-11


    C=Conversion(KoornwinderTriangle(1,1,1),KoornwinderTriangle(2,1,1))
    testbandedblockbandedoperator(C)

    # Test conversions
    K=KoornwinderTriangle(1,1,1)
    f=Fun(K,[1.])
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,1.])
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,1.])
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,0.,1.])
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)
    f=Fun(K,[0.,0.,0.,0.,1.])
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ f(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,2,1))(0.1,0.2) ≈ f(0.1,0.2)

    f=Fun((x,y)->exp(x*cos(y)),K)
    @test f(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,1,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,2,1))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(2,2,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
    @test Fun(f,KoornwinderTriangle(1,1,2))(0.1,0.2) ≈ ((x,y)->exp(x*cos(y)))(0.1,0.2)
end

@testset "Triangle Derivative/Laplacian" begin
    Dx = Derivative(KoornwinderTriangle(1,0,1), [1,0])
    testbandedblockbandedoperator(Dx)

    Dy = Derivative(KoornwinderTriangle(1,0,1), [0,1])
    testbandedblockbandedoperator(Dy)

    Δ = Laplacian(KoornwinderTriangle(1,1,1))
    testbandedblockbandedoperator(Δ)
end


@testset "Triangle derivatives" begin
    K=KoornwinderTriangle(0,0,0)
    f=Fun((x,y)->exp(x*cos(y)),K)
    D=Derivative(space(f),[1,0])
    @test (D*f)(0.1,0.2) ≈ ((x,y)->cos(y)*exp(x*cos(y)))(0.1,0.2) atol=100000eps()

    D=Derivative(space(f),[0,1])
    @test (D*f)(0.1,0.2) ≈  ((x,y)->-x*sin(y)*exp(x*cos(y)))(0.1,0.2) atol=100000eps()

    D=Derivative(space(f),[1,1])
    @test (D*f)(0.1,0.2) ≈ ((x,y)->-sin(y)*exp(x*cos(y)) -x*sin(y)*cos(y)exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()

    D=Derivative(space(f),[0,2])
    @test (D*f)(0.1,0.2) ≈ ((x,y)->-x*cos(y)*exp(x*cos(y)) + x^2*sin(y)^2*exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()

    D=Derivative(space(f),[2,0])
    @test (D*f)(0.1,0.2) ≈ ((x,y)->cos(y)^2*exp(x*cos(y)))(0.1,0.2)  atol=1000000eps()

    d = Triangle(Vec(0,0),Vec(3,4),Vec(1,6))
    K = KoornwinderTriangle(1,0,1,d)
    f=Fun((x,y)->exp(x*cos(y)),K)
    Dx = Derivative(space(f), [1,0])
    x,y = fromcanonical(d,0.1,0.2)
    @test (Dx*f)(x,y) ≈ cos(y)*exp(x*cos(y)) atol=1E-8
    Dy = Derivative(space(f), [0,1])
    @test (Dy*f)(x,y) ≈ -x*sin(y)*exp(x*cos(y)) atol=1E-8

    Δ = Laplacian(space(f))
    testbandedblockbandedoperator(Δ)
    @test (Δ*f)(x,y) ≈ exp(x*cos(y))*(-x*cos(y) + cos(y)^2 + x^2*sin(y)^2) atol=1E-6
end


@testset "Triangle recurrence" begin
    S=KoornwinderTriangle(1,1,1)

    Mx=Lowering{1}(S)
    My=Lowering{2}(S)
    f=Fun((x,y)->exp(x*sin(y)),S)

    @test (Mx*f)(0.1,0.2) ≈ 0.1*f(0.1,0.2)
    @test (My*f)(0.1,0.2) ≈ 0.2*f(0.1,0.2) atol=1E-12
    @test ((Mx+My)*f)(0.1,0.2) ≈ 0.3*f(0.1,0.2) atol=1E-12


    Jx=Mx → S
    Jy=My → S

    @test (Jy*f)(0.1,0.2) ≈ 0.2*f(0.1,0.2) atol=1E-12

    x,y=0.1,0.2

    P0=[Fun(S,[1.])(x,y)]
    P1=Float64[Fun(S,[zeros(k);1.])(x,y) for k=1:2]
    P2=Float64[Fun(S,[zeros(k);1.])(x,y) for k=3:5]


    K=Block(1)
    @test Matrix(Jx[K,K])*P0+Matrix(Jx[K+1,K])'*P1 ≈ x*P0
    @test Matrix(Jy[K,K])*P0+Matrix(Jy[K+1,K])'*P1 ≈ y*P0

    K=Block(2)
    @test Matrix(Jx[K-1,K])'*P0+Matrix(Jx[K,K])'*P1+Matrix(Jx[K+1,K])'*P2 ≈ x*P1
    @test Matrix(Jy[K-1,K])'*P0+Matrix(Jy[K,K])'*P1+Matrix(Jy[K+1,K])'*P2 ≈ y*P1


    A,B,C=Matrix(Jy[K+1,K])',Matrix(Jy[K,K])',Matrix(Jy[K-1,K])'


    @test C*P0+B*P1+A*P2 ≈ y*P1
end

@testset "TriangleWeight" begin
    S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))
    C=Conversion(S,KoornwinderTriangle(0,0,0))
    testbandedblockbandedoperator(C)
    f=Fun(S,rand(10))
    @test f(0.1,0.2) ≈ (C*f)(0.1,0.2)



    @test (Derivative(S,[1,0])*Fun(S,[1.]))(0.1,0.2) ≈ ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

    @test (Laplacian(S)*Fun(S,[1.]))(0.1,0.2) ≈ -2*0.1-2*0.2

    S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))


    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])

    for k=1:10
        v=[zeros(k);1.0]
        @test (Dy*Fun(S,v))(0.1,0.2) ≈ (Derivative([0,1])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
        @test (Dx*Fun(S,v))(0.1,0.2) ≈ (Derivative([1,0])*Fun(Fun(S,v),KoornwinderTriangle(1,1,1)))(0.1,0.2)
    end

    Δ=Laplacian(S)

    f=Fun(S,rand(3))
    h=0.01
    QR=qrfact(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)

    # other domain
    S = TriangleWeight(1,1,1,KoornwinderTriangle(1,1,1))
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    Div = (Dx → KoornwinderTriangle(0,0,0))  + (Dy → KoornwinderTriangle(0,0,0))
    f = Fun(S,rand(10))

    @test (Div*f)(0.1,0.2) ≈ (Dx*f)(0.1,0.2)+(Dy*f)(0.1,0.2)
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == KoornwinderTriangle(0,0,0)


    d = Triangle(Vec(2,3),Vec(3,4),Vec(1,6))
    S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1,d))
    @test domain(S) ≡ d
    @test maxspace(S,KoornwinderTriangle(0,0,0,d)) == KoornwinderTriangle(0,0,0,d)
    Dx = Derivative(S,[1,0])
    Dy = Derivative(S,[0,1])
    @test maxspace(rangespace(Dx),rangespace(Dy)) == rangespace(Dx + Dy) == KoornwinderTriangle(0,0,0,d)
    @test rangespace(Laplacian(S)) ≡ KoornwinderTriangle(1,1,1,d)
    f = Fun(S, randn(10))
    x,y = fromcanonical(S, 0.1,0.2)
    @test weight(S,x,y) ≈ 0.1*0.2*(1-0.1-0.2)
    g = Fun(f, KoornwinderTriangle(0,0,0,d))
    @test g(x,y) ≈ f(x,y)
    @test (Derivative([1,0])*g)(x,y) ≈ (Derivative([1,0])*f)(x,y)
    @test (Laplacian()*g)(x,y) ≈ (Laplacian()*f)(x,y)

    f=Fun(S,rand(3))
    h=0.01
    Δ = Laplacian(S)
    @test maxspace(rangespace(Δ), S) == KoornwinderTriangle(1,1,1,d)
    QR=qrfact(I-h*Δ)
    @time u=\(QR,f;tolerance=1E-7)
    @time g=Fun(f,rangespace(QR))
    @time \(QR,g;tolerance=1E-7)
end
