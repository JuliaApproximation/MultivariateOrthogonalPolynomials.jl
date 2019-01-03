using ApproxFun, MultivariateOrthogonalPolynomials, Plots, SparseArrays
import ApproxFun: Vec, PiecewiseSegment, ZeroOperator, Block
    import MultivariateOrthogonalPolynomials: DirichletTriangle

d = [Triangle(Vec(1,1),Vec(0,1),Vec(0,2)) , Triangle(Vec(1,2),Vec(0,2),Vec(1,1)),
    Triangle(Vec(0,0), Vec(1,0), Vec(0,1)) , Triangle(Vec(1,1),Vec(1,0),Vec(0,1)) ,
    Triangle(Vec(1,0), Vec(2,0), Vec(1,1)) , Triangle(Vec(2,1), Vec(1,1), Vec(2,0))]


p = plot()
    for ▴ in d
        plot!(▴)
    end
    p

∂d = components(PiecewiseSegment([Vec(0,2), Vec(0,1), Vec(0,0), Vec(1,0), Vec(2,0), Vec(2,1), Vec(1,1), Vec(1,2), Vec(0,2)]))

for s in ∂d
    plot!(s)
end
p

# ιd = [Segment(Vec(0,2), Vec(1,1)), Segment(Vec(0,1), Vec(1,1)),
#         Segment(Vec(0,1),Vec(1,0)), Segment(Vec(1,0), Vec(1,1)),
#         Segment(Vec(1,1), Vec(2,0))]


# interfaces
ιd = Dict{NTuple{2,Int}, Segment{Vec{2,Int}}}()
    ιd[(1,2)] = Segment(Vec(0,2), Vec(1,1))
    ιd[(1,4)] = Segment(Vec(0,1), Vec(1,1))
    ιd[(3,4)] = Segment(Vec(0,1),Vec(1,0))
    ιd[(4,5)] = Segment(Vec(1,0), Vec(1,1))
    ιd[(5,6)] = Segment(Vec(1,1), Vec(2,0))


keys(ιd)


ds = vcat(fill.(DirichletTriangle{1,1,1}.(d),3)...) # repeat each triangle 3 times
rs = [Legendre.(∂d)...,
        Legendre(ιd[1,4]), Legendre(ιd[4,5]),  # straight interface
        Legendre(ιd[1,2]), Legendre(ιd[3,4]), Legendre(ιd[5,6]),
        Legendre(ιd[1,4]), Legendre(ιd[4,5]),  # straight interface
        Legendre(ιd[1,2]), Legendre(ιd[3,4]), Legendre(ιd[5,6]),
        vcat(fill.(JacobiTriangle.(d),3)...)...] # diagonal interface



# rs = [Legendre.(∂d); fill.(Legendre.(values(ιd)),2)...; fill.(JacobiTriangle.(d),3)...]

ui  = T -> 1 + (T-1)*3
ux = T -> 2 + (T-1)*3
uy = T -> 3 + (T-1)*3

Dx = Derivative([1,0])
Dy = Derivative([0,1])

N = 100
sprs = A -> (global N; sparse(A[Block.(1:N), Block.(1:N)]))

A = Matrix{SparseMatrixCSC{Float64,Int}}(undef, length(rs), length(ds))
    for K=1:length(rs), J=1:length(ds) # fill with zeros
        A[K,J] = ZeroOperator(ds[J],rs[K]) |> sprs
    end
    # add boundary conditions
    K = 0;
    K +=1; T = 1; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 3; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 3; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 5; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 6; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 6; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 2; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    K +=1; T = 2; J = ui(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    # add dirichlet interface conditions
    for (T1,T2) in ((1,4), (4,5), (1,2), (3,4), (5,6))
        global K +=1;
        J = ui(T1); A[K,J] = (I : ds[J] → rs[K]) |> sprs
        J = ui(T2); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
    end
    # add lr neumann
    K +=1;
    T = 1; J = uy(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    T = 4; J = uy(T); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
    # add ud neumann
    K +=1;
    T = 4; J = ux(T); A[K,J] = (I : ds[J] → rs[K]) |> sprs
    T = 5; J = ux(T); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
    # add diagonal neumann
    for (T1,T2) in ((1,2), (3,4), (5,6))
        global K +=1;
        J = ux(T1); A[K,J] = (I : ds[J] → rs[K]) |> sprs
        J = uy(T1); A[K,J] = (I : ds[J] → rs[K]) |> sprs
        J = ux(T2); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
        J = uy(T2); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
    end

    for T in 1:length(d)
        @show K,T
        global K +=1;
        J = ui(T); A[K,J] = (Dx : ds[J] → rs[K]) |> sprs
        J = ux(T); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
        global K +=1;
        J = ui(T); A[K,J] = (Dy : ds[J] → rs[K]) |> sprs
        J = uy(T); A[K,J] = -(I : ds[J] → rs[K]) |> sprs
        global K +=1;
        J = ux(T); A[K,J] = (Dx : ds[J] → rs[K]) |> sprs
        J = uy(T); A[K,J] = (Dy : ds[J] → rs[K]) |> sprs
    end

M = hvcat(ntuple(_ -> size(A,2),size(A,1)), permutedims(A)...)


rhs = vcat(coefficients.(Fun.(Ref((x,y) -> real(exp(x+im*y))), rs[1:length(∂d)], N))...)

rhs = vcat(coefficients.(Fun.(Ref((x,y) -> x^2), rs[1:length(∂d)], N))...)


F = factorize(M)
u_cfs = F \ pad(rhs, size(M,1))

u1 = Fun(ds[4], u_cfs[(4-1)*sum(1:N)+1:4*sum(1:N)])
u1(0.99,0.99)

u1.coefficients
u1(0.1,1.2)-real(exp(0.1+im*1.2))

plot(abs.([norm((M*u_cfs - pad(rhs, size(M,1)))[N*(K-1)+1:N*K]) for K=1:length(rs)] ); yscale=:log10)

length(rs)
K = 30; norm((M*u_cfs - pad(rhs, size(M,1)))[N*(K-1)+1:N*K])


U = Fun.(Ref((x,y) -> real(exp(x+im*y))), d, sum(1:N))


u_cfs = Vector{Float64}()
    for T in d
        append!(u_cfs, pad(Fun((x,y) -> exp(x)*cos(y), DirichletTriangle{1,1,1}(T)).coefficients, sum(1:N)))
        append!(u_cfs, pad(Fun((x,y) -> exp(x)*cos(y), DirichletTriangle{1,1,1}(T)).coefficients, sum(1:N)))
        append!(u_cfs, pad(Fun((x,y) -> -exp(x)*sin(y), DirichletTriangle{1,1,1}(T)).coefficients, sum(1:N)))
    end

u_cfs

((M*u_cfs) - pad(rhs, size(M,1))) |> norm
((M*u_cfs) - pad(rhs, size(M,1)))[1:(N*(length(rs)-6*3))] |> norm
NN = (N*(length(rs)-6*3))
((M*u_cfs) - pad(rhs, size(M,1)))[NN+1:NN+sum(1:N)] |> norm
((M*u_cfs) - pad(rhs, size(M,1)))[NN+sum(1:N):NN+2sum(1:N)] |> norm


T = d[1]
u1 = pad(Fun((x,y) -> exp(x)*cos(y), DirichletTriangle{1,1,1}(T)).coefficients, sum(1:N))
u1y = pad(Fun((x,y) -> -exp(x)*sin(y), DirichletTriangle{1,1,1}(T)).coefficients, sum(1:N))



A[19,1]*u1 + A[19,3]*u1y

NN


size(M)
size(u_cfs)
size(rhs)

M*u_cfs - pad(rhs,size(M,1))  |> norm

(Dy*u1)(0.1,1.2)

Fun((x,y) -> -exp(x)*sin(y), DirichletTriangle{1,1,1}(T))(0.1,1.2)





A[18,1]*u_cfs[1:sum(1:N)] +
    A[18,2]*u_cfs[sum(1:N)+1:2sum(1:N)]
å
(length(rs)-6*3)

length(rs)

6*



(I : ds[1] → rs[1])[Block.(1:N), Block.(1:N)] * u_cfs[1:210]














M[1:20,1:210]u_cfs[1:210]

Matrix(M[1:20,211:end]) |> norm

A[1,:]

















u1.coefficients


## Interlace operator



M

N,M = length(rs), length(ds)
    A = Matrix{Operator{Float64}}(undef, N,M)
    for K=1:N, J=1:M # fill with zeros
        A[K,J] = ZeroOperator(ds[J],rs[K])
    end
    # add boundary conditions
    K = 0;
    K +=1; T = 1; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 3; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 3; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 5; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 6; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 6; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 2; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    K +=1; T = 2; J = ui(T); A[K,J] = I : ds[J] → rs[K]
    # add dirichlet interface conditions
    for (T1,T2) in ((1,4), (4,5), (1,2), (3,4), (5,6))
        global K +=1;
        J = ui(T1); A[K,J] = I : ds[J] → rs[K]
        J = ui(T2); A[K,J] = -I : ds[J] → rs[K]
    end
    # add lr neumann
    K +=1;
    T = 1; J = uy(T); A[K,J] = I : ds[J] → rs[K]
    T = 4; J = uy(T); A[K,J] = -I : ds[J] → rs[K]
    # add up neumann
    K +=1;
    T = 4; J = ux(T); A[K,J] = I : ds[J] → rs[K]
    T = 5; J = ux(T); A[K,J] = -I : ds[J] → rs[K]
    # add diagonal neumann
    for (T1,T2) in ((1,2), (3,4), (5,6))
        global K +=1;
        J = ux(T1); A[K,J] = I : ds[J] → rs[K]
        J = uy(T1); A[K,J] = I : ds[J] → rs[K]
        J = ux(T2); A[K,J] = -I : ds[J] → rs[K]
        J = uy(T2); A[K,J] = -I : ds[J] → rs[K]
    end

    for T in 1:length(d)
        global K +=1;
        J = ui(T); A[K,J] = I : ds[J] → rs[K]
        J = ux(T); A[K,J] = -Dx : ds[J] → rs[K]
        global K +=1;
        J = ui(T); A[K,J] = I : ds[J] → rs[K]
        J = uy(T); A[K,J] = -Dy : ds[J] → rs[K]
        global K +=1;
        J = ux(T); A[K,J] = Dx : ds[J] → rs[K]
        J = uy(T); A[K,J] = Dy : ds[J] → rs[K]
    end

L = Operator(A)

N = 20
M = sparse(L[Block.(1:N), Block.(1:N)])

rhs = Fun.(Ref((x,y) -> real(exp(x+im*y))), rs[1:length(∂d)], N)

u_cfs = M \ pad(vcat(cfs...), size(M,1))

u_cfs[1:sum(1:N)]

Fun(rhs)

rangespace(L)


J
ds[7]

[randn(2,2) randn(2,2);
 randn(2,2) randn(2,2)]


U = Fun.(Ref((x,y) -> real(exp(x+im*y))), d, sum(1:N))
    u_cfs = Vector{Float64}()
    for u in U
        append!(u_cfs, u.coefficients)
        append!(u_cfs, (Dx*u).coefficients)
        append!(u_cfs, (Dy*u).coefficients)
    end

M*u_cfs
ds[1]
rs[1]


(I : ds[1] → rs[1])*U[1]

# add boundary conditions
for K = 1:length(∂d)
    for J = 1:length(d)
        if domain(rs[K]) ⊆ domain(ds[J])
            A[K,J] = I : ds[J] → rs[K]
            break
        end
    end
end

Ñ = length(∂d)
for (K, J1, J2) in ((Ñ+1, 1, 4),
                    (Ñ+2, 1, 2))
    A[K,J1] =  I : ds[J1] → rs[K]
    A[K,J2] = -I : ds[J2] → rs[K]
end

Operator(A)[1:20,1:20]

rs


# add continuity
for K = 1:length(ιd)



∂d[1] ⊆ d[1]

rs[3]



typeof(components(d))




length(ιd)




Vec{2,Float64}
Segment(Vec(0,0), Vec(1.0,0)) |> typeof



import Makie



S = DirichletTriangle{1,1,1}.(components(d))

C1 = I : S[1] → Legendre(Segment(Vec(0,0),Vec(1,0)))
C2 = I : S[1] → Legendre(Segment(Vec(1,0),Vec(0,0)))

a = Legendre(Segment(Vec(0,0),Vec(1,0)))
b = Legendre(Segment(Vec(1,0),Vec(0,0)))

@which Conversion(a,b)




DirichletTriangle{0,1,1}(d[1])

2
C1.op.op.ops




C2.op.op.ops

C1 == C2

C1[1:20,1:20] == C2[1:20,1:20]

Legendre(Segment(Vec(0,0),Vec(1,0)))

∂d = PiecewiseSegment([Vec(0.,0), Vec(1.,0), Vec(1,1), Vec(1,2), Vec(0,1), Vec(-1,1.5), Vec(0,0)])

o = Fun.(S, Ref([1.0]))

o[1]
