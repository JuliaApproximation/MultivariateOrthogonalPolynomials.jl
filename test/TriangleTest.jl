using FixedSizeArrays,Plots,BandedMatrices,
    ApproxFun,MultivariateOrthogonalPolynomials, Base.Test

import MultivariateOrthogonalPolynomials: Recurrence, ProductTriangle, clenshaw, block
import ApproxFun: bandedblockbandedoperatortest

pf=ProductFun((x,y)->exp(x*cos(y)),ProductTriangle(1,1,1))
@test_approx_eq pf(0.1,0.2) exp(0.1*cos(0.2))

f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
@test_approx_eq f(0.1,0.2) exp(0.1*cos(0.2))

# Test recurrence operators
Jx=MultivariateOrthogonalPolynomials.Recurrence(1,space(f))

#bandedblockbandedoperatortest(Jx)

# Jx[1:5,1:5]|>full
#
# Jx[2:5,1:5]|>full
#
# S=view(Jx,2:5,1:5)
#
# kr,jr=parentindexes(S)
# KO=parent(S)
# l,u=ApproxFun.blockbandinds(KO)
# λ,μ=ApproxFun.subblockbandinds(KO)
#
# rt=ApproxFun.rangetensorizer(KO)
# dt=ApproxFun.domaintensorizer(KO)
# ret=ApproxFun.bbbzeros(S)
#
# Bs=ApproxFun.viewblock(ret,2,2)
# K=J=2
# kshft=kr[1]+ApproxFun.blockstart(rt,K)-2
# j=1
#
#
# for J=1:ApproxFun.blocksize(ret,2)
#     jshft=jr[1]+ApproxFun.blockstart(dt,J)-2
#     for K=ApproxFun.blockcolrange(ret,J)
#         Bs=ApproxFun.viewblock(ret,K,J)
#         kshft=kr[1]+ApproxFun.blockstart(rt,K)-2
#         for j=1:size(Bs,2),k=ApproxFun.colrange(Bs,j)
#             @show K,J,k,j,k+kshft,j+jshft
#             Bs[k,j]=KO[k+kshft,j+jshft]
#         end
#     end
# end

@test ApproxFun.colstop(Jx,1) == 2
@test ApproxFun.colstop(Jx,2) == 4
@test ApproxFun.colstop(Jx,3) == 5
@test ApproxFun.colstop(Jx,4) == 7


@test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))
@test isa(Jy[1:10,1:10],ApproxFun.BandedBlockBandedMatrix)
@test_approx_eq Jy[3,1] 1/3

@test ApproxFun.colstop(Jy,1) == 3
@test ApproxFun.colstop(Jy,2) == 5
@test ApproxFun.colstop(Jy,3) == 6
@test ApproxFun.colstop(Jy,4) == 8

@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
@test_approx_eq pyf(0.1,0.2) 0.2exp(0.1*cos(0.2))





C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))
@test eltype(C)==Float64
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,0,1))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,0))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(0,1,1),KoornwinderTriangle(1,1,1))
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,1,1))-f).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))↦space(f)
@test isa(Jy[1:10,1:10],ApproxFun.BandedBlockBandedMatrix)

@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-10




K=KoornwinderTriangle(0,0,0)

f=Fun([1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,0.,0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)
f=Fun([0.,0.,0.,0.,1.],K)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) f(0.1,0.2)


f=Fun((x,y)->exp(x*cos(y)),K)
@test_approx_eq f(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,0,0))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,0))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(1,1,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)
@test_approx_eq Fun(f,KoornwinderTriangle(0,0,1))(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)








K=KoornwinderTriangle(0,0,0)
f=Fun((x,y)->exp(x*cos(y)),K)
D=Derivative(space(f),[1,0])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->cos(y)*exp(x*cos(y)))(0.1,0.2) 100000eps()

D=Derivative(space(f),[0,1])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-x*sin(y)*exp(x*cos(y)))(0.1,0.2) 100000eps()

D=Derivative(space(f),[1,1])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-sin(y)*exp(x*cos(y)) -x*sin(y)*cos(y)exp(x*cos(y)))(0.1,0.2)  1000000eps()

D=Derivative(space(f),[0,2])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->-x*cos(y)*exp(x*cos(y)) + x^2*sin(y)^2*exp(x*cos(y)))(0.1,0.2)  1000000eps()

D=Derivative(space(f),[2,0])
@test_approx_eq_eps (D*f)(0.1,0.2) ((x,y)->cos(y)^2*exp(x*cos(y)))(0.1,0.2)  1000000eps()



## Recurrence



S=KoornwinderTriangle(1,1,1)

Mx=Recurrence(1,S)
My=Recurrence(2,S)
f=Fun((x,y)->exp(x*sin(y)),S)

@test_approx_eq (Mx*f)(0.1,0.2) 0.1*f(0.1,0.2)
@test_approx_eq_eps (My*f)(0.1,0.2) 0.2*f(0.1,0.2) 1E-12
@test_approx_eq_eps ((Mx+My)*f)(0.1,0.2) 0.3*f(0.1,0.2) 1E-12


Jx=Mx
Jy=My ↦ S


@test_approx_eq_eps (Jy*f)(0.1,0.2) 0.2*f(0.1,0.2) 1E-12

x,y=0.1,0.2

P0=[Fun([1.],S)(x,y)]
P1=Float64[Fun([zeros(k);1.],S)(x,y) for k=1:2]
P2=Float64[Fun([zeros(k);1.],S)(x,y) for k=3:5]

@test_approx_eq Jx[1:1,1:1]*P0+Jx[2:3,1:1]'*P1 x*P0
@test_approx_eq Jy[1:1,1:1]*P0+Jy[2:3,1:1]'*P1 y*P0

k=2
@test_approx_eq Jx[k-1,k]'*P0+Jx[k,k]'*P1+Jx[k+1,k]'*P2 x*P1
@test_approx_eq Jy[k-1,k]'*P0+Jy[k,k]'*P1+Jy[k+1,k]'*P2 y*P1


A,B,C=Jy[k+1,k]',Jy[k,k]',Jy[k-1,k]'


@test_approx_eq C*P0+B*P1+A*P2 y*P1

cfs0=rand(1)
cfs1=[0.,0.]
cfs2=[1.,0.,0.]

cfs=Vector{Float64}[]
    for k=1:20
        push!(cfs,rand(k))
    end


x,y=0.01,0.9

@time clenshaw2D(Jx,Jy,cfs,x,y)
@time Fun([cfs...;],S)(x,y)


f=Fun((x,y)->exp(x*cos(x*y)),KoornwinderTriangle(1,1,1))
    cfs=ApproxFun.totree(f.coefficients)

clenshaw2D(Jx,Jy,cfs,x,y)-exp(x*cos(x*y))

f(x,y)-exp(x*cos(x*y))

f.coefficients
## Derive

bk1=zeros(length(cfs)+1)
    bk2=zeros(length(cfs)+2)

    for n=3:-1:1
        Ap1=Jx[n+2,n+1]'
        Ap2=Jy[n+2,n+1]'
        A1=Jx[n+1,n]'
        A2=Jy[n+1,n]'
        B1=Jx[n,n]'
        B2=Jy[n,n]'
        C1=Jx[n,n+1]'
        C2=Jy[n,n+1]'

        bk1,bk2=cfs[n] + ([diagm(fill(x,n)) diagm(fill(y,n))]-[B1' B2'])*([A1' A2']\bk1) -
            [C1' C2']*([Ap1' Ap2']\bk2),bk1
    end
    bk1



Fun([cfs...;],S)(x,y)
n=2
    Ap1=Jx[n+2,n+1]'
    Ap2=Jy[n+2,n+1]'
    A1=Jx[n+1,n]'
    A2=Jy[n+1,n]'
    B1=Jx[n,n]'
    B2=Jy[n,n]'
    C1=Jx[n,n+1]'
    C2=Jy[n,n+1]'

    bk1,bk2=cfs[n] + ([diagm(fill(x,n)) diagm(fill(y,n))]-[B1' B2'])*([A1' A2']\bk1) -
        [C1' C2']*([Ap1' Ap2']\bk2),bk1

n=1
    Ap1=Jx[n+2,n+1]'
    Ap2=Jy[n+2,n+1]'
    A1=Jx[n+1,n]'
    A2=Jy[n+1,n]'
    B1=Jx[n,n]'
    B2=Jy[n,n]'
    C1=Jx[n,n+1]'
    C2=Jy[n,n+1]'

    bk1,bk2=cfs[n] + ([diagm(fill(x,n)) diagm(fill(y,n))]-[B1' B2'])*([A1' A2']\bk1) -
        [C1' C2']*([Ap1' Ap2']\bk2),bk1




mult = ([diag(x*ones(n+1,1));diag(y*ones(n+1,1))] - [B1(n);B2(n)])';
bk = cfs(1:n+1,n+1) + mult*([A1(n);A2(n)]'\bkp1) - [C1(n+1);C2(n+1)]'*([A1(n+1);A2(n+1)]'\bkp2);
