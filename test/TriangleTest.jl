using FixedSizeArrays,Plots,BandedMatrices,
    ApproxFun,MultivariateOrthogonalPolynomials, Base.Test

import MultivariateOrthogonalPolynomials: Recurrence, ProductTriangle, clenshaw

pf=ProductFun((x,y)->exp(x*cos(y)),ProductTriangle(1,1,1))
@test_approx_eq pf(0.1,0.2) exp(0.1*cos(0.2))

f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
@test_approx_eq f(0.1,0.2) ((x,y)->exp(x*cos(y)))(0.1,0.2)

ApproxFun.block(ApproxFun.tensorizer(KoornwinderTriangle(1,1,1)),length(f.coefficients))
ApproxFun.blockrange(ApproxFun.tensorizer(KoornwinderTriangle(1,1,1)),120)

full(ApproxFun.RaggedMatrix(pad(f.coefficients,sum(1:15)),Int[1;1+cumsum(1:15)],15))

cumsum(1:1)

Fun(pf)(0.1,0.2)

ApproxFun.tensorizer(ProductTriangle(1,1,1))


import ApproxFun: ∞
it=ApproxFun.TensorIterator((∞,∞))

C=rand(10,10)
ApproxFun.fromtensor(it,(C.'))


@profile clenshaw(f,0.1,0.2)
Profile.print()
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

Fun([0.,1.],S)

Jx=Mx
Jy=My ↦ S


@test_approx_eq_eps (Jy*f)(0.1,0.2) 0.2*f(0.1,0.2) 1E-12

x,y=0.1,0.2

P0=[Fun([1.],S)(x,y)]
P1=Float64[Fun([zeros(k);1.],S)(x,y) for k=1:2]
P2=Float64[Fun([zeros(k);1.],S)(x,y) for k=3:5]

@test_approx_eq Jx[1,1]*P0+Jx[2,1]'*P1 x*P0
@test_approx_eq Jy[1,1]*P0+Jy[2,1]'*P1 y*P0

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



bk1


P2
cfs

mult = ([diag(x*ones(n+1,1));diag(y*ones(n+1,1))] - [B1(n);B2(n)])';
bk = cfs(1:n+1,n+1) + mult*([A1(n);A2(n)]'\bkp1) - [C1(n+1);C2(n+1)]'*([A1(n+1);A2(n+1)]'\bkp2);

This is cocdde
