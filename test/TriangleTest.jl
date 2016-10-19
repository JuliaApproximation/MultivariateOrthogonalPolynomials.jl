using FixedSizeArrays,Plots,BandedMatrices,
        ApproxFun,MultivariateOrthogonalPolynomials, Base.Test
    import MultivariateOrthogonalPolynomials: Recurrence, ProductTriangle, clenshaw, block, TriangleWeight
    import ApproxFun: bandedblockbandedoperatortest, Block

pf=ProductFun((x,y)->exp(x*cos(y)),ProductTriangle(1,1,1))
@test_approx_eq pf(0.1,0.2) exp(0.1*cos(0.2))

f=Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,1))
@test_approx_eq f(0.1,0.2) exp(0.1*cos(0.2))

# Test recurrence operators
Jx=MultivariateOrthogonalPolynomials.Recurrence(1,space(f))

bandedblockbandedoperatortest(Jx)

@test ApproxFun.colstop(Jx,1) == 2
@test ApproxFun.colstop(Jx,2) == 4
@test ApproxFun.colstop(Jx,3) == 5
@test ApproxFun.colstop(Jx,4) == 7


@test norm((Jx*f-Fun((x,y)->x*exp(x*cos(y)),KoornwinderTriangle(1,1,1))).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))

bandedblockbandedoperatortest(Jy)

@test_approx_eq Jy[3,1] 1/3

@test ApproxFun.colstop(Jy,1) == 3
@test ApproxFun.colstop(Jy,2) == 5
@test ApproxFun.colstop(Jy,3) == 6
@test ApproxFun.colstop(Jy,4) == 8

@test norm((Jy*f-Fun((x,y)->y*exp(x*cos(y)),KoornwinderTriangle(1,0,1))).coefficients) < 1E-11

pyf=ProductFun((x,y)->y*exp(x*cos(y)),ProductTriangle(1,0,1))
@test_approx_eq pyf(0.1,0.2) 0.2exp(0.1*cos(0.2))





C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,0,1),KoornwinderTriangle(1,1,1))
bandedblockbandedoperatortest(C)
@test eltype(C)==Float64
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,0,1))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(1,1,0),KoornwinderTriangle(1,1,1))
bandedblockbandedoperatortest(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(1,1,0))-f).coefficients) < 1E-11
C=ApproxFun.ConcreteConversion(KoornwinderTriangle(0,1,1),KoornwinderTriangle(1,1,1))
bandedblockbandedoperatortest(C)
norm((C*Fun((x,y)->exp(x*cos(y)),KoornwinderTriangle(0,1,1))-f).coefficients) < 1E-11


Jy=MultivariateOrthogonalPolynomials.Recurrence(2,space(f))↦space(f)
bandedblockbandedoperatortest(Jy)


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


K=Block(1)
Jx[K,K]|>full
@test_approx_eq Jx[K,K]*P0+Jx[K+1,K]'*P1 x*P0
@test_approx_eq Jy[K,K]*P0+Jy[K+1,K]'*P1 y*P0

K=Block(2)
@test_approx_eq Jx[K-1,K]'*P0+Jx[K,K]'*P1+Jx[K+1,K]'*P2 x*P1
@test_approx_eq Jy[K-1,K]'*P0+Jy[K,K]'*P1+Jy[K+1,K]'*P2 y*P1


A,B,C=Jy[K+1,K]',Jy[K,K]',Jy[K-1,K]'


@test_approx_eq C*P0+B*P1+A*P2 y*P1

cfs0=rand(1)
cfs1=[0.,0.]
cfs2=[1.,0.,0.]

cfs=Vector{Float64}[]
    for k=1:20
        push!(cfs,rand(k))
    end


x,y=0.01,0.9
@time Fun([cfs...;],S)(x,y)


Block(3):-1:Block(1)|>collect

f=Fun((x,y)->exp(x*cos(x*y)),KoornwinderTriangle(1,1,1))
    cfs=ApproxFun.totree(f.coefficients)|

clenshaw2D(Jx,Jy,cfs,x,y)-exp(x*cos(x*y))

f(x,y)-exp(x*cos(x*y))

f.coefficients
## Derive

using SO
K=Block(3)
    A=[Jx[K+1,K] Jy[K+1,K]]
    Ai=A\eye(K.K+1)|>chopm

Q,R=qr(A')

R'*Q'-A|>norm

Q*inv(R')-Ai|>norm

At=[Jx[K+1,K]'; Jy[K+1,K]']

L1=[At[4:end,1:3]*inv(At[1:3,1:3]) zeros(3,3); zeros(3,3) eye(3)]
    L2=[eye(3) eye(3); -eye(3) eye(3)]
    At2=L2*L1*At
    L3=[inv(At2[1:3,1:3]) zeros(3,3); zeros(3,3) eye(3)]
    P=eye(6)[[1:3;6;4:5],:]
    R2=P*L3*L2*L1*At

    inv(L1)*inv(L2)*inv(L3)*P'*R2-At|>norm


At

At2


Ati2=(R2\eye(6))*P*L3*L2*L1*eye(6)
Ati=At\eye(6)


Ati*At
Ati2*At

norm((inv(L1)*inv(L2)*inv(L3)*P'*R2)\eye(6)-At\eye(6))
norm((inv(L2)*inv(L3)*P'*R2)\(L1*eye(6))-At\eye(6))

A'


L*U-A'|>norm
Jy=Recurrence(2,KoornwinderTriangle(1.23,1.0321,2.21))↦KoornwinderTriangle(1.23,1.0321,2.21)
Recurrence(2,KoornwinderTriangle(1.23,1.0321,2.21))|>rangespace

Jy[Block(3),Block(4)]|>full
Recurrence(1,KoornwinderTriangle(1.23,1.0321,2.21))[Block(11),Block(12)]|>full

A*Ai
R

A\[1.,2.,3.]


Ai*[1.,2.,3.]
[Ap1' Ap2']

bk1=zeros(length(cfs)+1)
    bk2=zeros(length(cfs)+2)

    for N=Block(3):-1:Block(1)
        Ap1=Jx[N+2,N+1]'
        Ap2=Jy[N+2,N+1]'
        A1=Jx[N+1,N]'
        A2=Jy[N+1,N]'
        B1=Jx[N,N]'
        B2=Jy[N,N]'
        C1=Jx[N,N+1]'
        C2=Jy[N,N+1]'

        bk1,bk2=cfs[N.K] + ([diagm(fill(x,N.K)) diagm(fill(y,N.K))]-[B1' B2'])*([A1' A2']\bk1) -
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





## Weighted


S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))

@test_approx_eq (Derivative(S,[1,0])*Fun([1.],S))(0.1,0.2) ((x,y)->y*(1-x-y)-x*y)(0.1,0.2)

@test_approx_eq (Laplacian(S)*Fun([1.],S))(0.1,0.2) -2*0.1-2*0.2


Δ=Laplacian(S)

f=Fun(rand(3),S)
QR=qrfact(Δ)
@time u=linsolve(QR,f;tolerance=1E-7)

x=y=linspace(0.,1.,10)

X=[ApproxFun.fromcanonical(domain(f),xy)[1] for xy in tuple.(x,y')]
Y=[ApproxFun.fromcanonical(domain(f),xy)[2] for xy in tuple.(x,y')]


pyplot()
surface(X,Y,u.(X,Y))


S=TriangleWeight(1.,1.,1.,KoornwinderTriangle(1,1,1))
Δ=Laplacian(S)

h=0.0001

f=Fun([1.,2.,3.],S)
QR=qrfact(I-h*Δ)

u=Array(typeof(f),200)
u[1]=f

for k=1:length(u)-1
    @time u[k+1]=chop(linsolve(QR,u[k];tolerance=1E-7),1E-7)
end

x=y=linspace(0.,1.,10)

X=[ApproxFun.fromcanonical(domain(f),xy)[1] for xy in tuple.(x,y')]
Y=[ApproxFun.fromcanonical(domain(f),xy)[2] for xy in tuple.(x,y')]


pyplot()
@gif for k=1:5
    surface(X,Y,u[k].(X,Y);zlims=(-0.1,0.1))
end
