using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, StaticArrays, LazyBandedMatrices, BlockArrays, Test
using ContinuumArrays: grid

####
# 2D
####

P = RectPolynomial(Legendre(), Legendre())
f = (x,y) -> exp(x*cos(y))
c = transform(P, splat(f))

𝐟 = P * c # polynomial expansion to roughly machine precision
𝐱 = SVector(0.1,0.2) # point of evaluation
@test 𝐟[𝐱] ≈ f(𝐱...)

N = 60 # polynomial degree + 1
@test P[𝐱,Block.(1:N)]' * c[Block.(1:N)] ≈ f(𝐱...)

N = 10
@test P[𝐱,Block.(1:N)]' * c[Block.(1:N)] ≈ f(𝐱...) atol=1E-5 # only 5 digits accuracy



####
# 3D
####

# following isn't implemented yet:
# P = KronPolynomial(Legendre(), Legendre(), Legendre())

# instead we can do the transform manually, using 1D plan_transforms
# the following only uses ClassicalOrthogonalPolynomials.jl, no functionality from MultivariateOrthogonalPolynomials.jl
# We first redo the 2D example but manually

P = Legendre() # 1D basis

f = (x,y) -> exp(x*cos(y-0.1)) 
N = 100 # compute N^2 coefficients as a matrix
Pl = plan_transform(P, (N,N)) # plan 2D transform from tensor product grid
g = grid(P, N) # vector with grid points corresponding to 1D transform
F = [f(x,y) for x in g, y in g] # matrix of f evaluated at tensor grid
C = Pl * F # coefficient matrix in tensor basis
@test P[0.1,1:N]' * C * P[0.2,1:N] ≈ f(0.1,0.2)

c = DiagTrav(C) # arrange by total degree
ret = 0.0
for n=0:N-1, k=0:n
    ret += c[Block(n+1)[k+1]] * P[0.1,n-k + 1] * P[0.2,k + 1]
end
@test ret ≈ f(0.1,0.2)

f = (x,y,z) -> exp(x*cos(y-0.1)+z)
N = 50 # compute N^3 coefficients as a 3-array
g = grid(P, N) # vector with grid points corresponding to 1D transform
Pl = plan_transform(P, (N,N,N))

F = [f(x,y,z) for x in g, y in g, z in g] # 3-array of f evaluated at tensor grid
C = Pl * F # coefficient 3-array in tensor basis

@test sum(C[k,j,ℓ]*P[0.1,k]*P[0.2,j]*P[0.3,ℓ] for k = 1:N, j = 1:N, ℓ = 1:N) ≈ f(0.1,0.2,0.3)


c = DiagTrav(C) # arrange by total degree
ret = 0.0
for n=1:N
    iter=1
        for i=reverse(1:n)
            k=0
            for j=reverse(1:(n-i+1)) 
                k+=1
                ret += c[Block(n)[iter]] * P[0.1,i] * P[0.2,j] * P[0.3,k]
                iter+=1
                
            end
        end
end
@test ret ≈ f(0.1,0.2,0.3)