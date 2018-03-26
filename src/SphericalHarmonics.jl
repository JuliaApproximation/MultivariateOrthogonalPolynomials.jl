export Sphere, SphericalHarmonics, ProductSphericalHarmonics

struct Sphere <: Domain{Vec{3,Float64}} end



checkpoints(::Sphere) = [Vec(-0.5751280945468711,-0.33435465685352894,-0.7466154554409142),
                         Vec(-0.2905718118534361,-0.7696562684874642,0.5685043979901683),
                         Vec(-0.7647193583394475,0.5439082568028465,0.3454969047076867)]
# (1-x^2)^m/2*P
struct ProductSphericalHarmonics <: AbstractProductSpace{Tuple{WeightedJacobi{Segment{Float64},Float64},
                                                        Laurent{PeriodicInterval{Float64},ComplexF64}},
                                                  Sphere,ComplexF64} end


struct SphericalHarmonics <: Space{Sphere,Float64} end

struct DoubleWrappedSphere <: Space{Sphere,Float64} end


domain(::ProductSphericalHarmonics) = Sphere()
domain(::DoubleWrappedSphere) = Sphere()

canonicaldomain(sp::ProductSphericalHarmonics) = Segment(-1,1) * PeriodicInterval()


# we want to represent as τ and φ where  τ = cos(θ), to use OPs

sphericalcoordinates(x,y,z) = Vec(acos(z), atan2(y,x))
isphericalcoordinates(θ, φ) = Vec(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))

quasi_sphericalcoordinates(x,y,z) = Vec(z,atan2(y,x))
function iquasi_sphericalcoordinates(τ, φ)
   υ = sqrt(1-τ^2)
   Vec(υ*cos(φ), υ*sin(φ), τ)
end

quasi_sphericalcoordinates(xyz::Vec) = quasi_sphericalcoordinates(xyz...)
iquasi_sphericalcoordinates(τφ::Vec) = iquasi_sphericalcoordinates(τφ...)


tocanonical(sp::ProductSphericalHarmonics, x...) = quasi_sphericalcoordinates(x...)
fromcanonical(sp::ProductSphericalHarmonics, x...) = iquasi_sphericalcoordinates(x...)

function points(d::ProductSphericalHarmonics,n,m)
    ptsx = points(columnspace(d,1),n)
    ptst = points(factor(d,2),m)

    [fromcanonical(d,x,t) for x in ptsx, t in ptst]
end

function ProductFun(f::Function, S::ProductSphericalHarmonics, M::Integer, N::Integer; tol=100eps())
    xy = checkpoints(S)
    T = promote_type(eltype(f(first(xy)...)), rangetype(S))
    pts = points(S,M,N)
    vals = T[f(pt...) for pt in pts]
    ProductFun(transform!(S,vals),S;tol=tol,chopping=true)
end

function factor(T::ProductSphericalHarmonics, k::Integer)
    @assert k == 2
    Laurent()
end

function columnspace(T::ProductSphericalHarmonics, k::Integer)
    m = k ÷ 2
    JacobiWeight(m/2,m/2,Jacobi(m,m))
end
