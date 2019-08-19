using ApproxFun, MultivariateOrthogonalPolynomials, Test

include("test_triangle.jl")
include("test_dirichlettriangle.jl")


##  bessel

# for k=0:10
#     @test Fun(r->besselj(k,r),JacobiSquare(k))(0.9) ≈ besselj(k,0.9)
# end
#
#
# f=Fun([1.],Segment()^2)
#     f(0.1,0.2)
#
#
# f=Fun([1.],Disk())
# @test f(0.1,0.2) ≈ 1.
#
# f=ProductFun(Fun([0.,1.]+0.0im,Disk()))
# x,y=0.1,0.2
# r,θ=sqrt(x^2+y^2),atan2(y,x)
# @test exp(-im*θ)*r ≈ f(x,y)
#
#
# ## Disk
#
#
#
#
# f=(x,y)->exp(x.*sin(y))
# u=ProductFun(f,Disk(),50,51)
# @test u(.1,.1) ≈ f(.1,.1)
#
#
#
#
# # write your own tests here
# # Laplace
# d=Disk()
# u=[dirichlet(d),lap(d)]\Fun(z->real(exp(z)),Circle())
# @test u(.1,.2) ≈ real(exp(.1+.2im))
#
#
#
#
# # remaining numbers determined numerically, may be
# # inaccurate
#
# # Poisson
# f=Fun((x,y)->exp(-10(x+.2).^2-20(y-.1).^2),d)
# u=[dirichlet(d),lap(d)]\[0.,f]
# @test u(.1,.2) ≈ -0.039860694987858845
#
# #Helmholtz
# u=[dirichlet(d),lap(d)+100I]\1.0
# @test_approx_eq_eps u(.1,.2) -0.3675973169667076 1E-11
# u=[neumann(d),lap(d)+100I]\1.0
# @test_approx_eq_eps u(.1,.2) -0.20795862954551195 1E-11
#
# # Screened Poisson
# u=[neumann(d),lap(d)-100.0I]\1.0
# @test_approx_eq_eps u(.1,.9) 0.04313812031635443 1E-11
#
# # Lap^2
# u=[dirichlet(d),neumann(d),lap(d)^2]\Fun(z->real(exp(z)),Circle())
# @test_approx_eq_eps u(.1,.2) 1.1137317420521624 1E-11
#
#
# # Speed Test
#
#
# d = Disk()
# f = Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),d)
# S = discretize([dirichlet(d);lap(d)],100);
# @time S = discretize([dirichlet(d);lap(d)],100);
# u=S\Any[0.;f];
# @time u=S\Any[0.;f];
#
# println("Disk Poisson: should be ~0.16,0.016")
#
#
#
# ## README Test
#
# d = Disk()
# f = Fun((x,y)->exp(-10(x+.2)^2-20(y-.1)^2),d)
# u = [dirichlet(d);lap(d)]\Any[0.,f]
#
#
#
# d = Disk()
# u0 = Fun((x,y)->exp(-50x^2-40(y-.1)^2)+.5exp(-30(x+.5)^2-40(y+.2)^2),d)
# B= [dirichlet(d);neumann(d)]
# L = -lap(d)^2
# h = 0.001
#
# L=Laplacian(Space(d),1)
# @show real(ApproxFun.diagop(L,1)[1:10,1:10])
#
# d = Disk()
# u0 = Fun((x,y)->exp(-50x^2-40(y-.1)^2)+.5exp(-30(x+.5)^2-40(y+.2)^2),d)
# B= [dirichlet(d);neumann(d)]
# L = -lap(d)^2
# h = 0.001
