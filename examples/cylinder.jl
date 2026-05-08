using MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, GLMakie, BlockArrays, StaticArrays

f = (x,y,z) -> sin(3z+10x) * exp(-(x^2 + y^2))

M,n = Block(30),30
𝐱 = ContinuumArrays.grid(Zernike(), M)
x,y = first.(𝐱), last.(𝐱)
z = ContinuumArrays.grid(Fourier(), n)

Z_pl = plan_transform(Zernike(), M)
F_pl = plan_transform(Fourier(), n)


C = BlockMatrix{Float64}(undef, (axes(Zernike(),2)[Block.(Base.OneTo(Int(M+1)))], Base.OneTo(n)))

F = f.(x, y, reshape(z,1,1,:))
for k in axes(F,1), j in axes(F,2)
    F[k,j,:] = F_pl*F[k,j,:]
end
for ℓ in axes(F,3)
    C[:,ℓ] = Z_pl*F[:,:,ℓ]
end

@test Zernike()[SVector(0.1,0.2),Block(1):(M+1)]'C*Fourier()[0.3,1:n] ≈ f(0.1,0.2,0.3)

Z_ipl = inv(Z_pl)
F_ipl = inv(F_pl)



volumeslice()

x
y
z
F

vol = [
    (x^2 + y^2 <= 1) ? sin(3z+10x) * exp(-(x^2 + y^2)) : NaN
    # sin(3z) * exp(-(x^2 + y^2)) 
    for x in x, y in y, z in z
]

fig = Figure(;size=(1000,1000))
ax = LScene(fig[1, 1], show_axis=false)

X = range(-1, 1, 100)
Y = range(-1, 1, 100)
Z = range(-1, 1, 100)

M

[x^2 + y^2 ≤ 1 ? Zernike()[SVector(x,y),Block(1):(M+1)] : NaN for x in X, y in Y]

[ for x in X, y in Y]

plt = volumeslices!(ax, x, y, z, F; colorrange = (-1.0, 1.0))

# Typical interactive pattern with sliders
sgrid = SliderGrid(
    fig[2, 1],
    (label = "YZ slice (x)", range = 1:length(x)),
    (label = "XZ slice (y)", range = 1:length(y)),
    (label = "XY slice (z)", range = 1:length(z)),
)

sl_yz, sl_xz, sl_xy = sgrid.sliders

on(sl_yz.value) do v; plt[:update_yz][](v) end
on(sl_xz.value) do v; plt[:update_xz][](v) end
on(sl_xy.value) do v; plt[:update_xy][](v) end