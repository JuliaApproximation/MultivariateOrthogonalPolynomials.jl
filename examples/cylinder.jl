using MultivariateOrthogonalPolynomials, GLMakie



x = LinRange(-1.1, 1.1, 500)
y = x
z = LinRange(-1, 1, 500)
vol = [
    (x^2 + y^2 <= 1) ? sin(3z+10x) * exp(-(x^2 + y^2)) : NaN
    # sin(3z) * exp(-(x^2 + y^2)) 
    for x in x, y in y, z in z
]

fig = Figure(;size=(1000,1000))
ax = LScene(fig[1, 1], show_axis=false)

plt = volumeslices!(ax, x, y, z, vol; colorrange = (-1.0, 1.0))

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