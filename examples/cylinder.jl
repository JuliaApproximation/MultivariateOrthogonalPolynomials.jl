using MultivariateOrthogonalPolynomials, GLMakie

# --- Parameters ---
r = 1.0   # cylinder radius
h = 2.0   # cylinder half-height
n = 100   # grid resolution per axis

# --- 3D grid ---
xs = LinRange(-r, r, n)
ys = LinRange(-r, r, n)
zs = LinRange(-h, h, n)

# --- Scalar field: f(x, y, z) = sin(3z) * exp(-(x²+y²)) ---
# Masked to NaN outside the cylinder
vol = [
    (x^2 + y^2 <= r^2) ? sin(3z) * exp(-(x^2 + y^2)) : NaN
    #sin(3z) * exp(-(x^2 + y^2)) 
    for x in xs, y in ys, z in zs
]

# --- Plot ---
fig = Figure(size = (800, 700))
ax  = Axis3(fig[1, 1],
    title   = "f(x,y,z) = sin(3z)⋅exp(-(x²+y²)) inside a Cylinder",
    xlabel  = "X", ylabel = "Y", zlabel = "Z",
    azimuth = π / 5
)

# Volume rendering inside the cylinder
plt = volumeslices!(ax,xs,ys,zs, vol)
display(fig)