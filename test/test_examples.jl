examples_path = joinpath(@__DIR__, "..", "examples")
dir = readdir(examples_path)
filter!(file -> endswith(file, ".jl"), dir)
for example_path in dir
    script = joinpath(examples_path, example_path)
    mod = @eval module $(gensym()) end
    Base.include(mod, script) # make sure each script is self-contained
end