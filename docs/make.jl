using Documenter, RPTrees
push!(LOAD_PATH,"../src/")

makedocs(sitename="RPTrees.jl Documentation")
deploydocs(
    repo = "github.com/djpasseyjr/RPTrees.jl.git",
    devbranch = "main"
)
