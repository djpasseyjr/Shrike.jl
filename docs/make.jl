using Documenter, Shrike
push!(LOAD_PATH,"../src/")

makedocs(sitename="Shrike.jl Documentation")
deploydocs(
    repo = "github.com/djpasseyjr/Shrike.jl.git",
    devbranch = "main"
)
