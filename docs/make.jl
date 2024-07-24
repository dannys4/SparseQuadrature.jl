using SparseQuadrature
using Documenter

DocMeta.setdocmeta!(SparseQuadrature, :DocTestSetup, :(using SparseQuadrature); recursive=true)

makedocs(;
    modules=[SparseQuadrature],
    authors="Daniel Sharp <dannys4@mit.edu> and contributors",
    sitename="SparseQuadrature.jl",
    format=Documenter.HTML(;
        canonical="https://dannys4@mit.edu.github.io/SparseQuadrature.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dannys4@mit.edu/SparseQuadrature.jl",
    devbranch="main",
)
