using SparseQuadrature, MultiIndexing, StaticArrays
using Test

@testset "SparseQuadrature.jl" begin

    @testset "Smolyak Quadrature" include("smolyak.jl")
    @testset "Adaptive sparse quadrature" include("adaptiveSparseGrid.jl")
    
end
