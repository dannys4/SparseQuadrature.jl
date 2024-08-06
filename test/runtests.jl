using SparseQuadrature, MultiIndexing, StaticArrays
using Test

@testset "SparseQuadrature.jl" begin
    @testset "Sorted list" begin
        include("sortedList.jl")
    end
    @testset "Smolyak Quadrature" begin
        include("smolyak.jl")
    end
    @testset "Adaptive sparse quadrature" begin
        include("adaptiveSparseGrid.jl")
    end
    
end
