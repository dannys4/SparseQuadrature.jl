module SparseQuadrature

using MultiIndexing, StaticArrays, UnivariateApprox

include("sortedList.jl")
include("smolyak.jl")
include("adaptiveSparseGrid.jl")

end
