module SparseQuadrature

using MultiIndexing, FFTW, StaticArrays, Serialization

include("sortedList.jl")
include("univariateQuadrature.jl")
include("leja.jl")
include("smolyak.jl")
include("adaptiveSparseGrid.jl")

end
