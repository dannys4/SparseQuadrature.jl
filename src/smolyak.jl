export SmolyakQuadrature

function tensor_prod_quad(pts_wts_zipped, ::Val{d}) where {d}
    points_1d = [p for (p, _) in pts_wts_zipped]
    log_wts_1d = [[(sign(w), log(abs(w))) for w in wts] for (_, wts) in pts_wts_zipped]
    # Create all indices for the tensor product rule
    lengths_1d = ntuple(k -> length(points_1d[k]), d)
    idxs = CartesianIndices(lengths_1d)
    points = Vector{SVector{d,Float64}}(undef, length(idxs))
    weights = zeros(Float64, length(idxs))
    @inbounds for (j, idx) in enumerate(idxs)
        points[j] = SVector{d}(ntuple(k -> points_1d[k][idx[k]], d))
        weights[j] = exp(sum(log_wts_1d[k][idx[k]][2] for k in 1:d)) * prod(log_wts_1d[k][idx[k]][1] for k in 1:d)
    end
    points, weights
end

function tensor_prod_quad(midx::SVector{d,Int}, rules::Union{<:AbstractVector,<:Tuple}) where {d}
    rules_eval = ntuple(i -> rules[i](midx[i]), d)
    tensor_prod_quad(rules_eval, Val{d}())
end

"""
    SmolyakQuadrature(mset, rules)

Create a Smolyak quadrature rule from a multi-index set and a set of rules

# Arguments
- `mset`: MultiIndexSet
- `rules`: Vector of rules for each dimension. `rules[j](n::Int)` should return a quadrature rule `(pts,wts)` for dimension `j` exact up to order `n`

"""
function SmolyakQuadrature(mset::MultiIndexSet{d}, rules::Union{<:AbstractVector,<:Tuple}) where {d}
    if length(rules) != d
        throw(ArgumentError("Number of rules must match dimension"))
    end
    quad_rules = smolyakIndexing(mset)
    unique_elems = Dict{SVector{d,Float64},Float64}()
    for (idx, count) in quad_rules
        midx = mset[idx]
        pts_idx, wts_idx = tensor_prod_quad(midx, rules)
        for (pt, wt) in zip(pts_idx, wts_idx)
            entry = get(unique_elems, pt, 0.0)
            unique_elems[pt] = entry + wt * count
        end
    end
    points = Matrix{Float64}(undef, d, length(unique_elems))
    weights = Vector{Float64}(undef, length(unique_elems))
    for (i, (pt, wt)) in enumerate(unique_elems)
        points[:, i] .= pt
        weights[i] = wt
    end
    points, weights
end

function SmolyakQuadrature(mset::MultiIndexSet{d}, rule::Function) where {d}
    SmolyakQuadrature(mset, ntuple(_ -> rule, d))
end