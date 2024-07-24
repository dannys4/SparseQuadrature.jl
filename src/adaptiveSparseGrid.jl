export AdaptiveSparseGrid, adaptiveIntegrate!

function formDifference1dRule(q_rule, order)
    # Ensure each quadrature rule is sorted
    pts_hi, wts_hi = q_rule(order)
    sort_hi = sortperm(pts_hi)
    order == 0 && return pts_hi, wts_hi

    pts_lo, wts_lo = q_rule(order - 1)
    sort_lo = sortperm(pts_lo)

    diff_quad_pts = similar(pts_hi)
    diff_quad_wts = similar(wts_hi)

    # Match up points
    lo_idx = 1
    @inbounds for hi_idx in eachindex(pts_hi)
        p_hi = pts_hi[sort_hi[hi_idx]]
        diff_quad_pts[hi_idx] = p_hi
        diff_quad_wts[hi_idx] = wts_hi[sort_hi[hi_idx]]
        lo_idx > length(sort_lo) && continue
        p_lo = pts_lo[sort_lo[lo_idx]]
        if p_hi == p_lo
            diff_quad_wts[hi_idx] -= wts_lo[sort_lo[lo_idx]]
            lo_idx += 1
        elseif p_hi > p_lo
            throw(ArgumentError("Points are not nested"))
        end
    end
    diff_quad_pts, diff_quad_wts
end

function getPointEval!(dict, pt, fcn)
    pt_eval = get(dict, pt, nothing)
    if isnothing(pt_eval)
        pt_eval = fcn(pt)
        dict[pt] = pt_eval
    end
    pt_eval
end

function differenceDiagnostic!(eval_dict, rules, midx::SVector{d, Int}, fcn) where {d}
    difference_rules_1d = [formDifference1dRule(r, idx) for (r, idx) in zip(rules, midx)]
    pts, wts = tensor_prod_quad(difference_rules_1d, Val{d}())
    pts_evals = map(pt -> getPointEval!(eval_dict, pt, fcn), pts)
    sum(abs(p) * w for (p, w) in zip(pts_evals, wts)) / length(pts)
end

struct AdaptiveSparseGrid{d, DO, FO, R, D, T, V, C}
    base_mset::MultiIndexSet{d}
    diagnostic_output_type::Type{DO}
    function_output_type::Type{FO}
    quad_rules::R
    diagnostic::D
    tol::Float64
    is_valid_midx::V
    directions::T
    diagnostic_comparison::C
    max_fcn_evals::Int
    neg_tol::Float64
end

function AdaptiveSparseGrid(base_mset::MultiIndexSet{d}, quad_rules::Tuple;
        diagnostic = differenceDiagnostic!, tol = 1e-6, diagnostic_output_type::Type = Float64,
        function_output_type::Type = Float64,
        is_valid_midx = Returns(true), directions = Returns(true),
        diagnostic_comparison = >, max_fcn_evals = typemax(Int),
        neg_tol = -1e-6) where {d}
    @assert d==length(quad_rules) "Expected $d quadrature rules, got $(length(quad_rules))"
    neg_tol < 0 || throw(ArgumentError("Expected negative neg_tol, got $neg_tol"))

    AdaptiveSparseGrid(
        base_mset, diagnostic_output_type, function_output_type, quad_rules, diagnostic,
        tol, is_valid_midx, directions, diagnostic_comparison, max_fcn_evals, neg_tol)
end

function adaptiveIntegrate!(asg::AdaptiveSparseGrid{d, T, U}, fcn; verbose::Bool=false) where {d, T, U}
    eval_dict = Dict{Vector{Float64}, U}()

    mset = asg.base_mset
    rules = asg.quad_rules
    is_valid_midx = asg.is_valid_midx

    active_midxs = MultiIndexing.findReducedFrontier(mset)
    L = Tuple{SVector{d, Int}, T}
    sorted_midxs = SortedList{L}((x,y)->asg.diagnostic_comparison(x[2],y[2]))
    for idx in active_midxs
        midx = mset[idx]
        !is_valid_midx(midx) && continue
        midx_diag = asg.diagnostic(eval_dict, rules, midx, fcn)
        # Since the quadrature rule can have negative weights but integrand is positive
        # enforce that difference diagnostic is very large when est. integral is below
        # some floor
        midx_diag < asg.neg_tol && (midx_diag = Inf)
        push!(sorted_midxs, (midx, midx_diag))
    end

    # While we still have valid midxs, have more evaluations left, and remaining valid midxs have nonnegligible diagnostic
    while !isempty(sorted_midxs) && length(eval_dict) <= asg.max_fcn_evals && peek(sorted_midxs)[2] > asg.tol
        add_midx, diagnostic_eval = pop!(sorted_midxs)
        verbose && @info "Maximum diagnostic value on frontier from multi-index $add_midx, value: $diagnostic_eval"
        tmp_midx = collect(add_midx)
        # Iterate over forward neighbors of add_midx, adding ones that are valid and in the reduced margin
        for j in 1:d
            # If the approximate error doesn't allow direction j, skip
            asg.directions(diagnostic_eval, j) || continue
            tmp_midx[j] += 1
            midx_j = SVector{d}(tmp_midx)
            # If midx_j is not valid according to criterion, skip
            is_midx_j_valid = is_valid_midx(midx_j)
            # push! will return false and not include if midx_j is not in mset reduced margin
            if is_midx_j_valid && push!(mset, midx_j)
                midx_j_diag = asg.diagnostic(eval_dict, rules, midx_j, fcn)
                midx_j_diag < asg.neg_tol && (midx_j_diag = Inf)
                
                length(eval_dict) > asg.max_fcn_evals && break

                verbose && @info "Adding multi-index $midx_j"
                push!(sorted_midxs, (midx_j, midx_j_diag))
            end
            tmp_midx[j] -= 1
        end
    end
    if verbose
        if isempty(sorted_midxs)
            @info "Ran out of multi-indices to add"
        elseif length(eval_dict) > asg.max_fcn_evals
            @info "Reached max function evaluations: $(length(eval_dict))"
        elseif peek(sorted_midxs)[2] < asg.tol
            max_diag = peek(sorted_midxs)[2]
            @info "Reached tolerance with maximum diagnostic value $max_diag"
        else
            @warn "Unknown reason for loop termination"
        end
    end
    # Use adapted mset for quadrature
    final_pts, final_wts = SmolyakQuadrature(mset, rules)

    # Use previous evaluations to perform final quadrature
    res = zero(asg.function_output_type)
    for j in axes(final_pts,2)
        eval_j = get(eval_dict, final_pts[:,j], nothing)
        @assert !isnothing(eval_j) "Found point that fcn was not previously evaluated at"
        res += eval_j*final_wts[j]
    end
    res, eval_dict, final_pts, final_wts
end