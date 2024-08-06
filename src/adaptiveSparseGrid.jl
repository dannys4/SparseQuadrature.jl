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
    abs(sum(eval * w for (eval, w) in zip(pts_evals, wts)))
end

struct AdaptiveSparseGrid{d, DO, FO, R, D, V, C}
    base_mset::MultiIndexSet{d}
    diagnostic_output_type::Type{DO}
    function_output_type::Type{FO}
    quad_rules::R
    diagnostic::D
    tol::Float64
    is_valid_midx::V
    is_directional::Bool
    diagnostic_comparison::C
    max_fcn_evals::Int
    max_time::Float64
end

function AdaptiveSparseGrid(base_mset::MultiIndexSet{d}, quad_rules::Tuple;
        diagnostic = differenceDiagnostic!, tol = 1e-6, diagnostic_output_type::Type = Float64,
        function_output_type::Type = Float64,
        is_valid_midx = Returns(true), is_directional = false,
        diagnostic_comparison = >, max_fcn_evals = typemax(Int), max_time=typemax(Float64)) where {d}

    @assert d==length(quad_rules) "Expected $d quadrature rules, got $(length(quad_rules))"

    AdaptiveSparseGrid(
        base_mset, diagnostic_output_type, function_output_type, quad_rules, diagnostic,
        tol, is_valid_midx, is_directional, diagnostic_comparison, max_fcn_evals, max_time)
end

function AdaptiveSparseGrid(base_mset::MultiIndexSet{d}, quad_rules::Function; kwargs...) where {d}
    AdaptiveSparseGrid(base_mset, ntuple(Returns(quad_rules), d); kwargs...)
end

function adaptiveIntegrate!(asg::AdaptiveSparseGrid{d, Diag_T, U}, fcn; verbose::Bool=false) where {d, Diag_T, U}
    eval_dict = Dict{Vector{Float64}, U}()

    mset = asg.base_mset
    rules = asg.quad_rules
    is_valid_midx = asg.is_valid_midx
    is_directional = asg.is_directional

    # TODO: Need entire frontier, not just reduced. No difference with Total-Order msets though.
    active_midxs = MultiIndexing.findReducedFrontier(mset)
    MIdx_T = SVector{d, Int}
    # If directional diagnostic, list elements are (midx, scalar_diagnostic, direction)
    Eltype_T = is_directional ? Tuple{MIdx_T, Diag_T, Int} : Tuple{MIdx_T, Diag_T}
    sorted_midxs = SortedList{Eltype_T}((x,y)->asg.diagnostic_comparison(x[2],y[2]))
    global_error = 0.

    # Add all active indices in starting set to the list
    for idx in active_midxs
        midx = mset[idx]
        !is_valid_midx(midx) && continue
        midx_diag = asg.diagnostic(eval_dict, rules, midx, fcn)

        if is_directional
            for i in 1:d
                push!(sorted_midxs, (midx, midx_diag[i], i))
                global_error += midx_diag[i]
            end
        else
            push!(sorted_midxs, (midx, midx_diag))
            global_error += midx_diag
        end
    end

    start_time = time()

    # While we still have valid midxs, have more evaluations left, and remaining valid midxs have nonnegligible diagnostic
    while !isempty(sorted_midxs) && length(eval_dict) <= asg.max_fcn_evals && global_error > asg.tol && (time() - start_time) < asg.max_time
        next_midx = pop!(sorted_midxs)
        add_midx, diagnostic_eval = next_midx
        global_error -= sum(diagnostic_eval)

        if is_directional
            dir = next_midx[end]
            add_midx_dir = SVector{d}(ntuple(j->add_midx[j] + Int(j==dir), d))
            # If midx_j is not valid according to criterion, skip
            is_midx_valid = is_valid_midx(add_midx_dir)
            # push! will return false and not include if midx is not in mset reduced margin
            if is_midx_valid && push!(mset, add_midx_dir)
                verbose && @info "Maximum diagnostic value on frontier from multi-index $add_midx, value: $diagnostic_eval, direction $dir"
                midx_diag = asg.diagnostic(eval_dict, rules, add_midx_dir, fcn)
                verbose && @info "Adding midx $add_midx_dir, diagnostic values $(Tuple(midx_diag))"
                for j in 1:d
                    global_error += midx_diag[j]
                    push!(sorted_midxs, (add_midx_dir, midx_diag[j], j))
                end
            end
        else # If the diagnostic is not directional
            verbose && @info "Maximum diagnostic value on frontier from multi-index $add_midx, value: $diagnostic_eval"
            tmp_midx = collect(add_midx)
            # Iterate over forward neighbors of add_midx, adding ones that are valid and in the reduced margin
            for j in 1:d
                tmp_midx[j] += 1
                midx_j = SVector{d}(tmp_midx)
                # If midx_j is not valid according to criterion, skip
                is_midx_j_valid = is_valid_midx(midx_j)
                # push! will return false and not include if midx_j is not in mset reduced margin
                if is_midx_j_valid && push!(mset, midx_j)
                    midx_j_diag = asg.diagnostic(eval_dict, rules, midx_j, fcn)
                    global_error += midx_j_diag

                    # Stop evaluating if we hit max time or function evals
                    (time() - start_time) > asg.max_time && break
                    length(eval_dict) > asg.max_fcn_evals && break

                    verbose && @info "Adding multi-index $midx_j, diagnostic $midx_j_diag"
                    push!(sorted_midxs, (midx_j, midx_j_diag))
                end
                tmp_midx[j] -= 1
            end
        end
        verbose && @info "Current global error $global_error"
    end
    # Finished adaptivity, log why we finished
    if verbose
        if time() - start_time > asg.max_time
            @info "Ran out of time"
        elseif isempty(sorted_midxs)
            @info "Ran out of multi-indices to add"
        elseif length(eval_dict) > asg.max_fcn_evals
            @info "Reached max function evaluations: $(length(eval_dict))"
        elseif global_error < asg.tol
            @info "Reached tolerance with global error estimate $global_error"
        else
            @warn "Unknown reason for sparse quadrature adaptation termination"
        end
    end

    # Use adapted mset for quadrature
    final_pts, final_wts = SmolyakQuadrature(mset, rules)

    # Use previous evaluations to perform final quadrature
    res = sum(axes(final_pts,2)) do j
        eval_j = get(eval_dict, final_pts[:,j], nothing)
        @assert !isnothing(eval_j) "Found point that fcn was not previously evaluated at"
        eval_j*final_wts[j]
    end
    res, eval_dict, final_pts, final_wts
end