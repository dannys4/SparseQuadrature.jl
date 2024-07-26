using SparseQuadrature: tensor_prod_quad, formDifference1dRule

test_fcn_1d = x -> sin(0.4pi * cos(0.4pi * x[]))^2 / (x[] + 1)
for quad_rule in (clenshawcurtis01_nested, gausspatterson01_nested, leja01_nested)

    @testset "Difference Operators" begin
        order = 5
        @testset "One-dimensional function" begin
            diff_quad_pts, diff_quad_wts = formDifference1dRule(quad_rule, order)
            pts_hi, wts_hi = quad_rule(order)
            pts_lo, wts_lo = quad_rule(order - 1)
            test_fcn_diff_true = test_fcn_1d.(pts_hi)'wts_hi - test_fcn_1d.(pts_lo)'wts_lo
            test_fcn_diff_est = test_fcn_1d.(diff_quad_pts)'diff_quad_wts
            @test test_fcn_diff_true ≈ test_fcn_diff_est atol = 1e-10

            eval_count = 0
            mset_1d = CreateTotalOrder(1, order)
            function test_fcn_counter(pt)
                eval_count += 1
                test_fcn_1d(pt[])
            end
            eval_dict = Dict()
            rules = (quad_rule,)
            midx = @SVector[order]
            diff = SparseQuadrature.differenceDiagnostic!(eval_dict, rules, midx, test_fcn_counter)
            @test eval_count == length(pts_hi)
            @test diff ≈ test_fcn_diff_est / length(pts_hi) atol = 1e-10
        end
        @testset "two-dimensional function" begin
            eval_counter = 0
            function test_fcn_2d(x)
                eval_counter += 1
                (test_fcn_1d(x[1])^2 + 1) * test_fcn_1d(x[2])
            end
            orders = [3, 4]
            midx_test = SVector((orders...))

            # Calculate exact difference based on true formula
            U1_hi, U2_hi = quad_rule.(orders)
            U1_lo, U2_lo = quad_rule.(orders .- 1)
            TPQ = SparseQuadrature.tensor_prod_quad
            pts_hi_hi, wts_hi_hi = TPQ((U1_hi, U2_hi), Val{2}())
            pts_hi_lo, wts_hi_lo = TPQ((U1_hi, U2_lo), Val{2}())
            pts_lo_hi, wts_lo_hi = TPQ((U1_lo, U2_hi), Val{2}())
            pts_lo_lo, wts_lo_lo = TPQ((U1_lo, U2_lo), Val{2}())
            diff_true = test_fcn_2d.(pts_hi_hi)'wts_hi_hi
            diff_true -= test_fcn_2d.(pts_hi_lo)'wts_hi_lo
            diff_true -= test_fcn_2d.(pts_lo_hi)'wts_lo_hi
            diff_true += test_fcn_2d.(pts_lo_lo)'wts_lo_lo

            # Use MultiIndexing method
            eval_dict = Dict()
            eval_counter = 0
            rules = (quad_rule, quad_rule)
            diff_approx = SparseQuadrature.differenceDiagnostic!(eval_dict, rules, midx_test, test_fcn_2d)
            expected_evals = length(pts_hi_hi)
            @test eval_counter == expected_evals
            @test diff_true / length(pts_hi_hi) ≈ diff_approx atol = 1e-10
        end
    end

    @testset "AdaptiveSparseGrid" begin
        is_GP = quad_rule == gausspatterson01_nested
        @testset "One dimensional" begin
            mset_base = CreateTotalOrder(1, 1)
            num_evals = 0
            function test_fcn_1d_poly(x)
                num_evals += 1
                x[]^2 + x[] + x[]^10
            end
            test_fcn_int = (1 / 3) + (1 / 2) + (1 / 11)
            asg = AdaptiveSparseGrid(mset_base, (quad_rule,), tol=5eps())
            result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_1d_poly; verbose=false)
            sp = sortperm(final_pts[:])
            final_pts = final_pts[sp]
            final_wts = final_wts[sp]
            @test result ≈ test_fcn_int atol = 1e-10

            # Make sure that the level doesn't go more than 1 above
            # level required to integrate fcn _exactly_
            needed_level = expected_evals = 0
            if quad_rule == clenshawcurtis01_nested
                needed_level = ceil(Int, log2(10))
                expected_evals = 2^(needed_level + 1) + 1
            elseif quad_rule == gausspatterson01_nested
                needed_level = ceil(Int, log2((10+2)÷2))
                expected_evals = 2^(needed_level + 1) - 1
            elseif quad_rule == leja01_nested
                needed_level = 9
                expected_evals = needed_level+3
            else
                throw(InvalidStateException("Invalid quadrature rule $quad_rule"))
            end

            @test mset_base.maxDegrees[1] >= needed_level
            # Make sure we keep track of every evaluation
            @test length(eval_dict) == num_evals
            @test length(eval_dict) == expected_evals

            expect_pts, expect_wts = quad_rule(needed_level + Int(!is_GP))
            sortperm_expect = sortperm(expect_pts)
            permute!(expect_pts, sortperm_expect)
            permute!(expect_wts, sortperm_expect)
            @test all(isapprox.(expect_pts, final_pts, atol=1e-10))
            @test all(isapprox.(expect_wts, final_wts, atol=1e-10))
        end
        @testset "Two dimensional" begin
            mset_base = CreateTotalOrder(2, 1)
            num_evals = 0
            function test_fcn_2d_poly(x)
                num_evals += 1
                x[1]^11 + x[2]^8 + x[1] * x[2]
            end
            test_fcn_2d_int = (1 / 12) + (1 / 9) + (1 / 4)
            # Limit interaction indices to be total order < 3
            # In 2d this means [1,1] can be the only interaction midx
            function is_valid_midx(midx)
                sum(midx .> 0) <= 1 && return true
                sum(midx) < 3
            end
            asg = AdaptiveSparseGrid(mset_base, (quad_rule, quad_rule), tol=5eps(), neg_tol=-eps(); is_valid_midx)
            result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_2d_poly; verbose=false)

            @test length(eval_dict) == num_evals
            @test result ≈ test_fcn_2d_int atol = 1e-10
            # Required level for integrating each monomial in fcn
            expected_maxDegrees = -1
            if quad_rule == gausspatterson01_nested
                expected_maxDegrees = [3,3]
            elseif quad_rule == clenshawcurtis01_nested
                expected_maxDegrees = [5,4]
            elseif quad_rule == leja01_nested
                expected_maxDegrees = [11, 8]
            else
                throw(InvalidStateException("Unexpected quadrature rule $quad_rule"))
            end
            @test all(mset_base.maxDegrees .== expected_maxDegrees)

            # Check the rule it returns is actually the rule associated with the new mset
            sortperm_test = sortperm(collect(eachcol(final_pts)))
            final_pts = final_pts[:, sortperm_test]
            final_wts = final_wts[sortperm_test]

            ref_pts, ref_wts = SmolyakQuadrature(mset_base, quad_rule)
            sortperm_ref = sortperm(collect(eachcol(ref_pts)))
            ref_pts = ref_pts[:, sortperm_ref]
            ref_wts = ref_wts[sortperm_ref]
            @test all(isapprox.(ref_pts, final_pts, atol=1e-10))
            @test all(isapprox.(ref_wts, final_wts, atol=1e-10))
        end
    end

    @testset "High dimensional" begin
        dim = 10
        pow = 16
        mset_base = CreateTotalOrder(dim, 1)
        num_evals = 0
        function test_fcn_nd_poly(x)
            num_evals += 1
            sum(x -> x^pow, x) + prod(x)
        end
        function is_valid_midx(midx)
            num_nz = sum(midx .> 0)
            # valid if one nonzero or total order < dim+1
            num_nz < 2 || sum(midx) < dim + 1
        end
        test_fcn_nd_int = dim / (pow + 1) + 2.0^(-dim)
        asg = AdaptiveSparseGrid(mset_base, ntuple(_ -> quad_rule, dim), tol=5eps(), neg_tol=-0.1eps(); is_valid_midx)
        result, eval_dict, final_pts, final_wts = adaptiveIntegrate!(asg, test_fcn_nd_poly; verbose=false)
        @test result ≈ test_fcn_nd_int atol = 1e-10
        @test length(eval_dict) == num_evals
    end

end