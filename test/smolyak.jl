using FastGaussQuadrature

function unifquad01(exactness)
    n = exactness ÷ 2 + 1
    pts, wts = gausslegendre(n)
    @. pts = (pts + 1) / 2
    @. wts = wts / 2
    pts, wts
end

function monomialEval(midx::SVector{d}, x::AbstractMatrix) where {d}
    evals = ones(size(x, 2))
    for i in eachindex(midx)
        evals .*= (x[i, :] .^ midx[i])
    end
    evals
end

@testset "Total Order Quadrature" begin
    dp_vec = [(3, 2), (2, 4), (4, 5), (5, 4), (10, 2)]
    for (d, p) in dp_vec
        mset = CreateTotalOrder(d, p)
        pts, wts = SmolyakQuadrature(mset, unifquad01)
        quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
        exact_int = sum(prod(1 ./ (midx .+ 1)) for midx in mset)
        @test isapprox(quad_int, exact_int, atol=1e-10)
    end
end

@testset "Hyperbolic Quadrature" begin
    for j in 4:8

        mset = create_example_hyperbolic2d(2^j)
        prev_mset = create_example_hyperbolic2d(2^(j - 1))
        # Gaussian quadrature
        pts, wts = SmolyakQuadrature(mset, unifquad01)
        quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
        exact_int = sum(exp(sum(k -> log(1 / (k + 1)), midx)) for midx in mset)
        @test isapprox(quad_int, exact_int, atol=1e-10)
        # Clenshaw-Curtis quadrature
        pts, wts = SmolyakQuadrature(mset, clenshawcurtis01)
        # For N points, exact on N-1 polynomials; therefore, not actually exact on mset
        quad_int = sum(monomialEval(midx, pts) for midx in mset)' * wts
        exact_int = sum(exp(sum(k -> -log(k + 1), midx)) for midx in mset)
        @test isapprox(quad_int, exact_int, atol=1e-10)
    end
end