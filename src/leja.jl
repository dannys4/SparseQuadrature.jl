export leja01, leja01_nested

function EvaluateLegendrePolynomial(maxDegree, pt::T) where {T}

    # Initialize an array to store the values of the Legendre polynomials
    P = zeros(T, maxDegree + 1)

    # Base cases
    P[1] = 1  # P_0(x) = 1
    if maxDegree >= 1
        P[2] = pt # P_1(x) = x
    end

    # Use the three-term recurrence relation to compute higher degrees
    for n in 1:maxDegree-1
        P[n+2] = ((2n + 1) * pt * P[n+1] - n * P[n]) / (n + 1)
    end

    return P
end

function TrueIntegralLegendre(n)
    n == 1 && return [1.0]
    ret = zeros(n)
    ret[1] = 1.0
    ret
end

LegendreSquareNorm(degree) = 1 / (2degree + 1)

module FindLeja
# This module is included for reproducibility purposes. However, to minimize dependencies, we use a serialized version of the points
# and create the weights on the fly
# The default optimization functions require the dependencies: Optimization, OptimizationOptimJL, ForwardDiff
# TODO: Add this as a package extension

# An okay approximation to the asymptotic true christoffel weighting
# Includes log-barrier on {-1,1} to get first two nonzero points inside
# domain
function LogChristoffelSurrogate(x, _)
    abs(abs(x) - 1) < 100eps() && return 0.0
    -0.5log((1 / sqrt(2) + 0.1log(1 - x^2))^2 + 0.2)
end

# True inverse christoffel weighting for nth leja point on U(-1,1)
function LogChristoffelTrue(x, n, normalize=true)
    evals = EvaluateLegendrePolynomial(n - 1, x)
    christoffel = 0.0
    if normalize
        christoffel = sum(evals[j]^2 / LegendreSquareNorm(j - 1) for j in eachindex(evals))
    else
        christoffel = sum(x -> x^2, evals)
    end
    -0.5log(christoffel)
end

# Given previous leja points and density function, this is the 
function LejaLoss(x, p)
    logdensity, prev_leja = p
    log_diffs = sum(L -> log(abs(L - x[])), prev_leja)
    log_inv_sqrt_christoffel = logdensity(x[], length(prev_leja))
    -(log_diffs + log_inv_sqrt_christoffel)
end

function ExampleLejaOptimizationSetup_ForwardDiff(fcn)
    OptimizationFunction(fcn, Optimization.AutoForwardDiff())
end

function ExampleLejaConstrainedOptimizer_OptimizationOptimJL(fcn, u0, params, lb, ub)
    # Problem setup
    prob = OptimizationProblem(fcn, [u0], params, lb=[lb], ub=[ub])
    reltol = 1000eps()
    sol = solve(prob, LBFGS(); reltol)
    bound_tol = 10reltol
    if sol.u[] > lb + bound_tol && sol.u[] < ub - bound_tol
        local_prob = OptimizationProblem(fcn, sol.u, params)
        sol = solve(local_prob, Newton(), reltol=100eps())
    end
    sol.u[], sol.objective[]
end

function FindLejaPoints1d(N, constrained_minimizer=ExampleLejaConstrainedOptimizer_OptimizationOptimJL,
    minimization_setup=ExampleLejaOptimizationSetup_ForwardDiff,
    first_leja_pts=Float64[0.0, 0.8359083678182149, -0.8809549190413212],
    leja_logdensity=LogChristoffelTrue)

    leja_pts = zeros(N)
    leja_pts[1:length(first_leja_pts)] .= first_leja_pts
    start = 1 + length(first_leja_pts)

    loss_fcn = minimization_setup(LejaLoss)

    for j in start:N
        leja_pts_sorted = sort(leja_pts[1:j-1])
        optim_j, optim_j_obj = 0.0, Inf
        # Check between all the poles in the process
        for k in 1:j
            # Create the box bounding the poles
            lb = k == 1 ? -1.0 : leja_pts_sorted[k-1]
            ub = k == j ? 1.0 : leja_pts_sorted[k]
            u0 = (ub + lb) / 2

            # Check that we aren't starting too close to the domain's edge
            criteria = 1 - u0^2 > 10eps()
            !criteria && continue

            # Find the best point using given minimizer
            params = (leja_logdensity, @view(leja_pts[1:j-1]))
            u_jk, objective_jk = constrained_minimizer(loss_fcn, u0, params, lb, ub)

            # If this u0 gives a better minimizer, use it
            if objective_jk < optim_j_obj
                optim_j = u_jk
                optim_j_obj = objective_jk
            end
        end
        @info "" j optim_j optim_j_obj
        leja_pts[j] = optim_j
    end
    leja_pts
end

end

__UNIFORMLEJA = open(deserialize, joinpath(@__DIR__, "serial", "uniformleja.ser"), "r")

function leja01(n)
    n > length(__UNIFORMLEJA) && throw(ArgumentError("Expected n < $(length(__UNIFORMLEJA)), got n=$n"))
    pts = __UNIFORMLEJA[1:n]
    legendre_poly_offset = (N, x) -> EvaluateLegendrePolynomial(N - 1, x)
    wts = CreateQuadratureWeights(pts, legendre_poly_offset, TrueIntegralLegendre)
    # Adjust pts for domain
    @. pts = (pts + 1) / 2
    pts, wts
end

# Skips two-point rule, which is identical to level 0 due to weight construction
leja01_nested(level) = leja01(level + 1 + (level > 0))