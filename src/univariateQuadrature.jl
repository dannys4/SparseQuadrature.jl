export clenshawcurtis01, clenshawcurtis01_nested, gausspatterson01, gausspatterson01_nested

# Inspired by implementation in ChaosPy: https://github.com/jonathf/chaospy/blob/53000bbb04f8d3f9908ebbf1be6bf139a21c2e6e/chaospy/quadrature/clenshaw_curtis.py#L76
function clenshawcurtis01(order::Int)
    if order == 0
        return [0.5], [1.0]
    elseif order == 1
        return [0.000000000000000000000, 1.0], [0.5, 0.5]
    end

    theta = (order .- (0:order)) .* π / order
    abscissas = 0.5 .* cos.(theta) .+ 0.5

    steps = 1:2:(order - 1)
    L = length(steps)
    remains = order - L

    beta = vcat(2.0 ./ (steps .* (steps .- 2)), [1.0 / steps[end]], zeros(remains))
    beta = -beta[1:(end - 1)] .- reverse(beta[2:end])

    gamma = -ones(order)
    gamma[L + 1] += order
    gamma[remains + 1] += order
    gamma ./= order^2 - 1 + (order % 2)

    weights = rfft(beta + gamma) / order
    @assert maximum(imag.(weights)) < 1e-15
    weights = real.(weights)
    weights = vcat(weights, reverse(weights)[(2 - (order % 2)):end]) ./ 2

    return abscissas, weights
end

__GAUSSPATTERSON = open(deserialize, joinpath(@__DIR__,"serial","gausspatterson.ser"), "r")

"""
    gausspatterson01(n)
Nested gausspatterson rule adapted from John Burkardt's implementation.

n must be 1, 3, 7, 15, 31, 63, 127, 255 or 511.

See [here](https://people.math.sc.edu/Burkardt/m_src/patterson_rule/patterson_rule.html) for more information
"""
function gausspatterson01(n::Int)
    j = floor(Int, log2(n+1))
    @assert n == 2^j - 1 "n=$n, expected n in [1, 3, 7, 15, 31, 63, 127, 255 or 511]"
    __GAUSSPATTERSON[j]
end

clenshawcurtis01_nested(j) = clenshawcurtis01(j == 0 ? 0 : 2^j)
gausspatterson01_nested(j) = gausspatterson01(2^(j+1) - 1)