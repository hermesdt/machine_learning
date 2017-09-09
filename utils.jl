module Utils

    module linear_regression
        error(X, y, θ) = mean((y .- (X .* θ)) .^ 2)
        gradient(X, y, θ) = -2 * sum(y .- (X .* θ), 1)
    end

    module logistic_regression
        import Utils

        function error(X, y, θ)
            error = -y.*log.(Utils.sigmoid(X, θ)) .- (1 .- y).*log.(1 - Utils.sigmoid(X, θ))
            mean(error)
        end

        function gradient(X, y, θ)
            sum((Utils.sigmoid(X, θ) .- y).*X, 1)'
        end
    end

function softmax(x)
    shiftx = x - maximum(x)
    (e.^shiftx) / (sum(e.^shiftx))
end

function sigmoid(X, θ)
    1 ./ (1 .+ e.^(-X*θ))
end

function one_hot(x)
    labels = x |> unique |> sort
    vector = [(labels .== n) * 1 for n in x]
    m = vcat(vector...)
    reshape(m, length(x), length(labels))
end

function accuracy(predicted, real)
    s = (predicted .== real) |> sum
    s / length(real)
end

function minimize_stochastic(error_fn, gradient_fn, X, y, theta; alpha_0=0.001)
    min_error = Inf
    min_theta = 0
    iterations_no_improvement = 0
    alpha = alpha_0
    max_error = 0.003

    while iterations_no_improvement < 100
        error = error_fn(X, y, theta)
        error <= max_error ? break : 1

        if error < min_error
            min_error, min_theta = error, theta
            iterations_no_improvement = 0
            alpha = alpha_0
        else
            iterations_no_improvement += 1
            alpha *= 0.9
        end

        g = gradient_fn(X, y, theta)
        theta = theta - alpha * g
    end

    min_theta
end

function negate(fn)
    (args...; kwargs...) -> -fn(args...; kwargs...)
end

function maximize_stochastic(error_fn, gradient_fn, X, y, theta; alpha_0=0.001)
    minimize_stochastic(negate(error_fn), negate(gradient_fn), X, y, θ; alpha_0=alpha_0)
end

error_rate(predicted, real) = 1 - accuracy(predicted, real)

function count_labels(labels)
    d = Dict()
    for label in labels
        count = get!(d, label) do
            0
        end

        m = Dict(label => count + 1)
        d = merge!(d, m)
    end
    d
end

end

# α = 0.01
# println("theta: $θ")
# println("error: $(error_fn(X, y, θ))")
# g = gradient_fn(X, y, θ)
# println("gradient: $g, α: $α")
# θ = θ - g*α
# println("new θ: $θ")

# X = [1 0 1; 0 1 0; 1 1 0; 0 0 1]
# y = [1; 0; 0; 1]
# θ = zeros(size(X)[2], 1)
#
# Utils.minimize_stochastic(
#     Utils.logistic_regression.error,
#     Utils.logistic_regression.gradient,
#     X, y, θ; alpha_0=0.1) |> print
#
# X = [0 1 0; 0 2 0; 1 1 0; 0 0 3]
# y = [0 2 0; 0 4 0; 1 2 0; 0 0 9]
# θ = [0 0 0]
#
# Utils.minimize_stochastic(
#     Utils.linear_regression.error,
#     Utils.linear_regression.gradient,
#     X, y, θ; alpha_0=0.01) |> print

# Utils.minimize_stochastic()

# Utils.accuracy([1 2 1],[1 3 1])
# Utils.one_hot([1, 2, 3, 1, 3, 9, 5, 6, 1])
# Utils.softmax([1000, 1300, 1000]) |> print
