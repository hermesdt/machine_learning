module Utils

    module linear_regression
        error(X, y, θ) = mean((y .- (X .* θ)) .^ 2)
        gradient(X, y, θ) = -2 * sum(y .- (X .* θ), 1)
    end

    module logistic_regression
        import Utils

        function error(X, y, θ)
            m = length(y)
            htheta = Utils.sigmoid(X*θ)
            1 / m * sum(-y .* log.(htheta) - (1 - y) .* log.(1 - htheta))
        end

        function gradient(X, y, θ)
            m = length(y)
            htheta = Utils.sigmoid(X*θ)
            1 / m * sum(X' * (y - Utils.sigmoid(X*θ)), 2)
        end

        function predict(X, θ)
            prediction = Utils.sigmoid(X*θ)
            prediction[prediction .>= 0.5] = 1
            prediction[prediction .< 0.5] = 0
            prediction
        end
    end

function softmax(x)
    shiftx = x - maximum(x)
    (e.^shiftx) / (sum(e.^shiftx))
end

function sigmoid(z)
    # [i >= 0 ? ) : exp.(i) ./ (1 .+ exp.(i)) for i in z]
    1 ./ (1 .+ exp.(-z))
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

function minimize_stochastic(error_fn, gradient_fn, X, y, theta;
        alpha=0.001, max_iterations = 100)
    iterations = 0

    while iterations < max_iterations
        error = error_fn(X, y, theta)

        if iterations % 20 == 0
            println("iterations: $iterations, error: $error")
        end

        for i in 1:size(X)[1]
            theta = theta - alpha*(gradient_fn(X[i:i,:], y[i], theta))
        end

        iterations += 1
    end

    theta
end

function negate(fn)
    (args...; kwargs...) -> -fn(args...; kwargs...)
end

function maximize_stochastic(error_fn, gradient_fn, X, y, theta;
    alpha=0.01, max_iterations=100)
    minimize_stochastic(error_fn, negate(gradient_fn), X, y, theta;
        alpha=alpha, max_iterations=max_iterations)
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

function convert_y_to_binary(y, label)
    y = copy(y)
    temp = maximum(y) + 1
    y[y .== label] = temp
    y[(y .!= label) .& (y .!= temp)] = 0
    y[y .== temp] = 1
    y
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
