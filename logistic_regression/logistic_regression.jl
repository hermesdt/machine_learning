include("../mnist.jl")
include("../utils.jl")

MAX_TRAIN = 5000
MAX_TEST = 200

# saber hacer softmax
# softmax calcula las probabilidades de cada clase
# cross entropy calcula el error dado el resultado de softmax
# el error total es la suma de cada error divido por todos los elementos
#
# para calcular el gradiantes hay que obtener la derivada de cross entropy
# y aplicarlo a los pesos iniciales y cambiarlos

X = [1 0 1; 0 1 0; 1 1 0; 0 0 1]
y = [1; 0; 0; 1]
θ = zeros(size(X)[2], 1)

# θ = Utils.minimize_stochastic(
#     Utils.logistic_regression.error,
#     Utils.logistic_regression.gradient,
#     X, y, θ; alpha_0=0.1)


error = Utils.logistic_regression.error(X, y, θ)
gradient = Utils.logistic_regression.gradient(X, y, θ)
println("error: $error")
println("gradient: $gradient")
for i in 1:1000
    gradient = Utils.logistic_regression.gradient(X, y, θ)
    θ = θ - 0.01 * gradient
end

error = Utils.logistic_regression.error(X, y, θ)
sigmoid = Utils.sigmoid(X, θ)

println("theta: $θ")
println("error: $error")
println("sigmoid: $sigmoid")

predicted = Utils.sigmoid(X, θ)
predicted[predicted .>= 0.5] = 1
predicted[predicted .< 0.5] = 0
# println("accuracy: $(Utils.accuracy(predicted, y))")
