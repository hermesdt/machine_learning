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

Utils.minimize_stochastic(
    Utils.logistic_regression.error,
    Utils.logistic_regression.gradient,
    X, y, θ; alpha_0=0.1) |> print
