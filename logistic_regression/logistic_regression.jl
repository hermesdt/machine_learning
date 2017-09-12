include("../mnist.jl")
include("../utils.jl")

MAX_TRAIN = 1000
MAX_TEST = 200

train_images, train_labels = MNIST.train.images(;center=true)[1:MAX_TRAIN,:], MNIST.train.labels()[1:MAX_TRAIN,:]
test_images, test_labels = MNIST.test.images(;center=true)[1:MAX_TEST,:], MNIST.test.labels()[1:MAX_TEST,:]

intercept = ones(size(train_images)[1])
train_images = hcat(train_images, intercept)
intercept = ones(size(test_images)[1])
test_images = hcat(test_images, intercept)

thetas = Dict()
for digit in sort(unique(train_labels))
    X = train_images
    y = Utils.convert_y_to_binary(train_labels, digit)
    y_test = Utils.convert_y_to_binary(test_labels, digit)
    θ = ones(size(train_images)[2])

    thetas[digit] = Utils.maximize_stochastic(
        Utils.logistic_regression.error,
        Utils.logistic_regression.gradient,
        X,
        y,
        θ; alpha_0 = 0.000001
    )
end

test_labels[1]



# saber hacer softmax
# softmax calcula las probabilidades de cada clase
# cross entropy calcula el error dado el resultado de softmax
# el error total es la suma de cada error divido por todos los elementos
#
# para calcular el gradiantes hay que obtener la derivada de cross entropy
# y aplicarlo a los pesos iniciales y cambiarlos
