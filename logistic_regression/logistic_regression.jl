include("../mnist.jl")
include("../utils.jl")

MAX_TRAIN = 1000
MAX_TEST = 100

train_images, train_labels = MNIST.train.images(;center=true)[1:MAX_TRAIN,:], MNIST.train.labels()[1:MAX_TRAIN,:]
test_images, test_labels = MNIST.test.images(;center=true)[1:MAX_TEST,:], MNIST.test.labels()[1:MAX_TEST,:]

intercept = ones(size(train_images)[1])
train_images = hcat(train_images, intercept)
intercept = ones(size(test_images)[1])
test_images = hcat(test_images, intercept)

thetas = Dict()
predicted_temp, predicted = 0, 0
# for digit in sort(unique(train_labels))
for digit in [5, 1, 2, 9]
    println("doing digit $digit")
    X = train_images
    y = Utils.convert_y_to_binary(train_labels, digit)
    θ = zeros(size(train_images)[2])
    y_test = Utils.convert_y_to_binary(test_labels, digit)

    thetas[digit] = Utils.maximize_stochastic(
        Utils.logistic_regression.error,
        Utils.logistic_regression.gradient,
        X,
        y,
        θ; alpha = 0.01, max_iterations=100
    )

    if false
    for i in 1:100
        for i in 1:size(X)[1]
            θ = θ + 0.01*(Utils.logistic_regression.gradient(X[i:i,:], y[i], θ))
        end

        error = Utils.logistic_regression.error(X, y, θ)
    	predicted = Utils.sigmoid(test_images*θ)
        predicted_temp = predicted
    	predicted[predicted .>= 0.5] = 1
    	predicted[predicted .< 0.5] = 0
        accuracy = Utils.accuracy(predicted, y_test)
        println("error: $error, accuracy: $accuracy")
    end
    thetas[digit] = θ
    end

    predicted = Utils.logistic_regression.predict(test_images, thetas[digit])
    accuracy = Utils.accuracy(predicted, y_test)
    println("digit: $digit, accuracy: $accuracy")
end




# saber hacer softmax
# softmax calcula las probabilidades de cada clase
# cross entropy calcula el error dado el resultado de softmax
# el error total es la suma de cada error divido por todos los elementos
#
# para calcular el gradiantes hay que obtener la derivada de cross entropy
# y aplicarlo a los pesos iniciales y cambiarlos
