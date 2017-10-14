include("../mnist.jl")
include("../utils.jl")

MAX_TRAIN = 200
MAX_TEST = 200

train_images, train_labels = MNIST.train.images(;center=true)[1:MAX_TRAIN,:], MNIST.train.labels()[1:MAX_TRAIN,:]
test_images, test_labels = MNIST.test.images(;center=true)[1:MAX_TEST,:], MNIST.test.labels()[1:MAX_TEST,:]

intercept = ones(size(train_images)[1])
train_images = hcat(train_images, intercept)
intercept = ones(size(test_images)[1])
test_images = hcat(test_images, intercept)

thetas = []
# train one logistic regression per digit
for digit in sort(unique(train_labels))
    println("doing digit $digit")
    X = train_images
    y = Utils.convert_y_to_binary(train_labels, digit)
    θ = zeros(size(train_images)[2])
    y_test = Utils.convert_y_to_binary(test_labels, digit)

    θ = Utils.maximize_stochastic(
        Utils.logistic_regression.error,
        Utils.logistic_regression.gradient,
        X,
        y,
        θ; alpha = 0.001, max_iterations=50
    )

    push!(thetas, θ)

    predicted = Utils.logistic_regression.predict(test_images, θ)
    accuracy = Utils.accuracy(predicted, y_test)
    println("digit: $digit, accuracy: $accuracy")
end


accuracy = 0
sample_size = length(test_labels)

for index in eachindex(test_labels)
    x = test_images[index:index,:]
    label = test_labels[index]

    predictions = [Utils.logistic_regression.predict(x, theta)[1] for theta in thetas]
    best_prediction = indmax(predictions) - 1
    println("predictions $predictions")
    println("label $label, prediction $best_prediction")

    if label == best_prediction
        accuracy += 1/sample_size
    end
end

println("accuracy: $accuracy")
