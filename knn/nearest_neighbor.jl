include("../mnist.jl")
include("../utils.jl")

# import MNIST
# import Utils

function classify_images(k, train_images, train_labels, test_images, test_labels)
    predictions = []
    for row in 1:size(test_images)[1]
        image = test_images[row:row,:]
        predicted_label = classify_image(k, train_images, train_labels, image, row)
        append!(predictions, predicted_label)
    end
    predictions
end

function distance_l1(x, y)
    sum(abs.(x .- y), 2) |> vec
end

function classify_image(k, train_images, train_labels, new_image, row)
    distances = distance_l1(train_images, new_image)
    indmin(distances)
end

function majority_vote(labels)
    counts = [(k,v) for (k,v) in Utils.count_labels(labels)]
    label, value = sort(counts, by=(t -> t[2]), rev=true) |> first
    value
end

train_images, train_labels = MNIST.train.images()[1:5000,:], MNIST.train.labels()[1:5000,:]
test_images, test_labels = MNIST.test.images()[1:1000,:], MNIST.test.labels()[1:1000,:]

@time begin
    for k in 1:1
        predictions = classify_images(k, train_images, train_labels,test_images, test_labels)
        accuracy = Utils.accuracy(predictions, test_labels)
        println("accuracy: $accuracy with k = $k")
    end
end

# accuracy: 0.1134 with k = 1
# accuracy: 0.0558 with k = 2
# accuracy: 0.0924 with k = 3
# accuracy: 0.0895 with k = 4
