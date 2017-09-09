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

train_images, train_labels = MNIST.test.images(), MNIST.test.labels()
test_images, test_labels = MNIST.test.images(), MNIST.test.labels()

train_images = train_images[1:MAX_TRAIN,:]
train_labels = train_labels[1:MAX_TRAIN,:]
test_images = test_images[1:MAX_TEST,:]
test_labels = test_labels[1:MAX_TEST,:]
