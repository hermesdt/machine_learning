include("../mnist.jl")
include("../utils.jl")

MAX_TRAIN = 2000
MAX_TEST = 200

train_images, train_labels = MNIST.train.images(;center=true)[1:MAX_TRAIN,:], MNIST.train.labels()[1:MAX_TRAIN,:]
test_images, test_labels = MNIST.test.images(;center=true)[1:MAX_TEST,:], MNIST.test.labels()[1:MAX_TEST,:]

m, n = size(train_images)
X = train_images
A = zeros(m, 1)
w = zeros(n, 1)
Y = zeros(m, 1)
b = 0

Hypotesis(x, w, b) = sign.(w'*x' + b) # should convert to int?
Hypotesis(X[1:1,:], w, b)
ERROR(x, y, w, b) = Hypotesis(x, w, b) - y
ERROR(X[1:1,:], Y[1:1,:], w, b)
C = 1
ϵ = 2

# KERNEL(x1, x2) = x1*x2' # linear
KERNEL(x1, x2) = (x1*x2' + 1) ^ 2 # quadratic
# KERNEL(x1, x2) = exp(-(sum((x1 - x2) .^ 2) / (2 * 1 ^ 2)));

function get_rand_int(a, b, i)
    temp = i
    while temp == i
        temp = rand(a:b)
    end
    temp
end

X = train_images
Y = train_labels

Y = Utils.convert_y_to_binary(train_labels, 5)
Y[Y .== 0] = -1

for i in 1:m
    # i = 1
    j = get_rand_int(1, m, i)[1]

    a1, a2, y1, y2 = A[i], A[j], Y[i], Y[j]
    x1, x2 = X[i:i, :], X[j:j, :]

    w = X' * (A.*Y)
    b = mean(Y - X*w)

    K_x1_x2 = KERNEL(x1, x1) + KERNEL(x2, x2) -2*KERNEL(x1, x2)

    if y1 != y2
        L, H = (maximum([0.0, a2 - a1]), minimum([C, C - a1 + a2]))
    else
        L, H = (maximum([0.0, a1 + a2 - C]), minimum([C, a1 + a2]))
    end

    E1, E2 = ERROR(x1, y1, w, b), ERROR(x2, y2, w, b)

    a2_new = a2 + y2*(E1 - E2)/K_x1_x2
    a2_new = maximum([a2_new[1], L])
    a2_new = minimum([a2_new, H])

    s = y1*y2
    a1_new = a1 + s*(a2 - a2_new)

    A[i] = a1_new[1]
    A[j] = a2_new[1]
end

x = X[1:1, :]
y = Y[1:1]
Y_hat = Hypotesis(X, w, b)'

Utils.accuracy(Y, Y_hat) |> print
