include("utils.jl")

module MNIST
    IMAGES_MAGIC = 0x00000803
    LABELS_MAGIC = 0x00000801

    function read_magic!(io, magic_number)
        if (value = bswap(read(io, UInt32))) != magic_number
            throw(ErrorException("""[MNIST] Invalid magic number
            , expected $value, got $magic_number"""))
        end
    end

    function read_images_file(images_file; scale=false, center=false)
        read_header!(io) = map(_ -> Int(bswap(read(io, UInt32))), 1:3)
        read_image!(io, nrows, ncols) = read(io, UInt8, nrows * ncols)

        io = open(images_file)
        read_magic!(io, IMAGES_MAGIC)

        num_images, nrows, ncols = read_header!(io)
        images = map(_ -> read_image!(io, nrows, ncols), 1:num_images)
        images = hcat(images...)'
        close(io)

        if center
            μ = mean(images)
            σ = std(images)
            images = (images .- μ) ./ σ
        end

        images
    end

    function read_labels_file(labels_file)
        read_header!(io) = Int(bswap(read(io, UInt32)))
        read_label!(io) = Int(read(io, UInt8))

        io = open(labels_file)
        read_magic!(io, LABELS_MAGIC)

        num_labels = read_header!(io)
        labels = map(_ -> read_label!(io), 1:num_labels)
        close(io)

        labels
    end

    module test
        import MNIST
        import Utils
        DEFAULT_TEST_IMAGES_PATH = "data/t10k-images-idx3-ubyte"
        DEFAULT_TEST_LABELS_PATH = "data/t10k-labels-idx1-ubyte"

        images(;scale=false, center=false) = MNIST.read_images_file(
            DEFAULT_TEST_IMAGES_PATH; scale=scale, center=center)

        function labels(;one_hot=false)
            labels = MNIST.read_labels_file(DEFAULT_TEST_LABELS_PATH)
            one_hot ? Utils.one_hot(labels) : labels
        end
    end

    module train
        import MNIST
        import Utils
        DEFAULT_TRAIN_IMAGES_PATH = "data/train-images-idx3-ubyte"
        DEFAULT_TRAIN_LABELS_PATH = "data/train-labels-idx1-ubyte"

        images(;scale=false, center=false) = MNIST.read_images_file(
            DEFAULT_TRAIN_IMAGES_PATH; scale=scale, center=center)
        function labels(;one_hot=false)
            labels = MNIST.read_labels_file(DEFAULT_TRAIN_LABELS_PATH)
            one_hot ? Utils.one_hot(labels) : labels
        end
    end
end

# MNIST.train.images(;center=true)
# l = MNIST.test.labels(one_hot=true)
# get_images
# get_labels


# io = open("data/train-images-idx3-ubyte", "r")
# seek(io, 0)
# magic = bswap(read(io, UInt32))
# num_images = bswap(read(io, UInt32))
# nrows = bswap(read(io, UInt32))
# ncols = bswap(read(io, UInt32))
#
# num_images, nrows, ncols = Int(num_images), Int(nrows), Int(ncols)
# image = read(io, UInt8, (nrows, ncols))
# ImageView.imshow(~image)
