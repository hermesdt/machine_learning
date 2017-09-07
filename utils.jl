module Utils

function accuracy(predicted, real)
    s = (predicted .== real) |> sum
    s / length(real)
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

end

Utils.accuracy([1 2 1],[1 3 1])
