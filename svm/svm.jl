"""
Implementation of SVM using SMO
"""

X = zeros(1000, 734)
a = zeros(1000, 1)
y = zeros(1000, 1)

H(x, w, b) = sign(w' * X' + b) # should convert to int?
error(x, y, w, b) = H(x, w, b) - y

C = ??
ϵ = ??

Kernel(x1, x2) = x1*x2' # linear

while true
    given {
        i, j
        a1, a2
        x1, x2
        y1, y2
    }

    w = sum(a*y*x)
    b = mean(y - w*x)


    K_x1_x2 = Kernel(x1, x1) + Kernel(x2, x2) -2Kernel(x1, x2)

    E2, E1 = error(x2, y2, w, b), error(x1, y1, w, b)
    a2_new = a2 + y2(E2 - E1)/K_x1_x2

    if y_i != y_j
        L, H = (maximum(0, a2 - a1), minimum(C, C - a1 + a2))
    else
        L, H = (maximum(0, a1 + a2 - C), minimum(C, a1 + a2))
    end

    a2_new = maximum(a2_new, L) |> minimum(H)


    s = y1*y2
    a1_new + s*a2_new = a1 + s*a2
end
