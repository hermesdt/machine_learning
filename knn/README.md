## K Nearest Neighborhood

Type: Superversided
Pros:
    * Really easy to use and implement
    * No need to train the model
Cons:
    * Because of its simplicity it usually gives less accurate results than other algorithms (although in some cases
    may be enough).
    * Be aware of the "curse of dimensionality": the minimum distance between elements increases with the number of
    dimensions
    * As a counter part of needing to trying the model, the time needed to classify a new element is expensive,
    as the algorithm have to compare it with each other.

# Description

Given a list of samples that have to be classified, and a list of elements labeled, iterate over each sample finding
the K nearest elements. Then assign the class with higher amount of elements in it.

In case of match try with K - 1

As we can see in the description we need some way to measure the distance between elements. As I will be using MNIST
images the distance will be L1 distance:

D(a,b) = |a - b|