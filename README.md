# CSC311-Project
咔咔的第三题

## a)

The purpose of ALS is matrix factorization for reconstruction - that is, factorize a matrix into two matrices U and Z, then combine them. On the other hand, the autoencoder first encodes the input by imposing a bottleneck that forces a compressed knowledge representation, then decodes it to get back the original data.

In ALS, we minimize the loss with respect to two factors, U and Z. However, we minimize with respect to only one factor, which is the weight matrix W, in autoencoder.

We use gradient decent to update W at each iteration in autoencoder until convergence. In the case of ALS, the objective is non-convex in U and Z jointly, but is convex as a function of either U or Z individually. Therefore, we fix Z and optimize U, then fix U and optimize Z, so on until convergence. 


## b)

Code in neural_network.py.


## c)

Plot for k = 10, learning rate = 0.005, num epoch = 300.
![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/k%3D10/plot.png)

Plot for k = 50, learning rate = 0.005, num epoch = 150.
![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/k%3D50/plot.png)

Plot for k = 100, learning rate = 0.005, num epoch = 100.
![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/k%3D100/plot.png)

Plot for k = 200, learning rate = 0.005, num epoch = 100.
![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/k%3D200/plot.png)

Plot for k = 500, learning rate = 0.001, num epoch = 300.
![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/k%3D500/plot.png)

![Alt Text](https://github.com/wsxwsx543/CSC311-Project/blob/kaka/starter_code/part_a/images/part_c/summary.png)

The k that achieves the highest validation accracy (0.692210) is 10, so we choose it.
