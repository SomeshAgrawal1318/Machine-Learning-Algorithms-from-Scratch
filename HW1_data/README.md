# 50.007 Machine Learning — Homework 1

My solutions for the four questions. Each algorithm is in its own folder
with a Jupyter notebook (`task-N.ipynb`) and the data files from the assignment.

```
perceptron/                        Perceptron (linear classification)
linear-and-polynomial-regression/  Linear and polynomial regression
ridge-regression/                  Ridge regression
k-means/                           K-Means on an image
```

## How to run

Everything is plain Python 3 with `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib jupyter
```

Open a notebook and run the cells top to bottom, for example:

```bash
jupyter notebook perceptron/task-1.ipynb
```

The data files are loaded with relative paths, so each notebook has to be run from
inside its own folder.

---

## Q1 — Perceptron

I implemented the perceptron with an offset. The training points are visited in
order (no shuffling), like the question asks. When a point is misclassified
($y_i(\theta \cdot x_i + \theta_0) \le 0$) I update $\theta \leftarrow \theta + y_i x_i$
and $\theta_0 \leftarrow \theta_0 + y_i$.

Results on the test set (`test_1_5.csv`):

| run | theta | offset | test accuracy |
|-----|-------|--------|---------------|
| (a) 1 epoch  | `[-2.4483, -5.8819]` | `0.0` | 0.9671 |
| (b) 5 epochs | `[-2.0590, -8.8352]` | `1.0` | 0.9671 |

The data is close to linearly separable, so even a single pass gets about 97% of
the test set right. Five epochs changes the weights but the test accuracy stays
the same.

## Q2 — Linear and polynomial regression

**(a) Closed form.** I add a column of 1s and solve the normal equations
$\theta = (X^TX)^{-1}X^TY$.

- theta = `[3.2447, 1.7816]` (bias, slope)
- training MSE = 1.1625

The fit is in `linear_regression.png`.

**(b) Gradient descent and stochastic gradient descent**, both with learning rate
$\eta = 0.01$ for 5 epochs:

- Batch GD final theta = `[0.2413, 0.2703]`, training MSE after 5 epochs ≈ 22.37
- SGD final theta = `[3.0091, 1.9637]`, training MSE after 5 epochs ≈ 1.18

Batch GD only makes one update per epoch, so with this small step size and only 5
epochs it barely moves away from zero and its error is still big. SGD makes one
update per point (200 per epoch), so it gets close to the closed-form answer in the
same 5 epochs. The comparison is in `gd_convergence.png`. (SGD shuffles the points
each epoch, so the exact numbers depend on the random seed; I fixed it with
`np.random.seed(0)`.)

**(c) Polynomial regression.** `PolyRegress(x, y, d)` builds the design matrix
$[1, x, x^2, \dots, x^d]$ and solves it the same way as part (a). Training MSE for
each degree:

| d | MSE | d | MSE |
|---|------|---|------|
| 1 | 1.1625 | 9  | 1.1058 |
| 2 | 1.1406 | 10 | 1.1052 |
| 3 | 1.1402 | 11 | 1.4402 |
| 4 | 1.1292 | 12 | 6.4399 |
| 5 | 1.1278 | 13 | 9.3212 |
| 6 | 1.1227 | 14 | 5.7676 |
| 7 | 1.1125 | 15 | 100.5221 |
| 8 | 1.1076 |    |        |

The error goes down nicely up to degree 10, but then it jumps up at degree 11 and
gets very large by degree 15. In theory adding more polynomial terms should never
increase the training error, so something numerical is going on. The reason is that
the columns $x, x^2, \dots, x^{15}$ become almost the same size and direction, so
$X^TX$ is nearly singular (ill-conditioned) and `np.linalg.inv` can't solve it
accurately anymore. So the big errors at high degree are a numerical problem, not a
genuinely worse fit. Using `np.linalg.lstsq` (or `np.linalg.pinv`) instead of `inv`
is more stable and would keep the error flat. The quadratic fit is `poly_degree2.png`,
all the degrees together are in `poly_all_degrees.png`, and the error-vs-degree curve
is in `poly_error_vs_degree.png`.

## Q3 — Ridge regression

The inputs already include the column of 1s. The first 10 rows are the validation
set and the last 40 rows are the training set. The closed form I use is
$\hat{\theta} = (n\lambda I + X^TX)^{-1}X^TY$, where $n = 40$ (the number of
training points).

**(a)** For $\lambda = 0.15$:

```
theta = [-0.5794,  1.1503,  0.0493, -1.5987]
```

**(b)** Sweeping $\lambda$ on a log scale (the same grid as the code in the question),
the validation loss is smallest at

```
lambda ~ 0.0126   (about 10^-1.9)
```

The plot of training vs validation loss is in `ridge_loss_vs_lambda.png`. The
training loss just keeps going up as $\lambda$ grows, while the validation loss dips
to a minimum and then rises again — that minimum is the lambda we want.

## Q4 — K-Means on an image

My own K-Means using squared Euclidean distance, with K = 8 and the initial
centroids from the table in the question. Each iteration: assign every pixel to its
nearest centroid, recompute each centroid as the mean of its pixels, and stop when
the assignments stop changing.

It converged after 48 iterations. The error (total within-cluster sum of squared
distances) drops fast in the first few iterations — from about 1.29e9 down to roughly
1.18e8 — and then flattens out.

Final centroids and pixel counts:

| k | R | G | B | pixels |
|---|------|------|------|--------|
| 0 | 241.23 | 238.63 | 233.86 |  4,930 |
| 1 | 194.41 | 136.33 |  90.94 | 15,190 |
| 2 | 136.27 |  61.09 |  10.10 | 52,535 |
| 3 |   0.00 | 255.00 |   0.00 |      0 |
| 4 | 157.29 |  97.59 |  51.43 | 22,075 |
| 5 |   0.00 |   0.00 | 255.00 |      0 |
| 6 |  78.93 |  37.11 |  13.07 | 40,365 |
| 7 |  25.98 |  23.24 |  23.61 | 74,917 |

Clusters 3 (pure green) and 5 (pure blue) end up with 0 pixels. The image is a
portrait with warm skin tones, a dark suit and a light background, so there are no
pure green or pure blue pixels for those two centroids to grab — they just stay where
they started. The other 6 centroids cover the real colours in the image.

The side-by-side of the original and the 8-colour image is saved as
`k-means/kmeans_result.png`.
