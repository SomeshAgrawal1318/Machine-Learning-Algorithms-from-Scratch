"""Build one consolidated HW1 notebook (Q1-Q4) from the per-question code.

Data files are referenced relative to the DATA folder so the notebook runs
from the project root (the folder that contains HW1_data/).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------- Title
md(
"""# 50.007 Machine Learning — Homework 1

This single notebook contains all four questions, numbered Q1–Q4 with their
sub-parts. Run the cells top to bottom.

**Data location.** All data files are read from the `HW1_data/` folder using the
`DATA` path set in the next cell. Run this notebook from the project folder that
contains `HW1_data/` (i.e. the folder with the per-algorithm subfolders). If your
data files live elsewhere, just change `DATA` below.
"""
)
code(
"""import os
import numpy as np
import matplotlib.pyplot as plt

# Folder that holds the per-algorithm data subfolders.
DATA = 'HW1_data'

# Each question maps to the folder of the algorithm it implements.
FOLDERS = {
    1: 'perceptron',
    2: 'linear-and-polynomial-regression',
    3: 'ridge-regression',
    4: 'k-means',
}

def datapath(q, name):
    \"\"\"Path to a data file `name` inside the folder for question `q`.\"\"\"
    return os.path.join(DATA, FOLDERS[q], name)
"""
)

# ================================================================ Q1
md(
"""# Q1 — Perceptron (linear classification)

We train a perceptron **with an offset** on the 1-vs-5 digit data
(`train_1_5.csv`) and evaluate accuracy on `test_1_5.csv`. Each row is
`symmetry, intensity, label`. As required, we **do not shuffle** — instances are
visited sequentially. On a misclassification ($y_i(\\theta\\cdot x_i+\\theta_0)\\le 0$)
we update $\\theta \\leftarrow \\theta + y_i x_i$ and $\\theta_0 \\leftarrow \\theta_0 + y_i$.
"""
)
code(
"""# Load the data. Each row is: symmetry, intensity, label.
train = np.loadtxt(datapath(1, 'train_1_5.csv'), delimiter=',')
test = np.loadtxt(datapath(1, 'test_1_5.csv'), delimiter=',')

X_train, y_train = train[:, :2], train[:, 2]
X_test,  y_test  = test[:, :2],  test[:, 2]


def perceptron(X, y, epochs):
    # Perceptron with an offset. Visit points in order (no shuffling); only
    # update on a misclassification.
    theta = np.zeros(X.shape[1])
    theta_0 = 0.0
    for e in range(epochs):
        for i in range(len(X)):
            xi, yi = X[i], y[i]
            if yi * (np.dot(theta, xi) + theta_0) <= 0:
                theta = theta + yi * xi
                theta_0 = theta_0 + yi
    return theta, theta_0


def accuracy(X, y, theta, theta_0):
    # fraction of points classified correctly
    preds = np.sign(np.dot(X, theta) + theta_0)
    return np.mean(preds == y)
"""
)
md("## Q1(a) — Perceptron with offset, 1 epoch")
code(
"""theta1, theta0_1 = perceptron(X_train, y_train, epochs=1)
acc1 = accuracy(X_test, y_test, theta1, theta0_1)

print('After 1 epoch:')
print('theta  =', theta1)
print('offset =', theta0_1)
print('test accuracy =', round(acc1, 4))
"""
)
md("## Q1(b) — Perceptron with offset, 5 epochs")
code(
"""theta5, theta0_5 = perceptron(X_train, y_train, epochs=5)
acc5 = accuracy(X_test, y_test, theta5, theta0_5)

print('After 5 epochs:')
print('theta  =', theta5)
print('offset =', theta0_5)
print('test accuracy =', round(acc5, 4))
"""
)
md(
"""## Q1(c) — How to run

This notebook *is* the submitted code. To reproduce Q1: run the three cells
above in order. `perceptron(X_train, y_train, epochs)` returns `(theta, offset)`
and `accuracy(...)` returns the test accuracy. The data is read from
`HW1_data/perceptron/`.
"""
)

# ================================================================ Q2
md(
"""# Q2 — Linear and polynomial regression

Inputs `hw1x.dat`, targets `hw1y.dat`. We report the weight vector $\\theta$ and
the **training error as empirical risk** (mean squared error)
$R(\\theta)=\\tfrac1n\\sum_i (x_i^\\top\\theta - y_i)^2$.
"""
)
md("## Q2(a) — Closed-form linear regression")
code(
"""# Load x, y and prepend a column of 1s (bias term).
x = np.loadtxt(datapath(2, 'hw1x.dat'))
y = np.loadtxt(datapath(2, 'hw1y.dat'))
X = np.column_stack([np.ones(len(x)), x])
print('X shape:', X.shape, ' y shape:', y.shape)


def linear_regression(X, y):
    # Closed-form normal equations: theta = (X^T X)^(-1) X^T y
    return np.linalg.inv(X.T @ X) @ X.T @ y


def mse(X, y, theta):
    # empirical risk = mean squared error
    return np.mean((X @ theta - y) ** 2)


theta_cf = linear_regression(X, y)
print('theta (bias, slope):', np.round(theta_cf, 4))
print('training MSE       :', round(mse(X, y, theta_cf), 4))
"""
)
code(
"""# Plot the data and the fitted line.
x_line = np.linspace(x.min(), x.max(), 200)
y_line = theta_cf[0] + theta_cf[1] * x_line

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=20, alpha=0.6, label='data')
plt.plot(x_line, y_line, 'r', linewidth=2, label='linear fit')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Q2(a) Linear Regression (closed form)')
plt.legend(); plt.tight_layout()
plt.show()
"""
)
md(
"""## Q2(b) — Gradient descent and stochastic gradient descent

Both use learning rate $\\eta=0.01$ for 5 epochs. We report the final $\\theta$ and
the training error (empirical risk).
"""
)
code(
"""def gradient_descent(X, y, eta=0.01, epochs=5):
    # Batch GD: one update per epoch using the full-data gradient.
    n = len(y)
    theta = np.zeros(X.shape[1])
    errors = []
    for e in range(epochs):
        grad = X.T @ (X @ theta - y) / n
        theta = theta - eta * grad
        errors.append(mse(X, y, theta))
    return theta, errors


def stochastic_gradient_descent(X, y, eta=0.01, epochs=5):
    # SGD: one update per data point; shuffle order each epoch.
    np.random.seed(0)
    n = len(y)
    theta = np.zeros(X.shape[1])
    errors = []
    for e in range(epochs):
        for i in np.random.permutation(n):
            grad = X[i] * (X[i] @ theta - y[i])
            theta = theta - eta * grad
        errors.append(mse(X, y, theta))
    return theta, errors


theta_gd, errors_gd = gradient_descent(X, y)
theta_sgd, errors_sgd = stochastic_gradient_descent(X, y)
print('Batch GD final theta:', np.round(theta_gd, 4),
      ' min training MSE:', round(min(errors_gd), 4))
print('SGD      final theta:', np.round(theta_sgd, 4),
      ' min training MSE:', round(min(errors_sgd), 4))
"""
)
code(
"""# Compare how fast each method drives the error down.
ep = range(1, 6)
plt.figure(figsize=(7, 4))
plt.plot(ep, errors_gd, 'o-', label='GD')
plt.plot(ep, errors_sgd, 's--', label='SGD')
plt.axhline(mse(X, y, theta_cf), color='gray', linestyle=':', label='closed form')
plt.xlabel('epoch'); plt.ylabel('training MSE')
plt.title('Q2(b) GD vs SGD (5 epochs, eta=0.01)')
plt.legend(); plt.tight_layout()
plt.show()
"""
)
md(
"""## Q2(c) — Polynomial regression

`PolyRegress(x, y, d)` builds the design matrix $[1, x, x^2, \\dots, x^d]$ and
solves it with the same closed form. We give the quadratic (d=2) fit, then sweep
d = 3 … 15 and report the training error.
"""
)
code(
"""def PolyRegress(x, y, d):
    # Build [1, x, x^2, ..., x^d] and reuse the closed-form solver.
    X_poly = np.column_stack([x ** k for k in range(d + 1)])
    theta = linear_regression(X_poly, y)
    return theta, mse(X_poly, y, theta)


def poly_predict(theta, x):
    return sum(theta[k] * x ** k for k in range(len(theta)))


theta2, mse2 = PolyRegress(x, y, 2)
print('degree-2 theta:', np.round(theta2, 4))
print('degree-2 MSE  :', round(mse2, 4))

x_line = np.linspace(x.min(), x.max(), 300)
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=18, alpha=0.5, label='data')
plt.plot(x_line, poly_predict(theta2, x_line), 'r', linewidth=2, label='degree-2 fit')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Q2(c) Polynomial Regression (degree 2)')
plt.legend(); plt.tight_layout()
plt.show()
"""
)
code(
"""# Fit degrees 1..15 and record the training error.
degrees = list(range(1, 16))
mse_list, theta_list = [], []
for d in degrees:
    th, er = PolyRegress(x, y, d)
    theta_list.append(th); mse_list.append(er)
    print('degree', d, '-> MSE', round(er, 4))
"""
)
code(
"""# All 15 fits in a grid.
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for i, ax in enumerate(axes.flatten()):
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.plot(x_line, poly_predict(theta_list[i], x_line), 'r', linewidth=1.5)
    ax.set_ylim(y.min() - 1, y.max() + 1)
    ax.set_title(f'd={degrees[i]}, MSE={round(mse_list[i], 2)}', fontsize=9)
plt.suptitle('Q2(c) Polynomial fits, degree 1 to 15', fontsize=13)
plt.tight_layout(); plt.show()
"""
)
code(
"""# Training error vs degree, and the first degree where error increases.
plt.figure(figsize=(8, 4))
plt.plot(degrees, mse_list, 'o-', color='purple')
plt.xlabel('polynomial degree'); plt.ylabel('training MSE')
plt.title('Q2(c) Training error vs polynomial degree')
plt.xticks(degrees); plt.tight_layout(); plt.show()

for i in range(1, len(mse_list)):
    if mse_list[i] > mse_list[i - 1]:
        print('Error first gets worse at degree', degrees[i],
              ':', round(mse_list[i - 1], 4), '->', round(mse_list[i], 4))
        break
"""
)
md(
"""**Why the error gets worse at high degree.** In theory the training error can
never increase as we add polynomial terms. It does here only because of
numerical conditioning: the columns $x, x^2, \\dots, x^{15}$ become almost
collinear, so $X^\\top X$ is nearly singular and `np.linalg.inv` loses accuracy.
The blow-up starting at degree 11 is a numerical artefact, not a genuinely worse
fit. Solving with `np.linalg.lstsq` / `np.linalg.pinv` keeps the error flat.
"""
)

# ================================================================ Q3
md(
"""# Q3 — Ridge regression

Inputs `hw1_ridge_x.dat` (already include a column of 1s), targets
`hw1_ridge_y.dat`. First 10 rows = validation set, last 40 rows = training set.
The closed-form ridge solution is

$$\\hat{\\theta} = (n\\lambda I + X^\\top X)^{-1} X^\\top Y,$$

with $n$ = number of training points (40).
"""
)
code(
"""Xr = np.loadtxt(datapath(3, 'hw1_ridge_x.dat'), delimiter=',')
yr = np.loadtxt(datapath(3, 'hw1_ridge_y.dat'))

# first 10 -> validation, last 40 -> training
vX, vY = Xr[:10], yr[:10]
tX, tY = Xr[10:], yr[10:]
print('training:', tX.shape, ' validation:', vX.shape)
"""
)
md("## Q3(a) — ridge_regression(tX, tY, l), report theta for lambda = 0.15")
code(
"""def ridge_regression(X, y, l):
    # closed-form ridge: theta = (n*l*I + X^T X)^(-1) X^T y
    n, d = X.shape
    return np.linalg.inv(n * l * np.eye(d) + X.T @ X) @ X.T @ y


theta_ridge = ridge_regression(tX, tY, 0.15)
print('theta for lambda = 0.15:')
print(theta_ridge)
"""
)
md(
"""## Q3(b) — Choose lambda with the validation set

We sweep $\\lambda$ on a log scale (the grid from the question) and plot training
vs validation loss. The chosen $\\lambda$ is the one minimising the validation loss.
"""
)
code(
"""tn = tX.shape[0]
vn = vX.shape[0]
index = -np.arange(0, 5, 0.1)         # log10(lambda): 0 down to -4.9
tloss, vloss = [], []
for i in index:
    w = ridge_regression(tX, tY, 10 ** i)
    tloss.append(np.sum((tX @ w - tY) ** 2) / tn / 2)
    vloss.append(np.sum((vX @ w - vY) ** 2) / vn / 2)

plt.plot(index, np.log(tloss), 'r', label='training loss')
plt.plot(index, np.log(vloss), 'b', label='validation loss')
plt.xlabel('log10(lambda)'); plt.ylabel('log(loss)')
plt.title('Q3(b) Ridge regression: training vs validation loss')
plt.legend(); plt.tight_layout(); plt.show()

best = int(np.argmin(vloss))
print('lambda minimising validation loss:', 10 ** index[best],
      '  (10 ^', round(index[best], 1), ')')
"""
)

# ================================================================ Q4
md(
"""# Q4 — K-Means on an image

`kmeans-image.txt` holds one RGB vector per pixel (516×407 image). We run our own
K-Means with $K=8$, the given initial centroids, and **squared Euclidean
distance**. Each iteration: assign every pixel to its nearest centroid, recompute
each centroid as the mean of its pixels; stop when assignments stop changing. We
report the error at each iteration and finally display the quantised image.
"""
)
code(
"""ROWS, COLS = 516, 407
img = np.loadtxt(datapath(4, 'kmeans-image.txt'))
print('data shape:', img.shape)

centroids_init = np.array([
    [255, 255, 255], [255,   0,   0], [128,   0,   0], [  0, 255,   0],
    [  0, 128,   0], [  0,   0, 255], [  0,   0, 128], [  0,   0,   0],
], dtype=float)
K = 8
"""
)
code(
"""def assign_clusters(data, centroids):
    # nearest centroid for each pixel (squared Euclidean distance)
    dist2 = np.zeros((len(data), K))
    for k in range(K):
        dist2[:, k] = np.sum((data - centroids[k]) ** 2, axis=1)
    return np.argmin(dist2, axis=1)


def update_centroids(data, labels, centroids):
    # move each centroid to the mean of its assigned pixels
    new = centroids.copy()
    for k in range(K):
        pts = data[labels == k]
        if len(pts) > 0:
            new[k] = pts.mean(axis=0)
    return new


def total_error(data, labels, centroids):
    # total within-cluster squared distance
    return sum(np.sum((data[labels == k] - centroids[k]) ** 2) for k in range(K))
"""
)
code(
"""# Run K-Means until assignments stop changing.
centroids = centroids_init.copy()
labels = None
it = 0
print('iter | error')
while True:
    it += 1
    new_labels = assign_clusters(img, centroids)
    err = total_error(img, new_labels, centroids)
    print(it, '|', round(err, 2))
    if labels is not None and np.array_equal(new_labels, labels):
        print('Converged after', it, 'iterations.')
        break
    labels = new_labels
    centroids = update_centroids(img, labels, centroids)
"""
)
code(
"""# Final centroids and pixel counts.
print('  k        R        G        B     pixels')
for k in range(K):
    r, g, b = centroids[k]
    print(f'{k:3d}  {r:7.2f}  {g:7.2f}  {b:7.2f}  {np.sum(labels == k):9d}')
"""
)
code(
"""# Replace each pixel by its assigned centroid colour and show both images.
quantised = centroids[labels].reshape(ROWS, COLS, 3).astype(np.uint8)
original  = img.reshape(ROWS, COLS, 3).astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(11, 7))
ax[0].imshow(original);  ax[0].set_title('Original');       ax[0].axis('off')
ax[1].imshow(quantised); ax[1].set_title('K-Means (K=8)');  ax[1].axis('off')
plt.tight_layout(); plt.show()
"""
)
md(
"""**Note.** Clusters 3 (pure green) and 5 (pure blue) end up with 0 pixels: the
image (a portrait with warm skin tones, a dark suit, a light background) has no
pure-green or pure-blue pixels, so those two centroids never attract any points
and stay at their initial positions. The other 6 centroids cover the real colours.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out = "Somesh_Agrawal_HW1.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
