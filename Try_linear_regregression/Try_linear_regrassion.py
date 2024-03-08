import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

x_train, y_train = load_data()

plt.scatter(x_train, y_train, marker="o", c="b")
plt.plot(x_train, y_train, c="r")
plt.title("CGPA Vs Avg_Package")
plt.xlabel("CGPA")
plt.ylabel("Avg_Package")
plt.show()


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = x[i] * w + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = cost_sum / (2 * m)
    return total_cost


m = len(x_train)


def compute_gredient(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = x[i] * w + b
        dj_db_i = f_wb - y[i]
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db = dj_db + dj_db_i
        dj_dw = dj_dw + dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


initial_w = 0
initial_b = 0


def gredient_descent(
    x, y, w_in, b_in, compute_cost, compute_gredient, alpha, num_iters
):
    j_history = []
    w_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gredient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 10000:
            cost = compute_cost(x, y, w, b)
            j_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(j_history[i]):8.2f}")
    return w, b, w_history, j_history


iterations = 1500
alpha = 0.0001

w, b, _, _ = gredient_descent(
    x_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gredient,
    alpha,
    iterations,
)
print("W,b found by gredient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    inp = float(input("Enter the CGPA:"))
    predicted[i] = w * inp + b
    print(f"Package of the student having CGPA: {inp} is {predicted[i]:} LPA")
plt.plot(x_train, predicted, c="r")

plt.plot(x_train, predicted, c="r")
plt.scatter(x_train, predicted, marker="o", c="g")
plt.title("CGPA Vs Avg_Package")
plt.xlabel("CGPA")
plt.ylabel("Avg_Package")
plt.show()
