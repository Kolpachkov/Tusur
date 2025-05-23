import numpy as np
import matplotlib.pyplot as plt

class Act:
    def act(self, g):
        return 1.0 / (np.exp(-np.clip(g, -100, 100)) + 1.0)

    def actdif(self, g):
        ff = np.exp(-np.clip(g, -100, 100))
        return ff / (ff + 1.0) ** 2

class LeakyRelu(Act):
    def act(self, g):
        return np.where(g < 0, 0.01 * g, g)

    def actdif(self, g):
        return np.where(g < 0, 0.01, 1.0)

class Line(Act):
    def act(self, g):
        return g

    def actdif(self, g):
        return np.ones_like(g)

class Layer:
    def __init__(self, n, neursz, act):
        self.n = n
        self.neursz = neursz
        self.w = np.random.uniform(low=-0.5, high=0.5, size=(n, neursz))
        self.act = act
        self.s = np.zeros(n)
        self.y = self.act.act(self.s)
        self.sg = self.y

    def ask(self, x):
        self.s = np.dot(self.w, x)
        self.y = self.act.act(self.s)
        return self.y

class NeuroNet:
    def __init__(self, nin, lays, acts):
        nins = [nin] + lays
        self.lays = [Layer(lays[i], nins[i], acts[i]) for i in range(len(lays))]

    def ask(self, x):
        y = self.lays[0].ask(x)
        for i in range(1, len(self.lays)):
            y = self.lays[i].ask(y)
        return y

    def back_ppg_learn(self, x, y, al):
        il = len(self.lays) - 1
        d = self.ask(x)
        self.lays[il].sg = (d - y) * self.lays[il].act.actdif(self.lays[il].s)

        for il in range(len(self.lays) - 2, -1, -1):
            self.lays[il].sg = np.dot(self.lays[il + 1].w.T, self.lays[il + 1].sg) * self.lays[il].act.actdif(self.lays[il].s)

        for il in range(len(self.lays)):
            grad = np.dot(np.array([self.lays[il].sg]).T, np.array([x if il == 0 else self.lays[il - 1].y]))
            if np.isnan(grad).any():
                print(f"NaN detected in layer {il}")
            self.lays[il].w -= grad * al

def find_intersection(A1, B1, C1, A2, B2, C2):
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return np.array([0.0, 0.0])
    x = (B1 * C2 - B2 * C1) / det
    y = (A2 * C1 - A1 * C2) / det
    return np.array([x, y])

N = 1000
coefficients = np.random.uniform(-100, 100, (N, 6))
intersections = np.array([find_intersection(*coefficients[i]) for i in range(N)])

# **Нормализация данных**
coefficients /= np.max(np.abs(coefficients), axis=0)
intersections /= np.max(np.abs(intersections), axis=0)

plt.scatter(intersections[:, 0], intersections[:, 1])
plt.title("Real intersections before training")
plt.show()

net = NeuroNet(6, [20, 40, 20, 2], [LeakyRelu(), LeakyRelu(), LeakyRelu(), Line()])
N_EPOCHS = 500
for i in range(N_EPOCHS):
    for j in range(N):
        n_ex = np.random.randint(0, N)
        net.back_ppg_learn(coefficients[n_ex], intersections[n_ex], 0.05)
    if i % 10 == 0:
        print(f"Epoch {i}: Gradient sample {np.mean(net.lays[-1].sg)}")

predictions = np.array([net.ask(coefficients[j]) for j in range(N)])

plt.scatter(intersections[:, 0], intersections[:, 1], label="True")
plt.scatter(predictions[:, 0], predictions[:, 1], label="Predicted", alpha=0.5)
plt.legend()
plt.title("Predicted vs True Intersections")
plt.show()
