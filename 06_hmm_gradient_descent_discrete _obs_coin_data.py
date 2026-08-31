import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, M):
        self.M = M  # Number of hidden states

    def build(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        self.M, V = preSoftmaxB.shape

        self.preSoftmaxPi = tf.Variable(preSoftmaxPi, dtype=tf.float32)
        self.preSoftmaxA = tf.Variable(preSoftmaxA, dtype=tf.float32)
        self.preSoftmaxB = tf.Variable(preSoftmaxB, dtype=tf.float32)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    def get_cost(self, x):
        # Calculate log-likelihood for a single sequence x using the Forward algorithm
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        pi = tf.nn.softmax(self.preSoftmaxPi)
        A = tf.nn.softmax(self.preSoftmaxA)
        B = tf.nn.softmax(self.preSoftmaxB)

        def recurrence(old_a_old_s, x_t):
            old_a = tf.reshape(old_a_old_s[0], (1, self.M))
            a = tf.matmul(old_a, A) * tf.gather(B, x_t, axis=1)
            a = tf.reshape(a, (self.M,))
            s = tf.reduce_sum(a)
            return (a / s), s

        init_alpha = pi * tf.gather(B, x[0], axis=1)
        init_scale = tf.constant(1.0, dtype=tf.float32)

        # Loop through x[1:] using tf.scan
        alpha, scale = tf.scan(
            fn=recurrence,
            elems=x[1:],
            initializer=(init_alpha, init_scale),
        )

        cost = -tf.reduce_sum(tf.math.log(scale))
        return cost

    @tf.function(reduce_retracing=True)
    def train_step(self, x):
        # Perform one step of gradient descent using GradientTape
        with tf.GradientTape() as tape:
            cost = self.get_cost(x)
        trainable_vars = [self.preSoftmaxPi, self.preSoftmaxA, self.preSoftmaxB]
        grads = tape.gradient(cost, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return cost

    def fit(self, X, max_iter=10, print_period=1):
        # Train the HMM model using stochastic gradient descent
        N = len(X)
        print("Number of training samples:", N)

        costs = []
        for it in range(max_iter):
            if it % print_period == 0:
                print("Iteration:", it)

            for n in range(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                self.train_step(X[n])

        plt.plot(costs)
        plt.show()

    def get_cost_multi(self, X):
        return np.array([self.get_cost(x).numpy() for x in X])

    def init_random(self, V):
        # Initialize variables randomly before training
        preSoftmaxPi0 = np.zeros(self.M, dtype=np.float32)  # Initial state distribution
        preSoftmaxA0 = np.random.randn(self.M, self.M).astype(np.float32)  # Transition matrix
        preSoftmaxB0 = np.random.randn(self.M, V).astype(np.float32)  # Emission distribution

        self.build(preSoftmaxPi0, preSoftmaxA0, preSoftmaxB0)

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        # Manually assign parameter values
        self.preSoftmaxPi.assign(preSoftmaxPi)
        self.preSoftmaxA.assign(preSoftmaxA)
        self.preSoftmaxB.assign(preSoftmaxB)


def fit_coin():
    X = []
    # Load data from file or generate sample data if missing
    try:
        for line in open('coin_data.txt'):
            # 1 for H (Heads), 0 for T (Tails)
            x = [1 if e == 'H' else 0 for e in line.rstrip()]
            X.append(x)
    except FileNotFoundError:
        print("coin_data.txt not found. Generating dummy test data...")
        X = [np.random.choice([0, 1], size=10).tolist() for _ in range(5)]

    hmm = HMM(2)
    hmm.init_random(2)

    hmm.fit(X, max_iter=5)
    L = hmm.get_cost_multi(X).sum()
    print("LL with fitted params:", L)

    # Test with true parameter values (must be in pre-softmax forms)
    pi = np.log(np.array([0.5, 0.5])).astype(np.float32)
    A = np.log(np.array([[0.1, 0.9], [0.8, 0.2]])).astype(np.float32)
    B = np.log(np.array([[0.6, 0.4], [0.3, 0.7]])).astype(np.float32)

    hmm.set(pi, A, B)
    L = hmm.get_cost_multi(X).sum()
    print("LL with true params:", L)


if __name__ == '__main__':
    fit_coin()