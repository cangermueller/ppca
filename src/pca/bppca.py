import numpy as np
from numpy import transpose as tr
from numpy.linalg import inv
import pdb

class BPPCA(object):
    def __init__(self, y, q=2, hyper=None):
        self.y = y
        self.p = y.shape[0]
        self.q = q
        self.n = y.shape[1]
        if hyper is None:
            self.hyper = HyperParameters()
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.n, self.p, self.q)

    def fit(self, *args, **kwargs):
        self.fit_vb(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform()

    def transform(self, y=None):
        if y is None:
            return self.q_dist.x_mean
        q = self.q_dist
        [w, mu, sigma] = [q.w, q.mu, q.gamma**-1]
        m = tr(w).dot(w) + sigma * np.eye(w.shape[1])
        m = inv(m)
        x = m.dot(tr(w)).dot(y - mu)
        return x

    def transform_infers(self, x=None, noise=False):
        q = self.q_dist
        if x is None:
            x = q.x_mean
        [w, mu, sigma] = [q.w_mean, q.mu_mean, q.gamma_mean()**-1]
        y = w.dot(x) + mu[:, np.newaxis]
        if noise:
            for i in xrange(y.shape[1]):
                e = np.random.normal(0, sigma, y.shape[0])
                y[:, i] += e
        return y

    def mse(self):
        d = self.y - self.transform_infers()
        d = d.ravel()
        return self.n**-1 * d.dot(d)

    def fit_vb(self, maxit=20):
        for i in xrange(maxit):
            self.update()

    def update(self):
        self.update_mu()
        self.update_w()
        self.update_x()
        self.update_alpha()
        self.update_gamma()

    def update_x(self):
        q = self.q_dist
        gamma_mean = q.gamma_a / q.gamma_b
        q.x_cov = inv(np.eye(self.q) + gamma_mean * tr(q.w_mean).dot(q.w_mean))
        q.x_mean = gamma_mean * q.x_cov.dot(tr(q.w_mean)).dot(self.y - q.mu_mean[:, np.newaxis])

    def update_w(self):
        q = self.q_dist
        # cov
        x_cov = np.zeros((self.q, self.q))
        for n in xrange(self.n):
            x = q.x_mean[:, n]
            x_cov += x[:, np.newaxis].dot(np.array([x]))
        q.w_cov = np.diag(q.alpha_a / q.alpha_b) + q.gamma_mean() * x_cov
        q.w_cov = inv(q.w_cov)
        # mean
        yc = self.y - q.mu_mean[:, np.newaxis]
        q.w_mean = q.gamma_mean() * q.w_cov.dot(q.x_mean.dot(tr(yc)))
        q.w_mean = tr(q.w_mean)

    def update_mu(self):
        q = self.q_dist
        gamma_mean = q.gamma_a / q.gamma_b
        q.mu_cov = (self.hyper.beta + self.n * gamma_mean)**-1 * np.eye(self.p)
        q.mu_mean = np.sum(self.y - q.w_mean.dot(q.x_mean), 1)
        q.mu_mean = gamma_mean * q.mu_cov.dot(q.mu_mean)

    def update_alpha(self):
        q = self.q_dist
        q.alpha_a = self.hyper.alpha_a + 0.5 * self.p
        q.alpha_b = self.hyper.alpha_b + 0.5 * np.linalg.norm(q.w_mean, axis=0)**2

    def update_gamma(self):
        q = self.q_dist
        q.gamma_a = self.hyper.gamma_a + 0.5 * self.n * self.p
        q.gamma_b = self.hyper.gamma_b
        w = q.w_mean
        ww = tr(w).dot(w)
        for n in xrange(self.n):
            y = self.y[:, n]
            x = q.x_mean[:, n]
            q.gamma_b += y.dot(y) + q.mu_mean.dot(q.mu_mean)
            q.gamma_b += np.trace(ww.dot(x[:, np.newaxis].dot([x])))
            q.gamma_b += 2.0 * q.mu_mean.dot(w).dot(x[:, np.newaxis])
            q.gamma_b -= 2.0 * y.dot(w).dot(x)
            q.gamma_b -= 2.0 * y.dot(q.mu_mean)



class HyperParameters(object):
    def __init__(self):
        self.alpha_a = 1.0
        self.alpha_b = 1.0
        self.gamma_a = 1.0
        self.gamma_b = 1.0
        self.beta = 1.0

class Qdistribution(object):
    def __init__(self, n, p, q):
        self.n = n
        self.p = p
        self.q = q
        self.init_rnd()

    def init_rnd(self):
        self.x_mean = np.random.normal(0.0, 1.0, self.q * self.n).reshape(self.q, self.n)
        self.x_cov = np.eye(self.q)
        self.w_mean = np.random.normal(0.0, 1.0, self.p * self.q).reshape(self.p, self.q)
        self.w_cov = np.eye(self.q)
        self.alpha_a = 1.0
        self.alpha_b = np.empty(self.q)
        self.alpha_b.fill(1.0)
        self.mu_mean = np.random.normal(0.0, 1.0, self.p)
        self.mu_cov = np.eye(self.p)
        self.gamma_a = 1.0
        self.gamma_b = 1.0

    def gamma_mean(self):
        return self.gamma_a / self.gamma_b

    def alpha_mean(self):
        return self.alpha_a / self.alpha_b
