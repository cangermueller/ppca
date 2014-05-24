import numpy as np
from numpy.linalg import inv
from numpy import transpose as tr
import ipdb


class PPCA(object):
    def __init__(self, q=2, sigma=1.0):
        self.q = q
        self.prior_sigma = sigma

    def fit(self, y, em=False):
        self.y = y
        self.p = y.shape[0]
        self.n = y.shape[1]
        if em:
            [self.w, self.mu, self.sigma] = self.__fit_em()
        else:
            [self.w, self.mu, self.sigma] = self.__fit_ml()

    def transform(self, y=None):
        if y is None:
            y = self.y
        [w, mu, sigma] = [self.w, self.mu, self.sigma]
        m = tr(w).dot(w) + sigma * np.eye(w.shape[1])
        m = inv(m)
        x = m.dot(tr(w)).dot(y - mu)
        return x

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform()

    def transform_infers(self, x=None, noise=False):
        if x is None:
            x = self.transform()
        [w, mu, sigma] = [self.w, self.mu, self.sigma]
        y = w.dot(x) + mu
        if noise:
            for i in xrange(y.shape[1]):
                e = np.random.normal(0, sigma, y.shape[0])
                y[:, i] += e
        return y

    def __ell(self, w, mu, sigma, norm=True):
        m = inv(tr(w).dot(w) + sigma * np.eye(w.shape[1]))
        mw = m.dot(tr(w))
        ll = 0.0
        for i in xrange(self.n):
            yi = self.y[:, i][:, np.newaxis]
            yyi = yi - mu
            xi = mw.dot(yyi)
            xxi = sigma * m + xi.dot(tr(xi))
            ll += 0.5 * np.trace(xxi)
            if sigma > 1e-5:
                ll += (2 * sigma)**-1 * float(tr(yyi).dot(yyi))
                ll -= sigma**-1 * float(tr(xi).dot(tr(w)).dot(yyi))
                ll += (2 * sigma)**-1 * np.trace(tr(w).dot(w).dot(xxi))
        if sigma > 1e-5:
            ll += 0.5 * self.n * self.p * np.log(sigma)
        ll *= -1.0
        if norm:
            ll /= float(self.n)
        return ll

    def __fit_em(self, maxit=20):
        w = np.random.rand(self.p, self.q)
        mu = np.mean(self.y, 1)[:, np.newaxis]
        sigma = self.prior_sigma
        ll = self.__ell(w, mu, sigma)

        yy = self.y - mu
        s = self.n**-1 * yy.dot(tr(yy))
        for i in xrange(maxit):
            m = inv(tr(w).dot(w) + sigma * np.eye(self.q))
            t = inv(sigma * np.eye(self.q) + m.dot(tr(w)).dot(s).dot(w))
            w_new = s.dot(w).dot(t)
            sigma_new = self.p**-1 * np.trace(s - s.dot(w).dot(m).dot(tr(w_new)))
            ll_new = self.__ell(w_new, mu, sigma_new)
            print "{:3d}  {:.3f}".format(i + 1, ll_new)
            w = w_new
            sigma = sigma_new
            ll = ll_new
        return (w, mu, sigma)

    def __fit_ml(self):
        mu = np.mean(self.y, 1)[:, np.newaxis]
        [u, s, v] = np.linalg.svd(self.y - mu)
        if self.q > len(s):
            ss = np.zeros(self.q)
            ss[:len(s)] = s
        else:
            ss = s[:self.q]
        ss = np.sqrt(np.maximum(0, ss**2 - self.prior_sigma))
        w = u[:, :self.q].dot(np.diag(ss))
        if self.q < self.p:
            sigma = 1.0 / (self.p - self.q) * np.sum(s[self.q:]**2)
        else:
            sigma = 0.0
        return (w, mu, sigma)
