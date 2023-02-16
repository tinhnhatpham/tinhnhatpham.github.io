---
layout: post
comments: false
title: Notation and Terminology
categories: [Likelihood Estimation]
---

### Maximum Likelihood Estimation
Given “data” $X_1, X_2, X_3,..., X_n$ , a random
sample (iid) from a distribution with unknown
parameter $\theta$, we want to find the value of $\theta$
in the parameter space that maximizes our
“probability” of observing that data.

* If $X_1, X_2, X_3,..., X_n$ are discrete, we can look at
$$
  P(X_1=x_1, X_2=x_2, ..., X_n=x_n)
$$
as a function of $\theta$, and find $\theta$ that maximizes it.

This is the joint pmf for $X_1, X_2, X_3,..., X_n$.

* The analogue for continous $X_1, X_2, X_3,..., X_n$ is to maximize the joint pdf with respect to $\theta$.

The pmf/pdf for any one of $X_1, X_2, X_3,..., X_n$ is denoted by $f(x)$.

We will emphasize the dependence of $f$ on a parameter $\theta$ by writing it as $f(x;\theta)$.

The join pmf/pdf for all of them is:

$$
    f(x_1,x_2,...,x_n;\theta) = f({\stackrel{\rightharpoonup}{x}; \theta}) = \prod_{i=1}^{n}f(x_i;\theta)
$$

* The data (the $x$'s) are fixed.
* Think of the $x$'s as fixed and the joint pdf as a function of $\theta$.

Call it a **likelihood function** and denote it by $L(\theta)$.

The likelihood $L(\theta)$ is defined to be anything proportional to the joint pmf/pdf.

---
### Discrete Example
$$X_1, X_2, X_3,..., X_n \stackrel{iid}{\sim} Bernoulli(p)$$

The pmf for one of them is
$$f(x;p) = p^x(1-p)^{1-x}I_{(0,1)}(x)$$

The joint pmf for all of them is:

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};p) &= \prod_{i=1}^{n}f(x_i;p) \\
    &= \prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i}I_{(0,1)}(x) \\
    &= p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}\prod_{i=1}^{n}I_{(0,1)}(x_i)
  \end{align}
$$

A likelihood is

$$
  L(p) = p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}
$$

It is almost always easier to minimize
the **log-likelihood**

$$
  l(p) = \ln L(p) = \sum_{i=1}^{n}x_i\ln p + (n - \sum_{i=1}^{n}x_i)\ln (1 - p)
$$

To maximize $l(p)$ w.r.t $p$, We take derivative w.r.t $p$ and set it to $0$

$$
  \begin{align}
    \frac{\partial}{\partial p}l(p) &= \frac{\sum_{i=1}^{n}x_i}{p} - \frac{n - \sum_{i=1}^{n}x_i}{1-p} = 0 \\
    &= p(1-p)\bigg[\frac{\sum_{i=1}^{n}x_i}{p} - \frac{n - \sum_{i=1}^{n}x_i}{1-p}\bigg] = p(1-p)0 \\
    &= (1-p)\sum_{i=1}^{n}x_i-p\bigg(n - \sum_{i=1}^{n}x_i \bigg) = 0 \\
    &\Rightarrow p = \frac{\sum_{i=1}^{n}x_i}{n}
  \end{align}
$$

The maximum likelihood estimator
for $p$ is:

$$\hat{p} = \frac{\sum_{i=1}^{n}x_i}{n} = \bar{X}$$

---

### Continous Example

$$X_1, X_2, X_3,..., X_n \stackrel{iid}{\sim} exp(rate = \lambda)$$

The pmf for one of them is

$$f(x;\lambda) = \lambda e^{-\lambda x}I_{(0, \infty)}(x)$$

The joint pdf for all of them is

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};\lambda) &= \prod_{i=1}^{n}f(x_i;\lambda) \\
  &= \prod_{i=1}^{n}\lambda e^{-\lambda x_i}I_{(0, \infty)}(x_i) \\
  &= \lambda^n e^{-\lambda \sum_{i=1}^{n}x_i}\prod_{i=1}^{n}I_{(0, \infty)}(x_i)
  \end{align} 
$$

A likelihood is

$$
  L(\lambda) = \lambda^n e^{-\lambda \sum_{i=1}^{n}x_i}
$$

The log-likelihood is

$$
  l(\lambda) = n\ln \lambda - \lambda\sum_{i=1}^{n}x_i
$$

To maximize $l(\lambda)$ w.r.t $\lambda$, We take derivative w.r.t $\lambda$ and set it to $0$

$$
  \begin{align}
    \frac{\partial}{\partial p}l(\lambda) &= \frac{n}{\lambda} - \sum_{i=1}^{n}x_i = 0 \\
    &= \lambda = \frac{n}{\sum_{i=1}^{n}x_i}
  \end{align}
$$

The maximum likelihood estimator for $\lambda$ is

$$
  \hat{\lambda} = \frac{n}{\sum_{i=1}^{n}x_i} = \frac{1}{\bar{X}}
$$
