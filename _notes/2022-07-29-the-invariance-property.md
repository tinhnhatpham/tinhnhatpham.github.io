---
layout: post
comments: false
title: The invariance property
categories: [Likelihood Estimation]
---

Suppose you have a random sample from a distribution with mean $\mu$ and variance $\sigma^2$.
* An unbiased estimator of $\mu$ is $\bar{X}$.
* If we want to estimate $\mu^2$, should we use $\bar{X}^2$?

Not if we want an **unbiased estimator**!

$$
  \begin{align}
    E[\bar{X}^2] &= Var[\bar{X}] + (E[\bar{X}])^2 \\
    &= \frac{\sigma^2}{n} + \mu^2 \neq \mu^2
  \end{align}
$$

---

### The Invariance Property of MLEs

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim} exp(rate=\lambda)
$$

Find the MLE of $\lambda^2$. 

Let $\tau=\lambda^2$.

Reparameterize the pdf:

$$
  f(x;\lambda)=\lambda e^{-\lambda x}I_{(0,\infty)}(X)
$$

Becomes:

$$
  f_2(x;\tau) = \sqrt{\tau}e^{-\sqrt{\tau}x}I_{(0,\infty)}(X)
$$

After the taking the derivative and set equal $0$, we will get the MLE for $\tau$ is:

$$
  \hat{\tau} = \frac{1}{\bar{X}^2} = \hat{\lambda}^2
$$

The MLE for $\lambda$ is:

$$
  \hat{\lambda} = \frac{1}{\bar{X}}
$$

This is known as the **invariance property of MLES**. It means that if you want to estimate a function of a parameter using MLEs, you can find the MLE of the parameter and plug it into function.

---

Example:

$$
  X_1,X_2,...,X_n \sim Poisson(\lambda)
$$

How can we estimate the probability that a typical measurement from this data set is greater than zero?

Can we do this more formally with a maximum likelihood estimator?

One answer:

$$
  \hat{p_1} = \frac{\text{# values in the sample that are > 0}}{n}
$$

Other answer:

$$
  \begin{align}
    P(X_i\gt 0) &= 1 - P(X_i=0) \\
    &= 1 - \frac{e^{-\lambda}\lambda^0}{0!} = 1 - e^{-\lambda} \quad (\tau(\lambda))
  \end{align}
$$

The pdf is:

$$
  f(x;\lambda) = \frac{e^{-\lambda}\lambda^x}{x!}I_{(0,1,2,...)}(x)
$$

The joint pdf is:

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x}&=\prod_{i=1}^{n}\frac{e^{-\lambda}\lambda^{x_i}}{x_i!}I_{(0,1,2,...)}(x_i) \\
    &= \frac{e^{-\lambda}\lambda^{\sum_{i=1}^{n}x_i}}{\prod_{i=1}^{n}x_i!}\prod_{i=1}^{n}I_{(0,1,2,...)}(x_i)\\
    &\text{Drop both indicators and the product of factorials}\\
    &\text{because they don't involve $\theta$}\\
  \end{align}
$$

A likelihood is:

$$
  L(\lambda) = e^{-\lambda}\lambda^{\sum_{i=1}^{n}x_i}
$$

The log-likelihood is:

$$
  \ell(\lambda) = -n\lambda+(\sum_{i=1}^{n}x_i)\ln\lambda
$$

Take the derivative and set to 0:

$$
  \frac{\partial}{\partial\lambda}\ell(\lambda)=0 \quad \implies \hat{\lambda} = \bar{X}
$$

The MLE for $\lambda$ is $\hat{\lambda} = \bar{X}$

By the invariance property of MLEs, the MLE for $p = \tau(\lambda) = 1 - e^{-\lambda}$ is:

$$
  \hat{\tau}(\lambda) \stackrel{invar}{=}\tau(\hat{\lambda}) = 1 - e^{-\bar{X}}
$$
