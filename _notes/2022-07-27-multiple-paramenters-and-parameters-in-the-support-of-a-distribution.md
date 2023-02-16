---
layout: post
comments: false
title: Multiple parameters and parameters in the support of a distribution a
categories: [Likelihood Estimation]
---


### Normal Distribution

$$
  X_1, X_2, ..., X_n \stackrel{iid}{\sim} N(\mu, \sigma^2)$$

The pdf for one of them is

$$
  f(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{1}{2\sigma^2}(x-\mu)^2}
$$

The joint pdf for all of them is

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};\mu,\sigma^2) &= \prod_{i=1}^{n}f(x_i;\mu,\sigma^2) \\
    &= (2\pi \sigma^2)^{-n/2}e^{-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2}
  \end{align}
$$

The parameter space: $-\infty \lt \mu \lt \infty, \sigma^2 \gt 0$

It is almost always easier to minimize
the **log-likelihood**

$$
  \begin{align}
    L(\mu, \sigma^2) &= (2\pi \sigma^2)^{-n/2}e^{-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2} \\
    \ell(\mu, \sigma^2) &= -\frac{n}{2}\ln (2\pi \sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2
  \end{align}
$$

To maximize $\ell(\mu, \sigma^2)$ w.r.t $\mu$ and $\sigma^2$, we take a partial derivative w.r.t $\mu$ and $\sigma^2$, and set them to $0$.

$$
  \frac{\partial}{\partial\mu}\ell(\mu,\sigma^2) \stackrel{set}{=} 0 \\
  \frac{\partial}{\partial\sigma^2}\ell(\mu,\sigma^2) \stackrel{set}{=} 0 \\
$$

Solve for $\mu$ and $\sigma^2$ simutaneously. First we solve the derivative w.r.t $\mu$

$$
  \begin{align}
    \frac{\partial}{\partial\mu}\ell(\mu,\sigma^2) &= -\frac{1}{2\sigma^2}\sum_{i=1}^{n}2(x_i-\mu)(-1)\\
    &= \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) = 0 \\
  \end{align}
$$

* $\sigma^2 \gt 0$, so the equation above equal $0$ only:

$$
  \begin{align}
    &\implies \sum_{i=1}^{n}(x_i-\mu) = 0 \\
    &\implies \sum_{i=1}^{n}x_i-n\mu = 0 \\
    &\implies \mu = \sum_{i=1}^{n}x_i/n \\
    &\implies \hat{\mu} = \bar{X}
  \end{align}
$$

And solve the derivative w.r.t $\sigma^2$

$$
  \ell(\mu, \sigma^2) = -\frac{n}{2}\ln (2\pi \sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2
$$

$$
  \frac{\partial}{\partial\sigma^2}\ell(\mu,\sigma^2)=-\frac{n}{2}\frac{2\pi}{2\pi\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i-\mu)^2
$$

We already solved the value of $\mu$ above, we can put it in the equation and set the equation equal 0

$$
  \frac{\partial}{\partial\sigma^2}\ell(\mu,\sigma^2)=-\frac{n}{2}\frac{1}{\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i-\bar{X})^2 = 0
$$

Multiply both sides by $2(\sigma^2)^2$ and solve for $\sigma^2$ to get:

$$
  \implies \sigma^2=\frac{\sum_{i=1}^{n}(x_i-\bar{X})^2}{n}
$$

**The maximum likelihood estimators for $\mu$ and $\sigma^2$ are:**

$$
  \begin{align}
    \hat{\mu} &= \bar{X} \\
    \hat{\sigma^2} &= \frac{\sum_{i=1}^{n}(x_i-\bar{X})^2}{n}
  \end{align}
$$

**Note:** $Var[X] = E[(X - \mu)^2]$

A "natural" estimator is the **sample variance**.

$$
  S_1^2 := \frac{\sum_{i=1}^{n}(x_i-\bar{X})^2}{n}
$$

Let's find the expectation of this sample variance

$$
  E[S_1^2] = E\bigg[\frac{\sum_{i=1}^{n}(x_i-\bar{X})^2}{n} \bigg] = \frac{n-1}{n}\sigma^2
$$

So we can conclude that this definition of the sample variance **biased estimator**. If it's unbiased, we would get just $\sigma^2$. 

So from this, we can get the **unbiased version** of the sample variance by take the original variance and multiply it by the reciprocal of $\frac{n-1}{n}$

$$
  S^2 := \frac{n-1}{n}S_1^2 = \frac{\sum_{i=1}^{n}(x_i-\bar{X})^2}{n-1}
$$

The $n-1$ is called **the degrees of freedom** of this calculation.

---

### Uniform Distribution

$$
X_1, X_2, ..., X_n \stackrel{iid}{\sim} unif(0, \theta)
$$

The pdf for one of them is

$$
  f(x;\theta) = \frac{1}{\theta}I_{(0,\theta)}(x)
$$

The joint pdf for all of them is

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};\theta) &= \prod_{i=1}^{n}f(x_i;\theta) \\
    &= \frac{1}{\theta^n}\prod_{i=1}^{n}I_{(0;\theta)}(x_i)
  \end{align}
$$

The goal is to maximize the likelihood as a function $L(\theta)$
$$
  L(\theta) = \frac{1}{\theta^n}
$$

This is the decreasing function of $\theta$. As $\theta$ get larger the function get smaller. So to maximize this, we want to take the $\theta$ as small as possible. Consider the interval $(0,\theta)$, all the sample values need to be less than $\theta$, so the smallest of $\theta$ is the largest $X$.

The maximum likelihood estimator of $\theta$ is 

$$
  \hat{\theta} = max(X_1, X_2, ..., X_n)
$$


---

### The Poisson Distribution

$$
X_1, X_2, ..., X_n \stackrel{iid}{\sim} exp(\lambda)
$$

The pmf for one of them

$$
  f(x;\lambda) = \frac{e^{-\lambda}\lambda^x}{x!}I_{(0,1,...)}(x)
$$

The joint pmd for all of them

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};\lambda) &= \prod_{i=1}^{n}f(x_i;\theta) \\
    &= \prod_{i=1}^{n}e^{-\lambda}\frac{1}{x_i!}\lambda^x_i
  \end{align}
$$

The log-likelihood function is

$$
  \begin{align}
    \ell(\lambda) &= \ln \bigg(\prod_{i=1}^{n}e^{-\lambda}\frac{1}{x_i!}\lambda^{x_i} \bigg) \\
    &= \sum_{i=1}^{n}\ln \bigg(e^{-\lambda}\frac{1}{x_i!}\lambda^{x_i} \bigg) \\
    &= \sum_{i=1}^{n}\bigg(\ln(e^{-\lambda}) - \ln(x_i!) + \ln(\lambda^{x_i}) \bigg) \\
    &= \sum_{i=1}^{n}\bigg(-\lambda - \ln(x_i!) + x_i\ln(\lambda)\bigg) \\
    &= -n\lambda - \sum_{i=1}^{n}\ln(x_i!) + \ln(\lambda)\sum_{i=1}^{n}x_i
  \end{align} 
$$

To maximize the log-likelihood function, we take the derivative and set to $0$

$$
  \begin{align}
    \frac{\partial}{\partial\sigma^2}\ell(\lambda) &= \frac{\partial}{\partial\sigma^2}\bigg(-n\lambda - \sum_{i=1}^{n}\ln(x_i!) + \ln(\lambda)\sum_{i=1}^{n}x_i \bigg) \\
    &= -n + \frac{1}{\lambda}\sum_{i=1}^{n}x_i = 0\\
    &\implies \lambda = \frac{1}{n}\sum_{i=1}^{n}x_i
  \end{align}
$$

This is equivalent to the **sample mean** of the $n$ observations in the sample.
