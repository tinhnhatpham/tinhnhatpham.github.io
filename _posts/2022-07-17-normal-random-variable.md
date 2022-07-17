---
layout: post
title: The Normal Random Variable
categories: [Probability]
---
#### Gaussian Random Variable
**Definition:** A continous random variable $X$ has the normal distribution with parameters $\mu$ and $\sigma^2$ if its density is given by:

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2} \quad for -\infty < x < \infty
$$

**Notation:** $X \sim N(\mu, \sigma^2)$

**Properties:**
* $f(x)$ is symmetric about the line $x = \mu$
* $f(x) > 0$ and $\int_{-\infty}^{\infty}f(x)dx = 1$
* $E(X) = \int_{-\infty}^{\infty}xf(x)dx = \mu$
* $V(X) = \int_{-\infty}^{\infty}(x - \mu_x)^2f(x)dx = \sigma^2$
* $\sigma = \text{Standard Deviation}$, $\mu + \sigma$ & $\mu - \sigma$ are inflection points for f(x).

#### Standard Normal Distribution
**Definition:** The normal distribution with parameter values $\mu = 0$ and $\sigma^2 = 1$ is called the **standard normal** distribution.

A rv with the standard normal distribution is customarily denoted by $Z \sim N(0, 1)$ and its pdf is given by:

$$
f_Z(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2} \quad for -\infty < x < \infty
$$

We use special notation to denote the cdf of the standard normal curve:

$$
F(z) = \phi(z) = P(Z \le z) = \int_{-\infty}^{z}\frac{1}{\sqrt{2\pi}}e^{-x^2/2}dx
$$

**Proposition:** if $X \sim N(\mu,\sigma^2),$ then $\frac{X - \mu}{\sigma} \sim N(0, 1)$ 
