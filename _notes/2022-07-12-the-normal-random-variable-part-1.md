---
layout: post
comments: false
title: The Gaussian (normal) Random Variable Part 1
categories: [Continuous Random Variables]
---

### **The Normal Distribution**

Normal (or Gaussian) distribution is probably the most important, and widely used, distribution in all of probability and statistics.

Many populations have distributions that can be fit very closely by an appropriate normal bell curve.

Examples: height, weight, and other physical characteristics, scores on some tests, some error measurements, etc. can be modeled by a Gaussian distribution.

* First used by Abraham deMoivre in 1733, later by many
others, including Carl Friedrich Gauss.

* Gauss used it so extensively in his astronomical
calculations, it came to be called the Gaussian
distribution.

* In 1893, Karl E. Pearson wrote “Many years ago | called
the Laplace-Gaussian curve the normal curve, which
name, while it avoids the international question of
priority, has the disadvantage of leading people to believe
that all other distributions of frequency are in one sense
or another abnormal.”

**Definition:** A continuous random variable $X$ has the normal distribution with parameters $\mu$ and $\sigma^2$ if its desity is given by

$$
  \color{red}{f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}}\quad\text{for }-\infty\lt x\lt\infty
$$

**Notation:** $X\sim N(\mu,\sigma^2)$

![png](\assets\images\notes\2022-07-12-the-normal-random-variable-part-1-1.png)

**Properties:**

* $f(x)$ is symmetric about the line $x=\mu$.

* $f(x)\gt0$ and $\int_{-\infty}^{\infty}f(x)dx = 1$.

* $E(X)=\int_{-\infty}^{\infty}xf(x)dx=\mu$.

* $V(X)=\int_{-\infty}^{\infty}(x-\mu_X)^2f(x)dx=\sigma^2$.

* $\sigma$ is standard deviation. $\mu + \sigma$ and $\mu - \sigma$ are inflection points for $f(x)$.

![png](\assets\images\notes\2022-07-12-the-normal-random-variable-part-1-2.png)

**Note:**

* Smaller $\sigma$ $\Leftrightarrow$ more peaked density function.

* Smaller $\sigma$ $\Leftrightarrow$ more spread out density function. 

![png](\assets\images\notes\2022-07-12-the-normal-random-variable-part-1-3.png)

<center><a href="https://deepai.org/machine-learning-glossary-and-terms/probability-density-function">Source</a></center>

$$
  \begin{align}
  P(a\le X\le b)&=\int_{a}^{b}f(x)dx\\
  &=\text{area under the curve from }x=a\text{ to }b
  \end{align}
$$

---

### **Standard Normal Distribution**

**Definition:** The normal distribution with parameter values $\mu=0$ and $\sigma^2=1$ is called the **standard normal** distribution.

An rv with the standard normal distribution is customarily denoted by $Z\sim N(0,1)$ and its pdf is given by

$$
  \color{red}{f_Z(x)=\frac{1}{\sqrt{2\pi}e^{-x^2/2}}}\quad\text{for }-\infty\lt x\lt\infty
$$

We use special notation to denote the cdf of the standard normal curve:

$$
  F(z)=\color{red}{\Phi(z)}=P(Z\le z)=\int_{-\infty}^{z}\frac{1}{\sqrt{2\pi}e^{-x^2/2}}dx
$$

* The standard normal density function is symmetric about
the y axis.

* The standard normal distribution rarely occurs naturally.

* Instead, it is a reference distribution from which
information about other normal distributions can be
obtained via a simple formula.

* The cdf of the standard normal, $\Phi$, can be found in tables
and it can also be computed with a single command in R.

* As we'll see, sums of standard normal random variables
play a large role in statistical analysis.

**Example calculations:**

* $$
    \begin{align}
      P(Z\gt1.25)&=1-P(Z\lt1.25)\\
      &=1 - \Phi(1.25)=0.1056
    \end{align}
  $$

* Why does $P(Z\le-1.25)=P(Z\ge1.25)$? Because of the symmetry.

* $$
    \begin{align}
      P(-0.38\le Z\le1.25)&=P(Z\le1.25)-P(Z\lt-0.38)\\
      &=P(Z\le1.25)-P(Z\le-0.38)\\
      &=\Phi(1.25)-\Phi(-0.38)
    \end{align}
  $$

* $P(-1\le Z\le1)\leftarrow$ probability that $Z$ is within 1 standard deviation of the mean.

  $$
    \begin{align}
      P(-1\le Z\le1)&=P(Z\le1)-P(Z\le-1)\\
      &=\Phi(1)-\Phi(-1)\\
      &=0.6826
    \end{align}
  $$

  This means $68\%$ of $z$ should be within one standard deviation of the mean.

*  $$
    \begin{align}
      P(-2\le Z\le2)&=P(Z\le2)-P(Z\le-2)\\
      &=\Phi(2)-\Phi(-2)\\
      &=0.9544
    \end{align}
  $$

  This means $95\%$ of $z$ should be within one standard deviation of the mean.

**Critical value:** In statistical inference, we need the $z$ values that give certain tail areas under the standard normal curve.

For example, find $z_\alpha$ so that $\Phi(z_\alpha)=P(Z\le z_\alpha)=0.95$

Find $z_\alpha=1.645$

In R:

```R
qnorm(0.95)
```

$$
  P(Z\le1.645)=0.95\\
  P(-1.645\le Z\le1.645)=0.90
$$