---
layout: post
title: Covariance and Correlation Coefficient
categories: [Probability]
---
### Covariance
When two random variables, $X$ and $Y$, are not independent, it is frequently of interest to assess how strongly they are related to each other.

**Definition** Covariance of $X$ and $Y$ is given by:

$Cov(X,Y) = E[(X - E(X))(Y - E(Y)]$

To calculate covariance:

$$
Cov(X,Y)=\begin{cases}
  \sum_{x}\sum_{y}(x-\mu_x)(y-\mu_y)P(X=x,Y=y)\\\\
  \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}(x-\mu_x)(y-\mu_y)f(x,y)dxdy
\end{cases}
$$

The covariance depends on both set of possible pairs and the probabilities for those pair.

**Notes:**
* If both variables tend to deviate in the same direction (both go above or below their means at the same time), then the covariance will be positive.
* If the opposite is true, the covariance will be negative.
* If $X$ and $Y$ are not strongly (linearly) related, the covariance will be near 0. There is possible to have a strong relationship between $X$ and $Y$ and still have $Cov(X,Y) \approx 0$.

**Computational formula for covariance:**

$$Cov(X,Y) = E(XY) - E(X)E(Y)$$

**What if $X$ and $Y$ are independent?**
* If $X$ and $Y$ are **independent**, $Cov(X,Y) = 0$.
* If $Cov(X,Y) = 0$, we cannot conclude $X$ and $Y$ are **independent**.

**Useful fomulas:**

$E(aX + bY) = aE(X) + bE(Y)$

$V(aX + bY) = a^2V(X) + b^2V(Y) + 2abCov(X,Y)$

---

### Correlation Coeficient
**Definition:** The **correlation coefficient** of $X$ and $Y$, denoted by $Cov(X,Y)$ or $\rho_{x,y}$, is defined by:

$$
\rho_{x,y} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}
$$

It represents a "scaled" covariance. The correlation is always between $-1$ and $1$.

**Two special cases:**

1.   What if $X$ and $Y$ are independent?

$$
  Cov(X,Y) = 0 \quad \text{so} \quad \rho_{x,y} = 0
$$

2.   What if $Y = aX + b$? ($Y$ is a linear function of $X$)

$$
\begin{align}
  Cov(X,Y) &= Cov(X, aX + b) \\
  &= E[(X - E(X))(aX + b - E(aX + b))] \\
  &= aE[(X - E(X))^2] \\
  &= aV(X) \\
  &= a\sigma_x^2
\end{align}
$$

$$
\begin{align}
  V(Y) &= E[(Y - E(Y))^2] \\
  &= E[(aX - b - E(aX + b))^2] \\
  &= a^2V(X) \\
  &= a^2\sigma_x^2 \\
  \sigma_y &= \sqrt{a^2\sigma_x^2} \\
  &= \mid a \mid \sigma_x
\end{align}
$$

$$
\begin{align}
  \rho_{x,y} & = \frac{a\sigma_x^2}{\sigma_x\mid a \mid \sigma_x} \\
  &= \begin{cases}
    1 \quad \text{if} \quad a \gt 0\\
    -1 \quad \text{if} \quad a \lt 0\\
  \end{cases}
\end{align}
$$




