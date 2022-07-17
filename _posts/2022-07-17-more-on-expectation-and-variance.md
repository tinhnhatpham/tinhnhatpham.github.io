---
layout: post
title: More on Expectation and Variance
categories: [Probability]
---

### Expectation

$E(X) = \sum_{k}kP(X=k)$ if $X$ is discrete

$E(X) = \int_{-\infty}^{\infty}xf(x)dx$ if $X$ is continous

**What can we say about $E(g(x))$?**

$$
  E(g(x))=\begin{cases}
    \sum_{k}g(k)P(X=k) & \text{if $X$ is discrete} \\\\
    \int_{-\infty}^{\infty}g(x)f(x)dx & \text{if $X$ is continous}
  \end{cases}
$$

$
E(aX + b) = \sum_{k}(ak + b)P(X=k) \\
E(aX + b) = a\sum_{k}kP(X=k) + b\sum_{k}P(X=k) = aE(X) + b
$

### Variance

$\sigma^2 = V(X) = E[(X-\mu)^2] = E(X^2) - (E(X))^2$
* $V(X) = \sum_{k}(k-\mu)^2P(X=k)$ if $X$ is discrete
* $V(X) = \int_{-\infty}^{\infty}(x-\mu)^2f(x)dx$ if $X$ is continous

**What about $V(g(X))$?**

$$
  V(g(x))=\begin{cases}
    \sum_{k}(g(k) - E(g(x)))^2P(X=k) & \text{if $X$ is discrete}\\\\
    \int_{-\infty}^{\infty}(g(k) - E(g(x)))^2f(x)dx & \text{if $X$ is continous}
  \end{cases}
$$

**Find** $V(aX + b)$:

$V(aX + b) = E[(aX + b - E(aX + b))^2]$

$V(aX + b) = E[(aX + b - aE(X) - b)^2]$

$V(aX + b) = E[a^2(X - E(X))^2]$

$V(aX + b) = a^2E[(X - E(X))^2] = a^2V(X)$


**Recall:** $E(aX + b) = aE(X) + b$

Variance measures spread of the data, the $b$ shift the data (mean) but doesn't affect the spread.
