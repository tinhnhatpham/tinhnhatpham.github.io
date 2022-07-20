---
layout: post
title: Introduction to The Central Limit Theorem
categories: [Probability]
---

For a random variable $X$, we need either PMF $p(k) = P(X = k)$ or PDF $f(x)$ to compute probability or to find:
* $\mu_X = E(X) = \sum_{k}kP(X=k)$

  $\mu_X = \int_{-\infty}^{\infty}xf(x)dx$

* $\sigma_X^2 = V(X) = E[(X-\mu_X)^2] = \sum_k(k-\mu_X)^2P(X = k)$

  $\sigma_X^2 = \int_{-\infty}^{\infty}(x - \mu_X)^2f(x)dx$

**Definition:** $X_1$, $X_2$,..., $X_n$ are a **random sample** of size $n$ if:
* $X_1$, $X_2$,..., $X_n$ are independent.
* Each random variable has the same distribution

We say that these $X_i$'s are **iid**, independent and identically distributed.

We use **estimators** to summarize our **iid** sapmle. An **estimator** of $\mu$ is denoted $\bar{X}$ and $\bar{X} = \frac{1}{n}\sum_{k=1}^{n}X_k$. Recall, $E(aX + bY) = aE(X) + bE(Y)$:

$$
\begin{align}
  E(\bar{X}) &= E(\frac{1}{n}\sum_{k=1}^nX_k) \\
  &= \frac{1}{n}E(\sum_{k=1}^nX_k) \\
  &= \frac{1}{n}\sum_{k=1}^nE(X_k) \\
  &= \frac{1}{n}\sum_{k=1}^n\mu = \mu
\end{align}
$$

The **Law of Large Numbers** is fairly techical. However, it says that under most conditions, if $X_1$, $X_2$,..., $X_n$ is a random sample with $E(X_k)=\mu$, then $\bar{X}=\frac{1}{n}\sum_{k=1}^{n}X_k$,converges to $\mu$ in the limit as $n$ goes to infinity.

**Variance**
$X_1$, $X_2$,..., $X_n$ with $V(X_i) = \sigma^2$
* Recall: $V(aX + bY) = a^2V(X) + b^2V(Y) + 2abCov(X,Y)$, if $X$ and $Y$ are independent, $Cov(X,Y) = 0$. So $V(aX + bY) = a^2V(X) + b^2V(Y)$.

$$
\begin{align}
  V(\bar{X}) &= V(\frac{1}{n}\sum_{k=1}^{n}X_k) \\
  &= \frac{1}{n^2}V(\sum_{k=1}^{n}X_k) \\
  &= \frac{1}{n^2}\sum_{k=1}^{n}V(X_k) \\
  &= \frac{1}{n^2}\sum_{k=1}^{n}\sigma^2 \\
  &= \frac{1}{n^2}n\sigma^2 \\
  &= \frac{\sigma^2}{n}
\end{align}
$$

We use estimators to summarize our iid sample. Any estimator, including the sample mean, $\bar{X}$, is a random variable (since it is based on a random sample).

This means that $\bar{X}$ has a distribution of it's own, which is referred to as the **sampling distribution of the sample mean**. This sampling distribution depends on:
* The sample size $n$.
* The population distribution of the $X$.
* The method of sampling.

![Image](/assets/images/sample-means-normal-population.jpg "Title")

*[Image Source](https://saylordotorg.github.io/text_introductory-statistics/section_10/86efb32d8f607d8fc448b44f66c8a4c7.jpg)*

What if the population distribution is not normal?
* When the population distribution is non-normal, averaging produces a distribution that is more bell-shaped than the one being sampled.
* A reasonable conjecture is that if $n$ is large, a suitable normal curve will approximate the actual distribution of the sample mean.
* The formal statement of this result is one of the most important theorems in probability and statistics: **Central Limit Theorem**

**Central Limit Theorem:** Let $X_1$, $X_2$,..., $X_n$ is a random sample with $E(X_i) = \mu$ and $V(X_i) = \sigma^2$. If $n$ is sufficiently large, $\bar{X}$ has approximately a normal distribution with mean $\mu_\bar{X} = \mu$ and variance $\sigma^2_\bar{X} = \sigma^2/n$.

We write $\bar{X} \approx N(\mu, \frac{\sigma^2}{n}$

The larger the value $n$, the better the approximation. Typical rule of thumb: $n \ge 30$.


