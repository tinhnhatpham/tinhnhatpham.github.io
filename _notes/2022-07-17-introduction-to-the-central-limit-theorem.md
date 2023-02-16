---
layout: post
comments: false
title: Introduction to the Central Limit Theorem
categories: [Central Limit Theorem]
---

For a random variable $X$, we need either the probability mass function $p(k)=P(X=k)$ or density function $f(x)$ to compute a probability or to find

* $\mu_X=E(X)=\sum_{k}kP(X=k)$ or $\mu_X=\int_{-\infty}^{\infty}xf(x)dx$

* $\sigma_X^2=V(X)=E[(X-\mu_X)^2]=\sum_{k}(k-\mu_X)^2P(X=k)$ or

  $\sigma_X^2=\int_{-\infty}^{\infty}(x-\mu_X)^2f(x)dx$

Question: What if we don't know how a random variable is distributed? What if we don't know the mean or the variance?

**Definition:** $X_1,X_2,\ldots,X_n$ are a **random sample** of size $n$ if

* $X_1,X_2,\ldots,X_n$ are independent

* Each radom variable has the same distribution

We say that these $X_i$'s are $iid$, independent and identically distributed.

We use **estimators** to summarize our iid sample. For example, suppose we want to understand the distribution of adult female heights in a certain area. We plan to select $n$ women at random and measure their height. Suppose the height of the $i^{th}$ woman is denoted by $X_i$. $X_1,X_2,\ldots X_n$ are iid with mean $\mu$.

An **estimator** of $\mu$ is denoted $\overline{X}$ and 

$$
  \overline{X}=\frac{1}{n}\sum_{k=1}^{n}X_k
$$

<u>Recall:</u> $E(aX+bY)=aE(X)+bE(Y)$

$$
  \begin{align}
    E(\overline{X})&=E\bigg(\frac{1}{n}\sum_{k=1}^{n}X_k\bigg)\\
    &=\frac{1}{n}E\bigg(\sum_{k=1}^{n}X_k\bigg)\\
    &=\frac{1}{n}\sum_{k=1}^{n}E(X_k)\\
    &=\frac{1}{n}\sum_{k=1}^{n}\mu=\mu
  \end{align}
$$

The Law of Large Numbers says that under most conditions, if $X_1,X_2,\ldots,X_n$ is a random sample with $E(X_k)=\mu$, then $\overline{X}=\frac{1}{n}\sum_{k=1}^{n}X_k$, converges to $\mu$ in the limit as $n$ goes to infinity.

---

What about the variance? Given a random sample $X_1,X_2,...,X_n$ with $V(X_i)=\sigma^2$.

<u>Recall:</u> 

  $$
    V(aX+bY)=a^2V(X)+b^2V(Y)+2ab\text{Cov}(X,Y)
  $$

* If $X$ and $Y$ are independent, $\text{Cov}(X,Y)=0$. So

  $$
    V(aX+bY)=a^2V(X)+b^2V(Y)
  $$

$$
  \begin{align}
    V(\overline{X})&=V(\frac{1}{n}\sum_{k=1}^{n}X_k)\\
    &=\frac{1}{n^2}V(\sum_{k=1}^{n}X_k)=\frac{1}{n^2}\sum_{k=1}^{n}V(X_k)\\
    &=\frac{1}{n^2}\sum_{k=1}^{n}\sigma^2=\frac{1}{n^2}n\sigma^2\\
    &=\frac{\sigma^2}{n}
  \end{align}
$$

---

We use estimators to summarize our iid sample. Any estimator, including the sample mean, $\overline{X}$, is a random variable (since it is based on a random sample).

This means that $\overline{X}$ has a distribution of it's own, which is referred as the **sampling distribution of the sample mean**. This sampling distribution depends on:

* The sample size $n$.

* The population distribution of the $X_i$.

* The method of sampling.

We know $E(\overline{X})=\mu$ and $V(\overline{X})=\sigma^2/n$. But we don't know in general, the distribution of $\overline{X}$.


Proposition: if $X_1,X_2,...,X_n$ is iid with $X_i\sim N(\mu,\sigma^2)$ then 

$$
  \color{red}{\overline{X}\sim N\bigg(\mu,\frac{\sigma^2}{n}\bigg)}
$$

![png](\assets\images\notes\2022-07-17-introduction-to-the-central-limit-theorem.png)

We know everything there is to know about the distribution of
the sample mean when the population distribution is normal.

What if the population distribution is not normal?

* When the population distribution is non-normal,
averaging produces a distribution that is more bell-shaped
than the one being sampled.

* A reasonable conjecture is that if n is large, a suitable
normal curve will approximate the actual distribution of
the sample mean.

* The formal statement of this result is one of the most
important theorems in probability and statistics: **Central Limit Theorem**.

**Central Limit Theorem** Let $X_1,X_2,...,X_n$ be a random sample with $E(X_i)=\mu$ and $V(X_i)=\sigma^2$. If $n$ is sufficiently large, $\overline{X}$ has approximately a normal distribution with mean $\mu_{\overline{X}}=\mu$ and variance $\sigma_{\overline{X}}^2=\sigma^2/n$. We write

$$
  \overline{X}\approx N\bigg(\mu,\frac{\sigma^2}{n}\bigg)
$$

The larger the value of $n$, the better the approximation.

Typical rule of thumb: $\color{red}{n\ge30}$