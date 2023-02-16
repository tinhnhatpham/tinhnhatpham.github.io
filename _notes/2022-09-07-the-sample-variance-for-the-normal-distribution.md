---
layout: post
comments: false
title: The Sample Variance for the Normal Distribution
categories: [t-Tests and Two-Sample Tests]
---

Let $X_1,X_2,...,X_n$ be a random sample from any distribution with mean $\mu$ and variance $\sigma^2$.

$$
  \sigma^2=Var[X]:=E[(X-\mu)^2]
$$

To estimate this from the sample, we could use

$$
  \stackrel{\sim}{S^2}=\frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{n}\\
$$

* Currently, before numerical observations, this is a random variable.

* It has its own distribution, its own mean, and its own variance.

$$
  \begin{align}
    \sum_{i=1}^{n}(X_i-\overline{X})^2&=\sum_{i=1}^{n}(X_i^2-2\overline{X}X_i+\overline{X}^2)\\
    &=\sum_{i=1}^{n}X_i^2-2\overline{X}\underbrace{\sum_{i=1}^{n}X_i}_{n\overline{X}}+n\overline{X}^2\\
    &=\sum_{i=1}^{n}X_i^2-n\overline{X}^2\\
    &=\sum_{i=1}^{n}X_i^2-n\Bigg(\sum_{i=1}^{n}X_i\bigg/n\Bigg)^2\\
    &=\sum_{i=1}^{n}X_i^2-\frac{\bigg(\sum_{i=1}^{n}X_i\bigg)^2}{n}\\
    &\text{(computationally computation)}\\
  \end{align}
$$

To find the expected value of $X^2$, we can use the definition of variance

$$
  Var[X]=E[(X-\mu)^2]=E[X^2]-(E[X])^2\\
  \Downarrow\\
  \begin{align}
    E[X^2]&=Var[X]+(E[X])^2\\
    &=\sigma^2+\mu^2\\
  \end{align}\\
  \Downarrow\\
  E[\stackrel{\sim}{S^2}]=\frac{n-1}{n}\sigma^2\\
  \text{(Biased variance)}
$$

Variant of Sample Variance:

<font color='red'>
$$
  S^2=\frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{n-1}
$$
</font>

We can observe that

$$
  S^2=\frac{n}{n-1}\stackrel{\sim}{S^2}\\
  \begin{align}
    \implies E[S^2]&=\frac{n}{n-1}E[\stackrel{\sim}{S^2}]\\
    &=\frac{n}{n-1}\frac{n-1}{n}\sigma^2=\sigma^2\\
    &\text{(Unbiased estimator)}
  \end{align}
$$

For the normal distribution, these two below are independent:

$$
  \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\\
  S^2=\frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{n-1}
$$

<u>Aside:</u>

$$
  X_1\sim\chi^2(n_1)\text{ and }X_2\sim\chi^2(2_1)\\
  \text{independent}\\
  \Downarrow\\
  X_1+X_2\sim\chi^2(n_1+n_2)
$$

And

$$
  X_1\sim\chi^2(n_1)\text{ and }X_2\sim\chi^2(2_1)\\
  X_3\sim?\\
  X_1=X_2+X_3\\
  \text{and $X_2$ and $X_3$ independent}\\
  \Downarrow\\
  X_3=X_1-X_2\sim\chi^2(n_1-n_2)\\
$$

Let $X_1,X_2,...,X_n$ be a random sample from the <font color='red'>normal</font> distribution with mean $\mu$ and variance $\sigma^2$.

$$
  \begin{align}
    &\sum_{i=1}^{n}(X_i-\mu)^2\\
    &=\sum_{i=1}^{n}(X_i-\overline{X}+\overline{X}-\mu)^2\\
    &=\sum_{i=1}^{n}(X_i-\overline{X})^2\\
    &\qquad+2(\overline{X}-\mu)\underbrace{\sum_{i=1}^{n}(X_i-\overline{X}}_{0})\\
    &\qquad+n(\overline{X}-\mu)^2\\
    &=\sum_{i=1}^{n}(X_i-\overline{X})^2+n(\overline{X}-\mu)^2\\
  \end{align}
$$

We divide the term above to $\sigma^2$

$$
  \begin{align}
    &\underbrace{\frac{\sum_{i=1}^{n}(X_i-\mu)^2}{\sigma^2}}_{1}\\
    &\qquad=\underbrace{\frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{\sigma^2}}_{2}+\underbrace{\frac{n(\overline{X}-\mu)^2}{\sigma^2}}_{3}
  \end{align}
$$

* (1)

  $$
    \frac{\sum_{i=1}^{n}(X_i-\mu)^2}{\sigma^2}=\underbrace{\sum_{i=1}^{n}\Bigg(\underbrace{\frac{X_i-\mu}{\sigma}}_{N(0,1)}\Bigg)^2}_{\chi^2(n)}
  $$

* (3)

  $$
    \frac{n(\overline{X}-\mu)}{\sigma^2}=\underbrace{\Bigg(\underbrace{\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}}_{N(0,1)}\Bigg)^2}_{\chi^2(1)}
  $$

* (2)

  $$
    \frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{\sigma^2}=\frac{(n-1)S^2}{\sigma^2}
  $$

We have

$$
  \underbrace{W_1}_{\chi^2(n)}=\overbrace{W_2+\underbrace{W_3}_{\chi^2(1)}}^{\text{independent}}
$$

We can say

$$
  \implies W_2=\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$