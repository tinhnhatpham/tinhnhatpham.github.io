---
layout: post
comments: false
title: Normal Computations
categories: [Fundamental Concepts of Hypothesis Testing]
---

**Notation/Terminology:**

Random Sample

$$
  X_1,X_2,...,X_n
$$

* Variables before they are sampled, observed, and "locked in".
* They are assumed to be <font color='red'>independent and identically distributed</font> (<font color='green'>iid</font>).

$$
  \text{Random Sample} = \text{iid}
$$

**More Notation:**

Suppose that $X_1,X_2,...,X_n$ is a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

We write

$$
  X_1,X_2,...,X_n\stackrel{iid}{\sim}N(\mu,\sigma^2)\\
  E[X_i]=\mu\qquad Var[X_i] = \sigma^2
$$

The pdf:

$$
  f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2\sigma^2}(x-\mu)^2}
$$

The expected value:

$$
  \begin{align}
    \mu=E[X_i]&=\int_{-\infty}^{\infty}xf(x)dx\\
    E[X_i^2]&=\int_{-\infty}^{\infty}x^2f(x)dx
  \end{align}
$$

The variance:

$$
  \begin{align}
    \sigma^2&=Var[X_i]=E[(X_i-\mu)^2]\\
    &=E[X_i^2] - (E[X_i])^2
  \end{align}
$$

Any linear combination of normal random variables has, again, a normal distribution.

$$
  \begin{align}
  &a_1X_1+a_2X_2+...+a_nX_n\sim N(?,?)\\
  &E\bigg[\sum_{i=1}^{n}a_iX_i\bigg]=\sum_{i=1}^{n}a_i\underbrace{E[X_i]}_{\mu}=\mu\sum_{i=1}^{n}a_i\\
  &Var\bigg[\sum_{i=1}^{n}a_iX_i\bigg]\stackrel{indep}{=}\sum_{i=1}^{n}a_i^2\underbrace{Var[X_i]}_{\sigma^2}=\sigma^2\sum_{i=1}^{n}a_i^2
  \end{align}
$$

In particular, if 

$$
  X_1,X_2,...,X_n\stackrel{iid}{\sim}N(\mu,\sigma^2)
$$

Then 

$$
  \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\sim N\bigg(\mu,\frac{\sigma^2}{n}\bigg)
$$

Note that the bigger the sample, the less variability we're going to see.

---

The $N(0,1)$ distribution is known as the <font color='red'>standard normal distribution</font>.

We typically use the letter $Z$:

$$
  Z\sim N(0,1)
$$

The <font color='red'>cumulative distribution function</font> (cdf)

$$
  \begin{align}
    \phi(z) &= P(Z\le z)\\
    &=\int_{-\infty}^{z}\frac{1}{2\pi}e^{-\frac{1}{2}x^2}dx
  \end{align}
$$

![png](\assets\images\notes\normal-computations.png)

* $X\sim N(\mu,\sigma^2)\implies\frac{X-\mu}{\sigma}\sim N(0,1)$
* $Z\sim N(0,1)\implies\sigma Z+\mu\sim N(\mu,\sigma^2)$

<font color='blue'>Example 1</font>

Let $X\sim N(2,3)$.

Then

$$
  \begin{align}
    P(X\le4.1)&=P\Bigg(\frac{X-\mu}{\sigma}\le\frac{4.1-2}{\sqrt{3}}\Bigg)\\
    &=P(Z\le1.21)\\
    \approx0.8868
  \end{align}
$$

We can use R: ```pnorm(1.21)```

<font color='blue'>Example 2</font>

$X_1,X_2,...,X_10\stackrel{iid}{\sim}N(2,3)$

$\overline{X}\sim N(\mu,\sigma^2/n)=N(2,3/10)$

$$
  \begin{align}
    P(\overline{X}\le2.3) &= P\Bigg(\frac{\overline{X}-\mu_{\overline{X}}}{\sigma_\overline{X}}\le\frac{2.3-2}{\sqrt{3/10}}\Bigg)\\
    &=P(Z\le0.5477)\\
    &\approx0.7081
  \end{align}
$$

---

**Critical Values**

* Values that cut off specified areas under pdfs.
* For the N(0,1) distribution, we will use the notation <font color='red'>$Z_\alpha$</font>

<font color='blue'>Example</font>

![png](\assets\images\notes\normal-computations-1.png)

We can use R to compute the critical value: ```qnorm(0.95)```
