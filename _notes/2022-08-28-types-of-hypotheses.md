---
layout: post
comments: false
title: Types of Hypotheses
categories: [Fundamental Concepts of Hypothesis Testing]
---

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

Example of random sample **after** it is observed:

$$2.73, 1.14, 3.98, 2.15, 5.85, 1.97, 2.54, 2.03$$

Example of random sample **before** it is observed:

$$X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8$$

Based on what we are seeing, do we believe that the true population mean $\mu$ is

$$
  \mu\le3 \quad\text{or}\quad\mu\gt3
$$

We have the sample mean is $\overline{x}=2.799$. It's below $3$, can we say that $\mu\lt3$?

This seem awfully dependent on the random sample we happened to get!

Let try to work with the most generic random sample size of $8$:

$$X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8$$

Let $X_1,X_2,...,X_n$ be a random sample of size $n$ from the $N(\mu,\sigma^2)$ distribution.

We say that

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim}N(\mu,\sigma^2)
$$

The <font color='red'>sample mean</font> is 

$$
  \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

* We're going to tend to think that $\mu\lt3$ when $\overline{X}$ is "significant" smaller that 3.

* We're going to tend to think that $\mu\gt3$ when $\overline{X}$ is "significant" larger that 3.

* We're never going to observe $\overline{X}=3, but we may be able to be convinced that $\mu=3$ if $\overline{X}$ is not too far away.

How do we formalize this? We're going to set up **hypotheses**:

$$
  \begin{align}
    &H_0: \mu\le3\quad\text{null hypothesis}\\
    &H_1: \mu\gt3\quad\text{alternate hypothesis}
  \end{align}
$$

* The null hypothesis is assumed to be true.
* The alternate hypothesis is what we are out to show.

Conclusion is either:

$$
  \text{Reject }H_0\quad\text{OR}\quad\text{Fail to reject }H_0
$$

Suppose that $X_1,X_2,...,X_n$ is a random sample from a continous distribution with probability density function (pdf)

$$
  f(x;\theta)=\begin{cases}
    \begin{align}
      &e^{-(x-\theta)}&\quad,x\ge\theta\\
      &0&\quad,x\lt\theta
    \end{align}
  \end{cases}
$$

![png](\assets\images\notes\types-of-hypotheses.png)

It's shifted exponential pdf, and the parameter is unknown. Suppose we want to test these hypotheses:

$$
  H_0:\theta\ge1\quad\text{versus}\quad H_1:\theta\lt1
$$

We can look at minimum of $x$, if the minimum is below $1$, then we know that the null hypothesis is not true, so we should go with the alternate. But if the minimum is a little bit above $1$, the null hypothesis may or may not be true and we can't be sure.

A simple set of hypotheses:

$$
  \left.
  \begin{align}
    H_0:\mu=3\\
    H_1:\mu\gt3
  \end{align}
  \right\}\text{all posibilities in the parameter space}
$$

Suppose we observe

$$
  \overline{x} = -59,349,348
$$

We might be thinking that the mean is not greater than $3$. Then we probably fail to reject $H_0$.

---

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

Suppose that the variance $\sigma^2$ is known.

$$
  H_0:\mu=3
$$

is called a <font color="red">simple</font> hypothesis.

$$
  H_0:\mu\le3
$$

is called a <font color="red">composite</font> hypothesis.

The definition of a simple hypothesis is if you know the hypothesis is true, do you know the exact distribution that your random sample came from? If you do, the it's <font color="red">simple</font>.

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

$$
  H_0:\mu=3
$$

is a <font color="red">composite</font> hypothesis. Because we don't know the variance.