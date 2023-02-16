---
layout: post
comments: false
title: Uniformly Most Powerful Tests
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Derive a <font color='red'><b>uniformly most powerful</b></font> hypothesis test of size $\alpha$ for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\gt\lambda_0
$$

---

The <font color='red'><b>uniformly most powerful</b></font> testof size $\alpha$ for testing

$$
  H_0:\lambda\in\Theta_0\quad\text{vs.}\quad H_1:\lambda\in\Theta\backslash\Theta_0
$$

is a test defined by a rejection region $R^*$ such that

1. It has size $\alpha$.

  i.e. $\underset{\theta\in\Theta_0}{\text{max}}P(\stackrel{\rightharpoonup}{X}\in R^*;\theta)=\alpha$

2. It has higher power for all $\theta\in\Theta\backslash\Theta_0$

  i.e. $\gamma_{R^*}(\theta)\ge\gamma_{R}(\theta)$ for all $\theta\in\Theta\backslash\Theta_0$.

  i.e. $P(\stackrel{\rightharpoonup}{X}\in R^*;\theta)\ge P(\stackrel{\rightharpoonup}{X}\in R;\theta)$ for all $\theta\in\Theta\backslash\Theta_0$.

---

<font color='blue'><b>Step One:</b></font>

Consider the simple versus simple hypothesis

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda=\lambda_1
$$

for some fixed $\lambda_1\gt\lambda_0$.

<font color='blue'><b>Step Two, Three, and Four:</b></font>

Find the best test of size $\alpha$ for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda=\lambda_1
$$

for some fixed $\lambda_1\gt\lambda_0$.

This test is to reject $H_0$, in favor of $H_1$ if 

$$
  \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

Note that this test does not depend on the particular value of $\lambda_1$. <font color='red'><b>It does, however, depend on the fact that $\lambda_1\gt\lambda_0$</b></font>.

It is the uniformly most powerful (best) test for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\gt\lambda_0
$$

---

Suppose we've looked at a different hypotheses

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\lt\lambda_0
$$

Reject $H_0$ in favor of $H_1$, if

$$
  \bigg(\frac{\lambda_0}{\lambda_1}\bigg)^ne^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}X_i}\le c\\
  \huge{.}\\
  \huge{.}\\
  \huge{.}\\
  -(\lambda_0-\lambda_1)\sum_{i=1}^{n}X_i\le c\\
  \sum_{i=1}^{n}X_i\color{red}{\ge} c\\
  (\color{red}{\text{if }\lambda_1\lt\lambda_0})\\
$$

---

The "UMP" test for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\gt\lambda_0
$$

is to reject $H_0$, in favor of $H_1$ if

$$
  \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

<hr size="2" color="yellow" width="70%">

The "UMP" test for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\lt\lambda_0
$$

is to reject $H_0$, in favor of $H_1$ if

$$
  \overline{X}\gt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

---

Does there exist a "UMP" test for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\neq\lambda_0
$$

In this case, the answer is <font color='red'><b>No</b></font>

For any $\lambda_1\neq\lambda_0$,

* The best test if $\lambda_1\gt\lambda_0$ is to reject $H_0$ if

  $$
    \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
  $$

* The best test if $\lambda_1\lt\lambda_0$ is to reject $H_0$ if

  $$
    \overline{X}\gt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
  $$

There is no one best test that we can use for all $\lambda_1\neq\lambda_0$!