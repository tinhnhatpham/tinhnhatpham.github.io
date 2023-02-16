---
layout: post
comments: false
title: Test Statistics and Significance
categories: [Fundamental Concepts of Hypothesis Testing]
---

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Consider testing the simple versus simple hypotheses

$$
  H_0:\mu=3\\
  H_1:\mu=5
$$

<u>Definition/Notation:</u>

$$
  \begin{align}
    \text{Let}\quad&=P(\text{Type I Error})\\
    &=P(\text{Reject $H_0$ when it's true})\\
    &=P(\text{Reject $H_0$ when $\mu=3$})
  \end{align}
$$

$\alpha$ is called the <font color='red'><b>level of significance</b></font> of the test.

It is also sometimes referred to as the <font color='red'><b>size</b></font> of the test.

<u>Developing a Test</u>

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

The form of the test:

* We are looking for evidence that $H_1$ is true.
* The $N(3,/sigma^2)$ takes on values from $-\infty$ to $\infty$.
* $\overline{X}\sim N(\mu,\sigma^2)\implies\overline{X}$ also takes on values from $-\infty$ to $\infty$.
* It is entirely possible that $\overline{X}$ is very large even if the mean of its distribution is $3$.
* However, if $\overline{X}$ is very large, it will start to seem more likely that $\mu$ is larger than $3$.
* Eventually, a population mean of $5$ will seem more likely than a population mean of $3$.

Give the "form" of test.

> Reject $H_0$, in favor of $H_1$, if $\overline{X}\gt c$ for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

> Reject $H_0$, in favor of $H_1$, if $\overline{X}\gt c$.

* If $c$ is too large, we are making it difficult to reject $H_0$.
  
  We are more likely to fail to reject when it should be rejected.

  This is <font color='red'><b>Type II Error</b></font>.

* If $c$ is too small, we are making it too easy to reject $H_0$.

  We are more like reject when it should not be rejected.

  This is <font color='red'><b>Type I Error</b></font>.

This is where $\alpha$ comes in.

$$
  \begin{align}
    \alpha&=P(\text{Type I Error})\\
    &=P(\text{Reject $H_0$ when true})\\
    &=P(\overline{X}\gt c\quad\text{when}\quad\mu=3)
  \end{align}
  
$$

<font color='blue'><b>Step Four:</b></font>

Give a conclusion!

---

### **Example**

$$
  X_1,X_2,...,X_{10}\stackrel{iid}{\sim}N(\mu,4)
$$

Find a hypothesis test for

$$
  H_0:\mu=5\quad\text{vs}\quad H_1:\mu=3
$$

Use level of significance $\alpha=0.05$.

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$, if $\overline{X}\lt c$ for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

$$
  \begin{align}
    0.05&=P(\text{Type I Error})\\
    &=P(\text{Reject $H_0$ when true})\\
    &=P(\overline{X}\lt c\text{ when }\mu=5)\\
    &=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\lt\frac{c-5}{2/\sqrt{10}}\text{ when }\mu=5\Bigg)
  \end{align}
$$

We have

$$
  0.05=P\Bigg(Z\lt\frac{c-5}{2/\sqrt{10}}\Bigg)\\
  \text{cut off point = pnorm(0.05)=-1.645}\\
  \begin{align}
    &\implies \frac{c-5}{2/\sqrt{10}} = -1.645\\
    &\implies c=3.9596
  \end{align}
$$

<font color='blue'><b>Step Four:</b></font>

Give a conclusion.

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}\lt3.9596
$$

![png](\assets\images\notes\test-statistics-and-significance.png)