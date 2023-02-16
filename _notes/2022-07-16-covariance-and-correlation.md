---
layout: post
comments: false
title: Covariance and Correlation
categories: [Joint Distributions and Covariance]
---

### **COVARIANCE**

**Example:** An insurance agency services customers who have
both a homeowner's policy and an automobile policy. For each
type of policy, a deductible amount must be specified. For an
automobile policy, the choices are \\\$100 or \\\$250 and for the
homeowner's policy, the choices are \\\$0, \\\$100, or \\\$200.

Suppose the **joint probability table** is given by the insurance company as follows:

![png](\assets\images\notes\2022-07-15-jointly-distributed-random-variables-2.png)

Joint probability mass function: $p(x,y)=P(X=x,Y=y)$.

When two random variable, $X$ and $Y$, are not independent, it is frequently of interest to assess how strongly they are related to each other.

**Definition:** The **covariance** between tow rv's, $X$ and $Y$, is defined as:

<font color='darkred'>
$$
  \begin{align}
    \text{Cov}(X,Y)&=E\big[\big(X-E(X)\big)\big(Y-E(Y)\big)\big]\\  &=E\big[\big(X-\mu_X\big)\big(Y-\mu_Y\big)\big]\\
  \end{align}
$$
</font>

To calculate covariance:

<font color='darkred'>
$$
  \text{Cov}(X,Y)=\begin{cases}
    \begin{align}
      &\sum_{x}\sum_{y}(x-\mu_X)(y-\mu_Y)P(X=x,Y=y)&&\text{(discrete)}\\
      &\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}(x-\mu_X)(y-\mu_Y)f(x,y)dxdy&&\text{(continuous)}\\
    \end{align}
  \end{cases}
$$
</font>


The covariance depends on both the sets of possible pairs and the probabilities for those pairs.

* If both variables tend to deviate in the same direction (both go above their means or below their means at the same time), then the covariance will be positive.

* If the opposite is true, the covariance will be negative.

* If $X$ and $Y$ are not strongly (linearly) related, the covariance will be near $0$.

* Possible to have a strong relationship between $X$ and $Y$ and still have $\text{Cov}(X,Y)\approx0$.

Covariance example calculation:

![png](\assets\images\notes\2022-07-15-jointly-distributed-random-variables-2.png)

| $x$ | $y$ | $x-\mu_X$ | $y-\mu_Y$ | $P(X=x,Y=y)$ |          |
|-----|-----|-----------|----------|--------------|----------|
| 100 | 0   | -75       | -125     | 0.2          | 1875     |
| 250 | 0   | 75        | -125     | 0.05         | -468.75  |
| 100 | 100 | -75       | -25      | 0.1          | 187.5    |
| 250 | 100 | 75        | -25      | 0.15         | -281.25  |
| 100 | 200 | -75       | 75       | 0.2          | -1125    |
| 250 | 200 | 75        | 75       | 0.3          | 1687.5   |
|     |     |           |          |              | **1875** |

$$
  \begin{align}
    \mu_X&=\sum_{x}xP(X=x)\\
    &=100(0.5) + 250(0.5) = 175\\
    \mu_Y&=\sum_{y}yP(Y=y)\\
    &=0(0.25)+100(0.25)+200(0.5)=125\\
  \end{align}
$$

$$
  \text{Cov}(X,Y)=1875
$$

**Computational formula for covariance:**

<font color='darkred'>
$$
  \text{Cov}(X,Y)=E(XY)-E(X)E(Y)
$$
</font>

<u>Proof:</u>

$$
  \begin{align}
    \text{Cov}(X,Y)&=E[(X-E(X)(Y-E(Y)]\\
    &=E[XY-YE(X)-XE(Y)+E(X)E(Y)]\\
    &=E[XY]-E[YE(X)]-E[XE(Y)]+E[E(X)E(Y)]\\
    &=E[XY] - E[X]E[Y]
  \end{align}
$$

<u>What if $X$ and $Y$ are independent?</u>

If $X$ and $Y$ are independent, $P(X=x,Y=y)=P(X=x)P(Y=y)$ for all possible $x,y$.

$$
  \begin{align}
    \text{Cov}&=\sum_{x}\sum_{y}(x-\mu_X)(y-\mu_Y)P(X=x,Y=y)\\
    &=\sum_{x}\sum_{y}(x-\mu_X)(y-\mu_Y)P(X=x)P(Y=y)\\
    &=\bigg[\sum_{x}(x-\mu_X)P(X=x)\bigg]\bigg[\sum_{y}(y-\mu_Y)P(Y=y)\bigg]\\
    &=\bigg[\underbrace{\sum_{x}xP(X=x)}_{E(X)}-\mu_X\underbrace{\sum_{x}P(X=x)}_{1}\bigg]\bigg[\text{similar for $Y$}\bigg]\\
    &=0
  \end{align}
$$

* If $X$ and $Y$ are independent, $\text{Cov}(X,Y)=0$.

* If $\text{Cov}(X,Y)=0$, we cannot conclude $X$ and $Y$ are independent.

<u>Useful formulas for random variables $X$ and $Y$ and real numbers $a$ and $b$:</u>

* $E(aX+bY)=aE(X)+bE(Y)$

* $V(aX+bY)=a^2V(X) + b^2(V(Y) + 2ab\text{Cov}(X,Y)$

  Proof:

  $$
    \begin{align}
      V(aX+bY)&=E\big[\big(aX+bY-E(aX+bY)\big)^2\big]\\
      &=E\big[\big(aX+bY-aE(X)-bE(Y)\big)^2\big]\\
      &=E\big[\big[a(X-E(X))+b(Y-E(Y))\big]^2\big]\\
      &=a^2E[(X-E(X))^2]+b^2E[(Y-E(Y))^2]\\
      &\quad+2abE[(X-E(X))(Y-E(Y))]\\
      &=a^2V(X)+b^2V(Y)+2ab\text{Cov}(X,Y)\\
    \end{align}
  $$

---

### **CORRELATION COEFFICIENT**

**Definition:** The **correlation coefficient** of $X$ and $Y$, denoted by $\text{Cor}(X,Y)$ or just $\rho_{x,y}$, is defined by

<font color='darkred'>
$$
  \rho_{x,y}=\frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$
</font>

It represents a "scale" covariance. The correlation is always between $-1$ and $1$.

<u>Two special cases:</u>

**1.** What if $X$ and $Y$ are independent?

$$
  \text{Cov}(X,Y)=0\quad\text{so}\quad\rho_{x,y}=0
$$

**2.** What if $Y=aX+b$ ($Y$ is a linear function of $X$)

* Find $\text{Cov}(X,Y)$:

  $$
    \begin{align}
      \text{Cov}(X,Y)&=\text{Cov}(X, aX+b)\\
      &=E\big[(X-E(X))(aX+b-E(aX+b))\big]\\
      &=aE[(X-E(X))^2]\\
      &=aV(X)=a\sigma_X^2
    \end{align}
  $$

* Find $V(Y)$:

  $$
    \begin{align}
      V(Y)&=E[(Y-E(Y))^2]\\
      &=E[(ax+b-E(ax+b))^2]\\
      &=a^2V(X)\\
      &=a^2\sigma_X^2
    \end{align}
  $$

  So: $\sigma_Y=\sqrt{a^2\sigma_X^2}=\vert a\vert\sigma_X$

* Find $\rho_{x,y}$:

  $$
    \begin{align}
      \rho_{x,y}&=\frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}\\
      &=\frac{a.\sigma_X^2}{\sigma_X.\vert a\vert.\sigma_X}\\
      &\begin{cases}
        &1&&\text{if }a\gt0\\
        &-1&&\text{if }a\lt0\\
      \end{cases}
    \end{align}
  $$

**Example:**

From previous example:

![png](\assets\images\notes\2022-07-15-jointly-distributed-random-variables-2.png)

Find $\rho_{x,y}$

We have:

$$
  \text{Cov}(X,Y)=1875\\
  E(X)=175\\
  E(Y)=125\\
$$

$$
  \begin{align}
    V(X)&=E(X^2)-(E(X))^2\\
    &=5625\\
    V(Y)&=E(Y^2)-(E(Y))^2\\
    &=6875\\
  \end{align}
$$

$$
  \rho_{x,y}=\frac{1875}{\sqrt{5625}\sqrt{6875}}\approx0.3
$$