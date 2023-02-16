---
layout: post
comments: false
title: Continuous Random Variables
categories: [Continuous Random Variables]
---

**Definition:** A random variable is **continuous** if possible values comprise either a single interval on the number line or a union of disjoint intervals.

<u>Examples:</u>

* In the study of the ecology of a lake, a rv $X$ could be the depth measurements at a random chosen locations

  $X\in[0,\text{maximum depth of lake}]$.

* In a study of a chemical reaction, $Y$ could be the concentration level of a particular chemical in solution.

* In a study of customer service, $W$ could be the time a customer waits for service.

**Note:** If $X$ is continuous, $P(X=x)=0$ for any $x$!

<u>Motivation example:</u> Suppose a train is scheduled to arrive at 1 pm. Let $X$ be the minutes past the hour that it arrives and $X\in\{0,1,2,3,4,5\}$.

| $x$    | 0   | 1    | 2   | 3    | 4    | 5    |
|--------|-----|------|-----|------|------|------|
| $p(x)$ | 0.1 | 0.15 | 0.3 | 0.25 | 0.15 | 0.05 |

$$
  X\in[0,5]\\
  P(1.5\lt X\lt 2.5)=\int_{1.5}^{2.5}f(x)d(x)
$$

$y = f(x)\leftarrow$ the probability density function for the continuous rv $X$.

$$
  P(a\le X\le b)=\int_a^b f(x)dx
$$

---

<font color='green'><b>Properties of the probability density function</b></font>

For any continuous rv $X$ with probability density function (pdf) $f$ we have:

* The probability density function $f:(-\infty,\infty)\rightarrow[0,\infty)$ so $f(x)\ge0$.

* $P(-\infty\lt X\lt\infty)=\int_{-\infty}^{\infty}f(x)dx=1\quad(=P(S))$.

* $P(a\le X\le b)=\int_{a}^{b}f(x)dx$

**Note:** $P(X=a)=\int_a^a f(x)dx = 0$ for all real numbers $a$.

---

<font color='green'><b>Cumulative Distribution Function</b></font>

**Definition:** The cumulative distribution function (cdf) for a continuous rv $X$ is given by 

$$
  F(x) = P(X\le x)=\int_{-\infty}^{x}f(t)dt
$$

* $0\le F(x)\le 1$

* $\lim_{x \to -\infty}F(x)=0$ and $\lim_{x \to \infty}F(x)=1$

* $F'(x)=f(x)$ by fundamental theorem of Calculus.

* $F(x)$ is always increasing.

---

<font color='green'><b>Uniform Random Variable</b></font>

**Definition:** A random variable $X$ has the **uniform distribution** on the interval $[a,b]$ if its density function is given by

$$
  f(x)=\begin{cases}
    \begin{align}
      &\frac{1}{b-a}&&\text{if }a\le x \le b\\
      &0&&\text{else}
    \end{align}
  \end{cases}
$$

Notation: $X\sim U[a,b]$

**Cumulative Distribution Function:**

$$
  \begin{align}
    F(x)&=P(X\le x)\\
    &=\int_{-\infty}^{x}f(t)dt\\
    &=\int_{a}^{x}\frac{1}{b-a}dt,&&a\lt x\lt b
  \end{align}
$$

Solve integral, we have

$$
  P(X\le x)=
  \begin{cases}
    \begin{align}
      &0&&\text{if }x\lt a\\
      &\frac{x-a}{b-a}&&\text{if }a\le x\le b\\
      &1&&\text{if }b\lt x
    \end{align}
  \end{cases}
$$

Example 1: Random number generators select numbers uniformly from a  specific interval, usually $[0,1]$.

Example 2: Suppose the diameter of aerosol particles in a particular application is uniformly distributed between 2 and 6 nanometers. Find the probability that a random measured particle has diameter greater that 3 nanometers.

$X\sim U[2,6]$

$$
  f(x)=\begin{cases}
    \begin{align}
      &\frac{1}{4}&&\text{for }2\le x\le6\\
      &0&&\text{else}
    \end{align}
  \end{cases}
$$

$$
  \begin{align}
    P(X\ge3)= 1-P(X\lt3)&=\int_2^3\frac{1}{4}dt\\
    &=\frac{3}{4}\\
  \end{align}
$$

Or

$$
  P(X\ge3)=\int_3^6\frac{1}{4}dt=\frac{3}{4}
$$

Example 3: You throw a dart at a dashboard. The radial distance of the dart from the x-axis can be modeled by a uniform random variable.

$$
  Y\sim U[0,360]
$$

---

<font color='green'><b>Expectation and Variance for a continuous rv $X$</b></font>

Recall: $Y$ is discrete

$$
  E(Y)=\sum_k kP(Y=k)\quad V(Y)=\sum_k(k-\mu_Y)^2P(Y=k)
$$

If $X$ is continuous RV with pdf $f(x)$

**Expected Value**

$$
  \color{red}{E(X) = \int_{-\infty}^{\infty}xf(x)dx}
$$

**Variance**

$$
  \begin{align}
    \color{red}{V(X)}&=\int_{-\infty}^{\infty}(x-\mu_X)^2f(x)dx\\
    &=\int_{-\infty}^{\infty}(x^2-2\mu_X x+\mu_X^2)f(x)dx\\
    &=\underbrace{\int_{-\infty}^{\infty}x^2f(x)dx}_{E(X^2)}\\
    &\qquad-2\mu_X x\underbrace{\int_{-\infty}^{\infty}xf(x)dx}_{E(X)}\\
    &\qquad+\mu_X^2\underbrace{\int_{-\infty}^{\infty}f(x)dx}_{1}\\
    &=\color{red}{E(X^2)-(E(X))^2}
  \end{align}
$$

---

Compute expectation and variance for $X\sim U[a,b]$

$$
  f(x)=\begin{cases}
    \begin{align}
      &\frac{1}{b-a}&&\text{if }a\le x \le b\\
      &0&&\text{else}
    \end{align}
  \end{cases}
$$

**Expectation**

$$
  \begin{align}
    \color{red}{E(X)}&=\int_a^b x.\frac{1}{b-a}dx\\
    &=\frac{1}{b-a}\frac{x^2}{2}\bigg\rvert_{a}^{b}\\
    &=\frac{b^2-a^2}{2(b-a)}\\
    &=\color{red}{\frac{b+a}{2}}
  \end{align}
$$

**Variance**

We have second moment

$$
  \begin{align}
    E(X^2)&=\int_a^b x^2.\frac{1}{b-a}dx\\
    &=\frac{1}{b-a}\frac{x^3}{3}\bigg\rvert_{a}^{b}\\
    &=\frac{b^3-a^3}{3(b-a)}\\
    &=\frac{b^2+ab+a^2}{3}
  \end{align}
$$

Variance

$$
  \begin{align}
    \color{red}{V(X)}&=E(X^2)-(E(X))^2\\
    &=\frac{b^2+ab+a^2}{3}-\bigg(\frac{b+a}{2}\bigg)^2\\
    &=\color{red}{\frac{(b-a)^2}{12}}
  \end{align}
$$