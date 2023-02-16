---
layout: post
comments: false
title: More on Expectation and Variance
categories: [Joint Distributions and Covariance]
---

In statistics and data science, we
frequently collect data from several random variables and we
want to understand and quantify the strength of their
interactions.

* The length of time a student studies and their score on
an exam.

* The relationship between male and female life expectancy
in a certain country.

* The relationship between the quantity of two different
products purchased by a consumer.

---

### **Expectation**

$$
  \begin{align}
    E(X)&=\sum_{k}kP(X=k)&&\text{if $X$ is discrete}\\
    E(X)&=\int_{-\infty}^{\infty}xf(x)dx&&\text{if $X$ is continuous}\\
  \end{align}
$$

What we can say about $E\big(g(x)\big)$?

$$
  E\big(g(x)\big)=\begin{cases}
    \begin{align}
      &\sum_{k}g(k)P(X=k)&&\text{$X$ is discrete}\\
      &\int_{-\infty}^{\infty}g(x)f(x)dx&&\text{$X$ is continuous}\\
    \end{align}
  \end{cases}
$$

$$
  \begin{align}
    E(aX+b)&=\sum_{k}(ak+b)P(X=k)\\
    &=a\underbrace{\sum_{k}kP(X=k)}_{E(X)}+b\underbrace{\sum_{k}P(X=k)}_{1}\\
    &=aE(X)+b
  \end{align}
$$

**Example:** Suppose a university has \\\$15,000 students and let $X$ equal the number of courses for which a randomly selected student is registered. The pmf is 

| $x$    | 1   | 2    | 3    | 4    | 5    | 6    | 7    |
|--------|-----|------|------|------|------|------|------|
| $p(x)$ | 0.01 | 0.03 | 0.13 | 0.25 | 0.39 | 0.17 | 0.02 |

If a student pay \\\$500 per course plus a \\\$100 per-semester registration fee, what is the average amount a student pays each semester?

Let $X$ = # of courses student takes

We want to find $E(500X+100)=500E(X)+100$.

$$
  \begin{align}
    E(X)&=\sum_{k=1}^7kP(X=k)\\
    &=1(0.01) + 2(0.03) + \ldots + 7(0.02)\\
    &=4.57
  \end{align}
$$

So 

$$
  E(500X+100) = 500(4.57) + 100 = \$2385
$$

---

### **Variance**

$$
  \sigma^2=V(X)=E[\underbrace{(X-\mu)^2}_{g(X)}]=\underbrace{E(X^2)-\big(E(X)\big)^2}_{\text{Computational formula}}
$$

$$
  \begin{align}
    V(X)&=\sum_{k}(k-\mu)^2P(X=k)&&\text{if $X$ is discrete}\\
    V(X)&=\int_{-\infty}^{\infty}(x-\mu)^2f(x)dx&&\text{if $X$ is continuous}\\
  \end{align}
$$

What about $V\big(g(X)\big)$?

$$
  V\big(g(X)\big)=\begin{cases}
    \begin{align}
      &\sum_{k}\bigg(g(k)-E\big(g(X)\big)\bigg)^2P(X=k)&&\text{$X$ is discrete}\\
      &\int_{-\infty}^{\infty}\bigg(g(x)-E\big(g(X)\big)\bigg)^2f(x)dx&&\text{$X$ is continuous}\\
    \end{align}
  \end{cases}
$$

Find $V(aX+b)$

$$
  \begin{align}
    V(aX+b)&=E\big[\big(aX+b-E(aX+b)\big)^2\big]\\
    &=E\big[\big(aX+b-aE(X)+b\big)^2\big]\\
    &=E\big[a^2\big(X-E(X))^2\big]\\
    &=a^2E\big[\big(X-E(X))^2\big]\\
    &= a^2V(X)
  \end{align}
$$

Recall: $E(aX+b)=aE(X)+b$

Variance measure spread ofthe data, the $b$ shifts the data but doesn't affect the spread.

**Example:** Suppose a university has \\\$15,000 students and let $X$ equal the number of courses for which a randomly selected student is registered. The pmf is 

| $x$    | 1   | 2    | 3    | 4    | 5    | 6    | 7    |
|--------|-----|------|------|------|------|------|------|
| $p(x)$ | 0.01 | 0.03 | 0.13 | 0.25 | 0.39 | 0.17 | 0.02 |

If a student pay \\\$500 per course plus a \\\$100 per-semester registration fee, what is the average amount a student pays each semester?

We found $E(500X+100) = 500(4.57) + 100 = \$2385$

$$
  \begin{align}
    V(X)&=E(X^2)-\big(E(X)\big)^2\\
    &=\sum_{k=1}^7k^2P(X=k)-(4.57)^2\\
    &\approx1.265
  \end{align}
$$

$$
  V(500X+100)=500^2V(X)=316,273
$$

---