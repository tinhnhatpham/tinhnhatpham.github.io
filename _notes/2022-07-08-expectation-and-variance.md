---
layout: post
comments: false
title: Expectation and Variance
categories: [Discrete Random Variable]
---

### **Expectation**

**Motivating Example** A patient needs a kidney transplant and is waiting for a matching donor.The probability that a random selected donor is a suitable match is $p$.

Let $X$ be the number of potential donors tested until a match is found.

PMF: $P(X=k)=(1-p)^{k-1}p$ for $k\in\{1,2,3,\ldots\}$.

<u>Question:</u> How many potential donors must be tested before there is a successful match? In other words, what is the expected value (also known as the average or mean) of the random variable?

<u>Notation:</u> $E(X)$ or $\mu_X$ is the expected value of a random variable $X$.

<u>Example:</u> 5 event $70,80,80,90,90$

$$
  \begin{align}
    \text{Average}&=\frac{70+80+80+90+90}{5}\\
    &=\frac{1}{5}(70)+\frac{2}{5}(80)+\frac{2}{5}(90)
  \end{align}
$$

---

**Definition:** The expected value of a discrete random variable, $E(X)$, is given by

$$
  E(X)=\sum_{k}k\underbrace{P(X=k)}_{(1)}\\
  (1)\rightarrow\text{fraction of the population with value }k
$$

---

If $X\sim\text{Bern}(p)$, what is $E(X)$?

$$
  P(X=0)=1-p,\qquad P(X=1)=p\\
  E(X) = 0P(X=0) + 1(P(X=1) = p
$$

---

If $Y\sim\text{Geom}(p)$, what is $E(Y)$?

PMF

$$
  \color{red}{P(X=k)=(1-p)^{k-1}p}\quad(k=1,2,3,\ldots)
$$

Recall from geometric series:

$$
  \sum_{k=1}^{\infty}ar^{k-1}=\frac{a}{1-r},\quad\vert r\vert\lt1\\
$$

Differentiate w.r.t $r$:

$$
  \sum_{k=1}^{\infty}a(k-1)r^{k-2}=\frac{a}{(1-r)^2}\\
  \sum_{k=2}^{\infty}a(k-1)r^{k-2}=\frac{a}{(1-r)^2}\\
$$

Reindex $k-1=j$:

$$
  \sum_{j=1}^{\infty}ajr^{j-2}=\frac{a}{(1-r)^2}
$$

We have the expected value:

$$
  \begin{align}
    E(Y)&=\sum_{k=1}^{\infty}kP(Y=k)\\
    &=\sum_{k=1}^{\infty}kp(1-p)^{k-1}\\
    &=\frac{p}{(1-(1-p))^2}\\
    &=\frac{p}{p^2}\\
    &=\color{red}{\frac{1}{p}}
  \end{align}
$$

---

Useful properties of the expected value definition,

$$
  E(X)=\sum_{k}kP(X=k)
$$

* If $c$ is a constant, then $E(c)=c$.

* If $a$ and $b$ are constants and $X$ is a rv, then

  $$
    \begin{align}
      E(aX+b)&=\sum_{k}(ak+b)P(X=k)\\
      &=a\underbrace{\sum_{k}kP(X=k)}_{E(X)}+b\underbrace{\sum_{k}P(X=k)}_{1}\\
      &=aE(X)+b
    \end{align}
  $$

* If $h(X)$ is any function of $X$, then

  $$
    E(h(X))=\sum_{k}h(k)P(X=k)
  $$

### **Variance**

The variance of a random variable $X$, denoted $V(X)$, measures how far we expect our random variabe to be from the mean.

**Definition:** The **variance** of a random variable is given by:

$$
  \begin{align}
    \sigma^2_X&=V(X)=E[(X-E(X))^2]\\
    &=\sum_{k}(k-\mu_X)^2P(X=k)\\
    &=\sum_{k}(k^2-2\mu_Xk+\mu_X^2)P(X=k)\\
    &=\underbrace{\sum_{k}k^2(P(X=k)}_{E(X^2)}-2\mu_X\underbrace{\sum_{k}kP(X=k)}_{\mu_X}+\mu_X^2\underbrace{\sum_{k}P(X=k)}_{1}\\
    &=E(X^2)-2\mu_X^2+\mu_X^2\\
    &=E(X^2)-\mu_X^2
  \end{align}
$$

Computational formula: $\color{red}{V(X)=E(X^2)-(E(X))^2}$

(Aside: Standard deviation, $\sigma_X=\sqrt{\sigma_X^2}\ge0$)

---

**Variance for $X\sim\text{Bern}(p)$:**

$$
  \begin{align}
  P(X=0)&=1-p\\
  P(X=1)&=p\\
  E(X)&=p\\
  E(X^2)&=\sum_{k}k^2P(X=k)=1^2.p=p
  \end{align}
$$

Variance:

$$
  \begin{align}
    V(X)&=E(X^2)-(E(X))^2\\
    &=p-p^2\\
    &=p(1-p)
  \end{align}
$$

---

**Variance for $Y\sim\text{Geom}(p)$:**

$$
  \begin{align}
    P(Y=k)&=(1-p)^{k-1}p,\quad k=1,2,3,\ldots\\
    E(Y)&=\frac{1}{p}\\
    E(Y^2)&=\sum_{k}k^2P(Y=k)\\
    &=\sum_{k}k^2(1-p)^{k-1}p\\
    &=\frac{2-p}{p^2}
  \end{align}
$$

Variance:

$$
  \begin{align}
    V(Y)&=E(Y^2)-(E(Y))^2\\
    &=\frac{2-p}{p^2}-\left(\frac{1}{p}\right)^2\\
    &=\frac{1-p}{p^2}
  \end{align}
$$

---

**Example:** Suppose you have 10 folded pieces of paper, labeled $0,1,2,\ldots,9$. Draw one paper at random. Define the rv $U$ to be the number drawn. Find the pmf, expectation, and variance for $U$. (Aside: this is a discrete, uniform rv.)

PMF: $P(U=k)=\frac{1}{10},\quad k=0,1,2,\ldots,9$

$$
  \begin{align}
  E(U)&=\sum_{k=0}^{9}kP(U=k)=0.\frac{1}{10}+1.\frac{1}{10}+\ldots+9.\frac{1}{10}=4.5\\
  E(U^2)&=\sum_{k=0}^{9}k^2P(U=k)=\sum_{k=0}^{9}k^2\left(\frac{1}{10}\right)=28.5\\
  V(U)&=E(U^2)-(E(U))^2=28.5-(4.5)^2=8.25
  \end{align}
$$