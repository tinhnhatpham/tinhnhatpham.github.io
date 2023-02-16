---
layout: post
comments: false
title: Bernoulli and Geometric Random Variables
categories: [Discrete Random Variable]
---

### **Bernoulli Random Variable**

**Bernoulli rv**, sometimes called a binary rv, is any random variable with only two possible outcomes: $0$ and $1$.

The probability mass function (pmf) is given by:

$$
  \begin{align}
    P(X=1)&=p\\
    P(X=0)&=1-p\\
  \end{align}
$$

CDF 

$$
  F(x) = P(X\le x)=\begin{cases}
    \begin{align}
      &0&&\text{if }x\lt 0\\
      &1-p&&\text{if }0\le x\lt 1\\
      &1&&\text{if }1\le x\\
    \end{align}
  \end{cases}
$$

<u>Notation:</u> We write $\color{red}{X\sim Bern(p)}$ to indicate that $X$ is a Bernoulli rv with success probability $p$.

### **Geometric Random Variable**

**Motivating Example** A patient needs a kidney transplant and is waiting for a matching donor.The probability that a random selected donor is a suitable match is $p$.

What is the sample space? What is an appropriate rv? What is the pmf?

$$
  S=\{1,01,001,0001,\ldots\}
$$

Let $X$ = # of donors tested until a match is found.

$$
  X\in\{1,2,3,\ldots\}
$$

$$
  \begin{align}
    P(X=1)&=p\\
    P(X=2)&=(1-p)p\\
    P(X=3)&=(1-p)^2p\\
  \end{align}
$$

PMF

$$
  \color{red}{P(X=k)=(1-p)^{k-1}p}\quad(k=1,2,3,\ldots)
$$

---

**Geometric series**

$$
  a+ar+ar^2+\ldots=\sum_{k=1}^{\infty}ar^{k-1}=\begin{cases}
    \begin{align}
      &\frac{a}{1-r}&&\text{if }\vert r\vert\lt1\\
      &\text{diverges}&&\text{if }\vert r\vert\ge1\\
    \end{align}
  \end{cases}
$$

We have pmf for a geometric r.v.

$$
  P(X=k)=\underbrace{(1-p)^{k-1}}_{r}\underbrace{p}_{a}
$$

Verify that the sum equal $1$:

$$
  \begin{align}
    \sum_{k=1}^{\infty}P(X=k)&=\sum_{k=1}^{\infty}(1-p)^{k-1}p\\
    &=\frac{p}{1-(1-p)}=1\\
    &\text{(note: }r=1-p\lt1)
  \end{align}
$$

---

A **geometric** rv consists of independent Bernoulli trials, each
with the same probability of success $p$, repeated until the first
success is obtained.

* Each trial is identical, and can result in a success or
failure.
* The probability of success, $p$, is constant from one trial to
the next.
* The trials are independent, so the outcome on any
particular trial does not influence the outcome of any
other trial.
* Trials are repeated until the first success.

### **Summary**

<p>&#9830; Sample space for a geometric rv:</p>

$$
  S=\{1,01,001,\ldots\}
$$

<p>&#9830; Probability mass function for a geometric rv with probabililty of success $p$:</p>

$$
  \color{red}{P(X=k)=(1-p)^{k-1}p,\quad k=1,2,3\ldots}
$$

<p>&#9830; Notation: We write $\color{red}{X\sim \text{Geom}(p)}$ to indicate that $X$ is a geometric rv with success probability $p$.</p>