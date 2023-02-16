---
layout: post
comments: false
title: Central Limit Theorem Examples
categories: [Central Limit Theorem]
---

<u>Proposition:</u> If $X_1,X_2,...,X_n$ are iid with $X_i\sim N(\mu,\sigma^2)$ then $\overline{X}\sim N(\mu,\sigma^2/n)$.

<u>Proposition:</u> If $X_1,X_2,...,X_n$ are independent with $X_i\sim N(\mu_i,\sigma_i^2)$ then

<font color='red'>
$$
  \sum_{i=1}^{n}X_i\sim N\bigg(\sum_{i=1}^{n}\mu_i,\sum_{i=1}^{n}\sigma_i^2\bigg)\\
$$
</font>

$$
  \begin{align}
    E\bigg(\sum_{i=1}^{n}X_i\bigg)&\stackrel{\text{iid}}{=}\sum_{i=1}^{n}E(X_i)=\sum_{i=1}^{n}\mu_i\\
    V\bigg(\sum_{i=1}^{n}X_i\bigg)&\stackrel{\text{iid}}{=}\sum_{i=1}^{n}V(X_i)=\sum_{i=1}^{n}\sigma_i^2\\
  \end{align}
$$

<u>Extended:</u> 

<font color='red'>
$$
  \sum_{i=1}^{n}c_iX_i\stackrel{\text{iid}}{\sim}N\bigg( \sum_{i=1}^{n}c_i\mu_i,\sum_{i=1}^{n}c_i^2\sigma_i^2\bigg)\\
$$
</font>

Suppose you have 3 errands to do in three different stores. Let $T_i$ be the time to make the $i^{th}$ purchase for $i=1,2,3$. Let $T_4$ be the total walking time between stores. Suppose that

$$
  \begin{align}
    T_1&\sim N(15,16)\\
    T_2&\sim N(5,1)\\
    T_3&\sim N(8,4)\\
    T_4&\sim N(12,9)\\
  \end{align}
$$

Assume $T_1,T_2,T_3,T_4$ are independent. If you leave at 10 in the morning and you want to tell a colleague, "I'll be back by time $t$", what should $t$ be so that you will return by the time with probability $0.99$?

Let $T_0=T_1+T_2+T_3+T_4=\text{total time}$

$$
  \begin{align}
    E(T_0)&=15+5+8+12&&=40\\
    V(T_0)&=16+1+4+9&&=30
  \end{align}
$$

$$\implies T_0\sim N(40,30)$$

We want to find $t$ so that $P(T_0\le t)=0.99$. We can solve the problem by standardize $T_0$:

$$
  P\bigg(\frac{T_0-40}{\sqrt{30}}\le\frac{t-40}{\sqrt{30}}\bigg)=0.99\\
  \implies P\bigg(Z\le\frac{t}{\sqrt{30}}\bigg)=\Phi\bigg(\frac{t-40}{\sqrt{30}}\bigg)=0.99\\
  \implies\Phi(2.333)=0.99\\
$$

We can use R to find $\Phi$:

```R
qnorm(0.99)
```

$$
  \frac{t-40}{\sqrt{30}}=2.33\\
  \implies t=52.76\text{ minutes}
$$

So if we leave at $10$ o'clock, we will return by $10:52:76$ with probability $0.99\%$.

---

**Central Limit Theorem** Let $X_1,X_2,...,X_n$ be a random sample with $E(X_i)=\mu$ and $V(X_i)=\sigma^2$. If $n$ is sufficiently large, $\overline{X}$ has approximately a normal distribution with mean $\mu_{\overline{X}}=\mu$ and variance $\sigma_{\overline{X}}^2=\sigma^2/n$. 

You want to verify that $25$-kg bags of fertilizer are being filled to the appropriate amount. You select a random sample of $50$ bags of fertilizer and weigh them. Let $X_i$ be the weight of the $i_{th}$ bag for $i=1,2,...,50$. 

You expect $E(X_i)=25$ and $V(X_i)=0.5$. Let $\overline{X}=(1/50)\sum_{i=1}^{50}X_i$.

Find $P(24.75\le\overline{X}\le25.25)$.

From CLT we have

$$
  \overline{X}\sim N\left(25,\frac{0.5}{50}\right)=N\left(25,0.01\right)
$$

$$
  \begin{align}
    &P(24.75\le\overline{X}\le25.25)\\
    &=P\left(\frac{24.75-25}{\sqrt{0.01}}\le\frac{\overline{X}-25}{\sqrt{0.01}}\le\frac{25.25-25}{\sqrt{0.01}}\right)\\
    &=P(-2.5\le Z\le2.5)\\
    &=\Phi(2.5)-\Phi(-2.5)\approx0.9876\\
  \end{align}
$$

Suppose $E(X_i)=24.5$, that is, the bags are underfilled, and $V(X)=0.5$. Now, find $P(24.75\le\overline{X}\le25.25)$.

$\overline{X}\sim N(24.5,0.01)$

$$
  \begin{align}
    &=P\left(\frac{24.75-24.5}{\sqrt{0.01}}\le\frac{\overline{X}-24.5}{\sqrt{0.01}}\le\frac{25.25-24.5}{\sqrt{0.01}}\right)\\
    &=P(2.5\le Z\le7.5)\\
    &=\Phi(7.5)-\Phi(2.5)\\
    &=1.000 - 0.9938\\
    &=0.0062
  \end{align}
$$

---

In a statistics class of $36$ students, past experience indicates that $53\%$ of the students will score at or above $80\%$. For a randomly selected exam, find the probability at least $20$ students will score above $80\%$.

Let 

$$
  X_i=\begin{cases}
    \begin{align}
      &1&&\text{if $i^{th}$ student scores }\ge80\%\\
      &0&&\text{if score }\lt80\%
    \end{align}
  \end{cases}
$$

Let $X=\sum_{i=1}^{36}X_i=\text{# of successes}$.

We know that

$$
  X\sim\text{Bin}(np, np(1-p))\\
  X\sim\text{Bin}(19.08,8.97)
$$

We have $P(X\ge20) = 1 - P(X\le19.5)$. We make it $19.5$ here because this is a discrete rv, and we're going to overlay a normal rv on top of it, so we want to go half way between $19$ and $20$ to account for that continuity on the domain.

$$
  \begin{align}
    P(X\ge20)&=1 - P(X\le19.5)\\
    &=1 - P\left(\frac{X-19.08}{\sqrt{8.97}}\le\frac{19.5-19.08}{\sqrt{8.97}}\right)\\
    &=1-P(Z\le0.14)\\
    &=1-\Phi(0.14)\\
    &\approx0.4443
  \end{align}
$$

Example: Normal approximation to the binomial. If $X\sim\text{Bin}(n,p)$ then $X$ counts the number of successes in $n$ independent Bernoulli trials, each with probability of success $p$. We know:

$$
  E(X)=np\qquad V(X)=np(1-p)
$$

So, by CLT:

$$
  \frac{X-np}{\sqrt{np(1-p)}}\approx N(0,1)
$$

The CLT provides insight into why many random variables
have probability distributions that are approximately normal.

For example, the measurement error in a scientific experiment.
can be thought of as the sum of a number of underlying
perturbations and errors of small magnitude.

A practical difficulty in applying the CLT is in knowing when $n$
is sufficiently large. The problem is that the accuracy of the
approximation for a particular $n$ depends on the shape of the
original underlying distribution being sampled.