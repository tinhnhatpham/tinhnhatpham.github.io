---
layout: post
comments: false
title: Properties of the Exponential Distribution
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

We know that

$$
  \sum_{i-1}^{n}X_i\sim\Gamma(n,\lambda)
$$

and therefore that

$$
  2\lambda\sum_{i-1}^{n}X_i\sim\Gamma\bigg(\frac{2n}{2},\frac{1}{2}\bigg)=\chi^2(2n)
$$

We also know that

$$
  \overline{X}=\frac{1}{n}\sum_{i-1}^{n}X_i\sim\Gamma(n,n\lambda)
$$

and there for that

$$
  2n\lambda\overline{X}\sim\Gamma\bigg(n,\frac{1}{2}\bigg)=\Gamma\bigg(\frac{2n}{2},\frac{1}{2}\bigg)=\chi^2(2n)
$$

---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

What is the distribution of minimum?

![png](\assets\images\notes\properties-of-the-exponential-distribution.png)

Let $Y_n=\text{min}(X_1,X_2,...,X_n)$.

The cdf for each $X_i$ is

$$
  F(x)=P(X_i\le x)=1-e^{-\lambda x}
$$

Let $Y_n=min(X_1,X_2,...,X_n)$.

The cdf for each $X_i$ is

$$
  F(x) = P(X_i\le x)=1-e^{-\lambda x}
$$

The cdf for $Y_n$ is

$$
  \begin{align}
    F_{Y_n}(y)&=P(Y_n\le y)\\
    &=P(min(X_1,X_2,...,X_n)\le y)
  \end{align}
$$

If we choose $x$'s like the way below, there's a lot of way for $x$'s to fall so that the minimum is less that or equal to $y$ it's going to be hard to translate this about the minimum to a statement involving where the individual $x$'s fall with respect to this fixed number $y$.

![png](\assets\images\notes\general-confidence-intervals-3.png)

But it will be easier if we look at the complement of this event. The minimum in a sample is greater than a fixed number $y$, if and only if every number in the sample is greater than $y$.

![png](\assets\images\notes\general-confidence-intervals-4.png)

$$
  \begin{align}
    F_{Y_n}(y)&=P(Y_n\le y)\\
    &=P(min(X_1,X_2,...,X_n)\le y)\\
    &=1-P(min(X_1,X_2,...,X_n)\gt y)\\
    &=1-P(X_1\gt y,X_2\gt y,...,X_n\gt y)\\
    &\stackrel{indep}{=}1-P(X_1\gt y).P(X_1\gt y)...P(X_n\gt y)\\
    &\stackrel{ident}{=}1-[P(X_1\gt y)]^n = 1-[1-F(y)]^n\\
    &=1-[1-(1-e^{-\lambda y}]^n\\
    &=1-[e^{-\lambda y}]^n
  \end{align}
$$

If we take the derivative and we get the pdf of an exponential with paramenter $n\lambda$

$$
  f(Y_n(y) = \frac{d}{dx}F_{Y_n}(y)=n\lambda e^{-n\lambda y}
$$

The minimum of $n$ iid exponential with rate $\lambda$ is exponential with rate $n\lambda$!

$$
  Y_n\sim \text{exp}(\text{rate}=n\lambda)
$$
