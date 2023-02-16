---
layout: post
comments: false
title: General Confidence Intervals Part 2
categories: [Confidence Intervals Beyond the Normal Distribution]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Construct a 95% confidence interval for $\lambda$ **based on the minimum value in the sample**.

![png](\assets\images\notes\general-confidence-intervals-2.png)

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
  Y_n\sim exp(rate=n\lambda)
$$

---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Construct a 95% confidence interval for $\lambda$ **based on the minimum value in the sample**.

**Step One:** Choose a statistic.

$$
  Y_n=min(X_1,X_2,...,X_n)
$$

**Step Two:** Find a function of statistic and the parameter you are trying to estimate whose distribution is known and parameter free.

$$
  \begin{align}
    Y_n&=min(X_1,X_2,...,X_n)\sim exp(rate=n\lambda)\\
    &=\Gamma(1,n\lambda)\\
    &\implies n\lambda Y_n\sim\Gamma(1,1) = exp(rate=1)
  \end{align}
$$

**Step Three:** Find appropriate critical values.

$$
  n\lambda Y_n\sim exp(rate=1)
$$

Now we want to intergrate the exponential rate $1$ distribution up to some number that gives us area $0.95$ to the left

![png](\assets\images\notes\general-confidence-intervals-5.png)

To find the number which we're calling question mask, we need to integrate the exponential rate $1$ pdf from 0 to the question mask and set that equal to $0.95$.

$$
  \int_0^? e^{-x}dx=0.95\\
  \Downarrow\\
  1-e^{-?}=0.95\\
  \Downarrow\\
  ?=-\ln(0.05)
$$

**Step Four:** Put your statistic from Step Two between the critical values and solve for the unknown paramenter "in the middle".

$$
  0\lt n\lambda Y_n\lt-\ln(0.05)\\
  \Downarrow\\
  \bigg(0,\frac{-\ln(0.05)}{nY_n}\bigg)
$$

Where $Y_n=min(X_1,X_2,...,X_n)$.

