---
layout: post
comments: false
title: Large Sample Properties of MLEs
categories: [Likelihood Estimation]
---

Let $X_1,X_2,...,X_n$ be a random sample from a distribution with pdf $f(x;\theta)$.

Let $\hat{\theta}$ be an MLE for $\theta$.

Under certain "regularity conditions" such as those needed for the CRLB.

* $\hat{\theta}$ exists and is unique.
* $\hat{\theta} \stackrel{P}{\rightarrow}\theta$. We say that $\hat{\theta}$ is a **consistent estimator** of $\theta$.
* $\hat{\theta}_n$ is an asymptotically unbiased estimator of $\theta$.

  $$
    \text{i.e} \quad \lim_{n\to\infty}E[\hat{\theta}_n] = 0
  $$

* $\hat{\theta}_n$ is **asymptotically efficient**.

  $$
    \text{i.e} \quad \lim_{n\to\infty}\frac{CRLB_\theta}{Var[\hat{\theta}_n]} = 1
  $$

* $\hat{\theta}\stackrel{asymp}{\sim}N(\theta,CRLB_\theta)$

  $$
    \frac{\hat{\theta}_n-\theta}{\sqrt{CRLB_\theta}}\stackrel{d}{\rightarrow}N(0,1)
  $$

**Example**

$$
  X_1,X_2,...,X_n \sim exp(\text{rate}=\lambda)
$$

We have seen that the MLE for $\lambda$ is 

* the MLE for $\lambda$ is $\hat{\lambda} = \frac{1}{\bar{X}}$

* $E[\hat{\lambda}]=\frac{n}{n-1}\lambda \quad \text{which goes to $\lambda$ as $n\rightarrow\infty$}$

* $\bar{X}\stackrel{P}{\rightarrow}E[X_1]=1/\lambda$

Is it true that

$$
  \hat{\lambda}=\frac{1}{\bar{X}}\rightarrow\frac{1}{1/\lambda}=\lambda\text{ ?}
$$

---

Suppose that $\{X_n\}$ and $\{Y_n\}$ be sequences of random variables such that $X_n\stackrel{P}{\rightarrow}X$ and $Y_n\stackrel{P}{\rightarrow}Y$ for random variables $X$ and $Y$. Some **properties of convergence** in probability:

* $X_n + Y_n \stackrel{P}{\rightarrow} X + Y$
* $X_nY_n\stackrel{P}{\rightarrow} XY$
* $X_n/Y_n\stackrel{P}{\rightarrow} X/Y$ (if $P(Y\neq0)=1$)
* $g(X_n)\stackrel{P}{\rightarrow}g(X)$ (for $g$ continous)

---

Thus,

Using $g(x)=1/x$, we do have that $\bar{X}\stackrel{P}{\rightarrow}E[X_1]=1/\lambda$ implies that:

$$
  \hat{\lambda}=\frac{1}{\bar{X}}\stackrel{P}{\rightarrow}\frac{1}{1/\lambda}=\lambda
$$

We saw that the CRLB for $\lambda$ is

$$
  CRLB_\lambda=\frac{\lambda^2}{n}
$$

$$
  \begin{align}
    Var[\hat{\lambda}] &= Var\Bigg[\frac{1}{\bar{X}}\Bigg]\\
    &= E\Bigg[\bigg(\frac{1}{\bar{X}}\bigg)^2\Bigg] - \Bigg(E\bigg[\frac{1}{\bar{X}}\bigg]\Bigg)^2
  \end{align}
$$

We already have $E[\frac{1}{\bar{X}}]=\frac{n}{n-1}\lambda$. Now we need to calculate $E[(\frac{1}{\bar{X}})^2]$.

$$
  \begin{align}
    E\Bigg[\bigg(\frac{1}{\bar{X}}\bigg)^2\Bigg]&=E\Bigg[\frac{n^2}{Y^2}\Bigg]\quad\text{where}\quad Y\sim\Gamma(\alpha,\beta)\\
    &=n^2\int_{-\infty}^{\infty}\frac{1}{y^2}f_Y(y)dy\\
    &=n\int_{0}^{\infty}\frac{1}{y^2}.\frac{1}{\Gamma(n)}\lambda^ny^(n-1)e^{-\lambda y}dy\\
    &=n\int_{0}^{\infty}\frac{1}{\Gamma(n)}\lambda^ny^(n-3)e^{-\lambda y}dy\\
    &=n^2\lambda^2\frac{\Gamma(n-2)}{\Gamma(n)}\int_{0}^{\infty}\frac{1}{\Gamma(n-2)}\lambda^{n-2}y^{n-3}e^{-\lambda y}dy\\
    &=\frac{n^2}{(n-1)(n-2)}\lambda^2
  \end{align}
$$

$$
  \begin{align}
    Var\Bigg[\frac{1}{\bar{X}}\Bigg]&=E\Bigg[\bigg(\frac{1}{\bar{X}}\bigg)^2\Bigg] - \Bigg(E\bigg[\frac{1}{\bar{X}}\bigg]\Bigg)^2\\
    &=\frac{n^2}{(n-1)(n-2)}\lambda^2 - \bigg(\frac{n}{n-1}\lambda^2\bigg)\\
    &=\frac{n^2}{(n-1)^2(n-2)}\lambda^2
  \end{align}
$$

And we calculate the ration of the CRLB to the variance:

$$
  \begin{align}
    \frac{CRLB_\theta}{Var[\hat{\theta}_n]}&=\frac{\frac{\lambda^2}{n}}{\frac{n^2\lambda^2}{(n-1)^2(n-2)}}\\
    &=\frac{(n-1)^2(n-2)}{n^3}=1 \quad \text{as }n\rightarrow\infty
  \end{align}
$$

That means that MLE is in fact **asymptotically efficient**.

Recall the Weak Law of Large Numbers where we showed that $\bar{X}\stackrel{P}{\rightarrow}\mu$. To prove this, we used:

* Chebyshev's inequality.
* The fact that $\bar{X}$ is an unbiased estimator of the mean $\mu$.
* The fact that $Var[\bar{X}]\rightarrow0$.

The exact same proof can be used to show the following.

If $\hat{\theta_n}$ is an unbiased estimator of $\theta$, and if $\lim_{n\to\infty}Var[\hat{\theta}_n]=0$, then $\hat{\theta}_n\stackrel{P}{\rightarrow}\theta$.

Using the generalized Markov inequality, we can show that this actually holds when "unbiased" is replaced by "asymptotically unbiased".

We can use this to show, for example, that if $X_1,X_2,...,X_n \sim unif(0,\theta)$, the maximum:

$$
  Y_n = \max(X_1,X_2,...,X_n)
$$

is a consistent estimator of $\theta$.

What is the distribution of $Y$?

$$
  P(Y_n\le y) = P(\max(X_1,X_2,...,X_n)\le y)
$$

The only way the maximum of our sample can be less than or equal to $y$ is if all our values in the sample are less than or equal to $y$ and the statement holds in the reverse.

$$
  \begin{align}
    &P(Y_n\le y) \\
    &=P(X_1\le y,X_2\le y,...,X_n\le y)\\
    &=P(X_1\le y)P(X_2\le y)...P(X_n\le y)\\
    &=[P(X_1\le y)]^2 = \bigg[\frac{y}{\theta}\bigg]^2\\
    &\text{for }0\le y\le\theta.
  \end{align}
$$

The pdf for $Y_n=\max(max(X_1,X_2,...,X_n)$ is

$$
  \begin{align}
    f_{Y_n}(y)&=\frac{d}{dy}F_{Y_n}(y)\\
    &=\frac{d}{dy}\bigg[\frac{y}{\theta}\bigg]^n\\
    &=\frac{n}{\theta^n}y^{n-1}\quad\text{for }0\le y\le\theta
  \end{align}
$$

The expected value of the maximum is then

$$
  \begin{align}
    E[Y_n]&=\int_{-\infty}^{\infty}yf_{Y_n}(y)dy\\
    &=\int_{0}^{\theta}\frac{n}{\theta^n}y^ndy\\
    &=\frac{n}{n+1}\theta
  \end{align}
$$

$$
  \begin{align}
    E[Y_n]&=\frac{n}{n+1}\theta\\
    Var[Y_n]&=\frac{n}{(n+1)^2(n+2)}\theta^2
  \end{align}
$$

We have that it's a asymptotically unbiased for $\theta$ and that its variance is going to $0$ as $n$ goes to infinity. We can conclude that we have a **consistent estimator**.
