---
layout: post
comments: false
title: Computational Simplifications for the CRLB
categories: [Likelihood Estimation]
---

### Computational Simplifications

$$
  I_n(\theta) := E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg)^2 \bigg]\\
$$

---

1) 

$$
  E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)\bigg] = 0 \quad \text{(1)}
$$

  Proof:

  $$
    \begin{align}
      E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)\bigg)\bigg] &= \int\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)\bigg)f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x})\\
      &= \int\Bigg(\frac{\frac{\partial}{\partial\theta} f(\stackrel{\rightharpoonup}{x};\theta)}{f(\stackrel{\rightharpoonup}{x};\theta)}\Bigg)f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x})\\
      &= \int \frac{\partial}{\partial\theta} f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x})\\
      &= \frac{\partial}{\partial\theta}\int f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x})\\
      &= \frac{\partial}{\partial\theta}1 = 0
    \end{align}
  $$

---

2) 

$$
    I_n(\theta) := -E\bigg[\frac{\partial^2}{\partial\theta^2}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg]  \quad \text{(2)} 
$$

  Proof: From $(1)$, we have that

  $$
    \int\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)\bigg)f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x}) = 0
  $$
  
  Take the derivative on both sides w.r.t $\theta$.

  $$
    \int\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)\bigg)\frac{\partial}{\partial\theta}f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x}) + \\
    \int\bigg(\frac{\partial^2}{\partial\theta^2}\ln f(\stackrel{\rightharpoonup}{x};\theta)\bigg)f(\stackrel{\rightharpoonup}{x};\theta)d(\stackrel{\rightharpoonup}{x})
  $$

  We have this:

  $$
    \underbrace{E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg)^2 \bigg]}_{I_n(\theta} + E\bigg[\frac{\partial^2}{\partial\theta^2}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg] = 0
  $$

---

3) 

If $X_1,X_2,...,X_n$ are iid, then:

$$
  I_n{\theta}= nI_1{\theta} \quad \text{(3)}
$$

  Proof:

  $$
    \begin{align}
      I_n(\theta) &:= E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg)^2 \bigg]\\
      &= E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln \prod_{i=1}^{n}f(x_i,\theta)\bigg)^2 \bigg]\\
      &= E\bigg[\bigg(\frac{\partial}{\partial\theta} \sum_{i=1}^{n}\ln f(x_i,\theta)\bigg)^2 \bigg]\\
      &= E\bigg[\bigg(\sum_{i=1}^{n}\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)^2 \bigg]\\
      &= E\Bigg[\bigg(\sum_{i=1}^{n}\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)\bigg(\sum_{j=1}^{n}\frac{\partial}{\partial\theta}\ln f(x_j,\theta)\bigg)\Bigg]\\
      &= E\Bigg[\sum_{i=1}^{n}\sum_{j=1}^{n}\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)\bigg(\frac{\partial}{\partial\theta}\ln f(x_j,\theta)\bigg)\Bigg]\\
      &= \sum_{i=1}^{n}\sum_{j=1}^{n}E\Bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)\bigg(\frac{\partial}{\partial\theta}\ln f(x_j,\theta)\bigg)\Bigg]\\
    \end{align}
  $$

  These two terms above are independent if $j\neq i$. In this case, the expectation factors and both are zero by (1)

  The surviving terms are

  $$
    \begin{align}
      &= \sum_{i=1}^{n}E\Bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)\Bigg]\\
      &= \sum_{i=1}^{n}E\Bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)^2\Bigg]\\
    \end{align}
  $$

  Because the $X_i$ are iid, these expectations are all the same!

  $$
    = n . E\Bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(x_i,\theta)\bigg)^2\Bigg]\\
  $$


---

### Example 1

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim} exp(rate=\lambda)
$$

Find the Cramer-Rao lower bound of the variance of all unbiased estimators of $\lambda$.

We have:

* $\tau(\lambda) = \lambda$
* $Var[T] \ge \frac{[\tau'(\lambda)]^2}{I_n(\lambda)}$

The numerator equal $1$, now we find the Fisher Information:

$$
  I_n(\lambda) := E\Bigg[\bigg(\frac{\partial}{\partial\lambda}\ln f(x_i,\lambda)\bigg)^2\Bigg]
$$

PDF: $f(x;\lambda) = \lambda e^{-\lambda x}I_{(0,\infty)}(x)$

Joint PDF:

$$
  f(\stackrel{\rightharpoonup}{x};\lambda) = \lambda^n e^{-\lambda\sum_{n=1}^{n}x}\prod_{i=1}^{n}I_{(0,\infty)}(x_i)
$$

Take the log:

$$
  \ln f(\stackrel{\rightharpoonup}{x};\lambda) = n\ln \lambda - \lambda\sum_{n=1}^{n}x_i
$$

Take the derivative:

$$
  \frac{\partial}{\partial\lambda}\ln f(x_i,\lambda) = \frac{n}{\lambda} - \sum_{n=1}^{n}x_i
$$

Put the random variable in, square, and take the expectation.

$$
  \begin{align}
    I_n(\lambda) &:= E\Bigg[\bigg(\frac{\partial}{\partial\lambda}\ln f(x_i,\lambda)\bigg)^2\Bigg] \\
    &= E\Bigg[\bigg(\sum_{n=1}^{n}x_i - \frac{n}{\lambda}\bigg)^2\Bigg]
  \end{align}
$$

We know that the sum of $n$ exponentials independent and identically distributed with rate $\lambda$ has a gamma distribution with parameters $n$ and $\lambda$

  * Let $Y = \sum_{n=1}^{n}x_i$
  * We know that $Y \sim \Gamma(n,\lambda)$
  * We know that $E(Y) = \frac{n}{\lambda}$

$$
  I_n(\lambda) := E\Bigg[\bigg(Y - \frac{n}{\lambda}\bigg)^2\Bigg] = Var[Y] = \frac{n}{\lambda^2}
$$

The **CRBL** is:

$$
  CRBL_\lambda = \frac{1^2}{I_n(\lambda)} = \frac{1}{n / \lambda^2} = \frac{\lambda^2}{n}
$$

Alternatively:

$$
  \frac{\partial}{\partial\lambda}\ln f(x_i,\lambda) = \frac{n}{\lambda} - \sum_{n=1}^{n}x_i \\
  \implies \frac{\partial^2}{\partial\lambda^2}\ln f(x_i,\lambda) = -\frac{n}{\lambda^2}
$$

$$
  \begin{align}
    I_n(\lambda) &= -E\bigg[\frac{\partial^2}{\partial\lambda^2}\ln f(\stackrel{\rightharpoonup}{x};\lambda)\bigg] \\
    &= -E\bigg[-\frac{n}{\lambda^2}\bigg] = \frac{n}{\lambda^2}
  \end{align}
$$

Either of these methods could have been done with a single $X_1$ as opposed to the vector $\stackrel{\rightharpoonup}{X}$

### Example 2

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim} exp(rate=\lambda)
$$

Find the Cramer-Rao lower bound of the variance of all unbiased estimators of $exp[-\lambda]$

We have:
* $\tau(\lambda)=e^{-\lambda}$.
* $Var[T] \ge \frac{[\tau'(\lambda)]^2}{I_n(\lambda)}$

We already computed the Fisher Information in the example 1. Now we only need to compute the derivative of $\tau(\lambda)=e^{-\lambda}$.

The CRLB on the variance of all unbiased estimators of $exp[-\lambda]$ is:

$$
  \frac{[-\lambda e^{-\lambda}]^2}{n / \lambda^2}=\frac{\lambda^4 e^{-2\lambda}}{n}
$$

Why might we care about $exp[-\lambda]$?
* The CDF of the exponential with rate $\lambda$ is $1 - e^{-\lambda x}$. The CDF evaluated at $1$ is $1 - e^{(-\lambda)}$, if we take $1-(1 - e^{-\lambda x})$, we have $e^{-\lambda}$. This represents the probability any particular $x$ is greater than one.
