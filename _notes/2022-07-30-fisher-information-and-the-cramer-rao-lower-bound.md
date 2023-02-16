---
layout: post
comments: false
title: Fisher Information and the Cramér-Rao Lower Bound
categories: [Likelihood Estimation]
---

### The Cramer-Rao Lower Bound (CRLB)

Let $X_1,X_2,...,X_n$ be a random sample for some distribution with pdf $f(x;\theta)$

Consider estimating some $\tau(\theta)$.

Suppose that $\hat{\tau(\theta)}$ is any unbiased estimator of $\tau(\theta)$.

Then

$$
  Var[\hat{\tau(\theta)}] \ge \underbrace{\frac{[\tau'(\theta)]^2}{I_n(\theta)}}_\text{The CRLB} \quad \text{(1)}
$$

The denominator is called **Fisher Information**

$$
  I_n(\theta) := E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X},\theta)\bigg)^2 \bigg]  \quad \text{(2)}
$$

The estimator: $\hat{\tau(\theta)}=T=t(\stackrel{\rightharpoonup}{X})$

so we can rewrite $(1)$:

$$
  Var[T] \ge \underbrace{\frac{[\tau'(\theta)]^2}{I_n(\theta)}}_\text{The CRLB} \quad \text{(3)}
$$

The proof of the Cramer-Rao Lower Bound depends on the **Cauchy-Schwartz inequality**:

$$
  \bigg(\int g(x)h(x)dx \bigg)^2 <= \bigg(\int g^2(x)dx \bigg)\bigg(\int h^2(x)dx \bigg) 
$$

* This holds with sum. (discrete version)
* This holds with $dx$ replaced by $f(x)dx$ where $f$ is a pdf:

$$
  (E[g(X)h(X)])^2 \le E[g^2(X)]E[h^2(X)]
$$

The equations $(1)$ and $(2)$ are true if we can show that:

$$
  \tau'(\theta) = \big(T - \tau(\theta) \big)\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg) \quad \text{(4)}\\
  \text{Where $T=t(\stackrel{\rightharpoonup}{X})$}
$$

If we take the expectation of both sides of the above equation

$$
  E[\tau'(\theta)] = E\bigg[\big(T - \tau(\theta) \big)\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)\bigg]
$$

On the left, we don't have any random variable. So it is a constant, we can drop the expectation. Now we square both sides:

$$
  [\tau'(\theta)]^2 = \Bigg(E\bigg[\big(T - \tau(\theta) \big)\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)\bigg]\Bigg)^2
$$

Based on the **Cauchy-Schwartz inequality**, we can say that:

$$
  [\tau'(\theta)]^2 \le \underbrace{E\big[\big(T - \tau(\theta)\big)^2\big]}_\text{Var[T]} \underbrace{E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)^2\bigg]}_\text{$I_n(\theta)$  (Fisher Information)}
$$

Now we prove that (4) is true

$\tau$ is the expected value of the random variable $T$, because $T$ was assumed to be an unbiased estimator of $\tau(\theta)$.

$$
  \begin{align}
    \tau'(\theta) &=\frac{\partial}{\partial\theta}\tau(\theta)=\frac{\partial}{\partial\theta}E[T]\\
    &=\frac{\partial}{\partial\theta}\int t(\stackrel{\rightharpoonup}{x})f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x}\\
    &=\frac{\partial}{\partial\theta}\int t(\stackrel{\rightharpoonup}{x})f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x} - \tau(\theta)\underbrace{\frac{\partial}{\partial\theta}\overbrace{\int f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x}}^1}_0\\
    &= \int \big(t(\stackrel{\rightharpoonup}{x})-\tau(\theta)\big)\frac{\partial}{\partial\theta}f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x} \quad \text{(5)}
  \end{align}
$$

We want to see the expectation of $(5)$, then we can apply the Cauchy-Schwartz inequality and we will have proven the Cramer-Rao Lower Bound. So we want to see:

* $ E\Bigg[\big(t(\stackrel{\rightharpoonup}{x})-\tau(\theta)\big)\bigg(\frac{\partial}{\partial\theta}f(\stackrel{\rightharpoonup}{x};\theta)\bigg)\Bigg] \quad \text{(6)}$

* Note that: $\frac{\partial}{\partial\theta}f(\stackrel{\rightharpoonup}{x};\theta)=\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)f(\stackrel{\rightharpoonup}{x};\theta) \quad \text{(7)}$

And we take the $(7)$ and plug it in the $(5)$ and we get $(6)$.

The CRLB is valid if:

* $\frac{\partial}{\partial\theta}\int f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x}=\int\frac{\partial}{\partial\theta} f(\stackrel{\rightharpoonup}{x};\theta)d\stackrel{\rightharpoonup}{x}=$

  This one doesn't hold true whenever the parameter is in indicator or support of the distribution. Example: The CRLB doesn't hold for the $unif(0,\theta)$ distribution!

* $\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{x};\theta)$ exists.

* $0 \lt E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)^2\bigg] \lt \infty$

---

### Example:

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim}Bernoulli(p)
$$

Find the Cramer-Rao lower bound of the variance of all unbiased estimators of $p$.

Here, $\theta=p$ and $\tau(p)=p$.

The Fisher Information:
 
$$
  I_n(p) := E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)^2\bigg]
$$

PDF: $f(x;p)=p^x(1-p)^{1-x}I_{0,1}(x)$

Joint PDF: $f(\stackrel{\rightharpoonup}{x};p)$

$$
  = p^{\sum_{i=1}^{n}(1-p)^{n-\sum_{i=1}^{n}x_i}}\prod_{i=1}^{n}I_{0,1}(x_i)
$$

And we take the log and derivative. The indicators are just placehoders.

Take the log $\ln f(\stackrel{\rightharpoonup}{x};p)$:

$$
  = \Bigg(\sum_{i=1}^{n}\Bigg)\ln p + \Bigg(n-\sum_{i=1}^{n}\Bigg)\ln (1-p)
$$

Take the derivative:

$$
  \begin{align}
    \frac{\partial}{\partial p}\ln f(\stackrel{\rightharpoonup}{x};p) &= \frac{\sum_{i=1}^{n}x_i}{p} - \frac{n-\sum_{i=1}^{n}x_i}{1-p}\\
    &= \frac{\sum_{i=1}^{n}x_i - np}{p(1-p)}
  \end{align}
$$

Note that $Y = \sum_{i=1}^{n}X_i \sim binomial(n,p)$

$$
  \begin{align}
    I_n(p) &= E\bigg[\bigg(\frac{\partial}{\partial\theta}\ln f(\stackrel{\rightharpoonup}{X};\theta)\bigg)^2\bigg]\\
    &= E\bigg[\bigg(\frac{Y - np}{p(1-p)}\bigg)^2\bigg]\\
    &= \frac{1}{p^2(1-p)^2}\underbrace{E[(Y-np)^2]}_\text{variance of binomial}\\
    &= \frac{np(1-p)}{p^2(1-p)^2}\\
    &= \frac{n}{p(1-p)} \quad \text{Fisher Information}
  \end{align}
$$

Based of CRLB:

$$
  \begin{align}
    Var[\hat{p}] &= \frac{[\tau'(p)]^2}{I_n(p)}\\
    &= \frac{1^2}{n/[p(1-p)]}\\
    &= \frac{p(1-p)}{n}
  \end{align}
$$

This is saying that no matter how you try estimate $p$ in the Bernoulli distribution, if you have an unbiased estimator, its variance cannot go lower that this.

Let's talk about the unbiased estimator of Bernoulli distribution.

* mean: $p$
* variance: $p(1-p)$
* $E[\bar{X}] = E[X_1] = p$
* $Var[\bar{X}] = \frac{Var[X_1]}{n} = \frac{p(1-p)}{n} (= CRLB)$

So $\bar{X}$ is an unbiased estimator of $p$ with the smallest possible variance.

This estimator for $p$ for the Bernoulli distribution is actually a **U**niform **M**inimum **V**ariance **U**nbiased **E**stimator.

