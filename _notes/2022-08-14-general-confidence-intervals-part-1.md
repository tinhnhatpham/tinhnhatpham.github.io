---
layout: post
comments: false
title: General Confidence Intervals Part 1
categories: [Confidence Intervals Beyond the Normal Distribution]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Contruct a 95% confidence interval for $\lambda$.

**Step One:** Choose a statistic.

Choose one whose distribute you know and is one that depends on the unknown parameter.

We have the sample mean:

$$
  \bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

We know that:

* $\sum_{i=1}^{n}X_i\sim\Gamma(n,\lambda)$
* $X\sim\Gamma(\alpha,\beta)\quad c\gt0$ \\
  $\implies cX \sim\Gamma(\alpha,\beta/c)$
* So, if we take the sample mean of $n$ iid random variable, we get a Gamma distribution below:

  $$
    \bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\sim\Gamma(n,n\lambda)
  $$

**Step Two:** Find a function of the statistic and the parameter you are trying to estimate whose distribution is known and parameter free.

$$
  \bar{X}\sim\Gamma(n,n\lambda)
$$

To get rid of the unknown parameter $\lambda$

$$
  \lambda\bar{X}\sim\Gamma(n,n)
$$

How do we find critical values for the $\Gamma(n,n)$ distribution? We can get them by using R:

```{r}
qgamma(0.975,n,n)
qgamma(0.025,n,n)  
```

Or we can do by transforming into chi-squared

---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Contruct a 95% confidence interval for $\lambda$.

**Step One:** Choose a statistic.

Choose one whose distribute you know and is one that depends on the unknown parameter.

We have the sample mean:

$$
  \bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

**Step Two:** Find a function of the statistic and the parameter you are trying to estimate whose distribution is known and parameter free.

$$
  \begin{align}
    \bar{X}\sim\Gamma(n,n\lambda)&\implies\lambda\bar{X}\sim\Gamma(n,n)\\
    &\implies 2n\lambda\bar{X}\sim\Gamma\bigg(n,\frac{1}{2}\bigg)\\
    &=\Gamma\bigg(\frac{2n}{2},\frac{1}{2}\bigg)\\
    &=\chi^2(2n)
  \end{align}
$$

**Step Three:** Find appropriate critical values. We put 0.025 in both tails:

![png](\assets\images\notes\general-confidence-intervals-1.png)

**Step Four:** Put your statistic form Step Two between the critical values and solve for the unknown parameter "in the middle".

$$
  2n\lambda\bar{X}\sim\chi^2(2n)\\
  \Downarrow\\
  \chi_{0.975,2n}^2\lt2n\lambda\bar{X}\lt\chi_{0.025,2n}^2\\
  \Downarrow\\
  \frac{\chi_{0.975,2n}^2}{2n\bar{X}}\lt\lambda\lt\frac{\chi_{0.025,2n}^2}{2n\bar{X}}
$$

