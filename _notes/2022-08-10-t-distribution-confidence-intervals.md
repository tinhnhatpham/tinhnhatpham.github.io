---
layout: post
comments: false
title: t Distribution Confidence Interval
categories: [Confidence Intervals Involving the Normal Distribution]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from any distribution with mean $\mu$ and variance $\sigma^2$.

Suppose that $n$ is "large". $(n\gt30)$

Suppose that $\mu$ and $\sigma^2$ are both unknown.

By the Central Limit Theorem we know that the sample mean $\bar{X}$ is approximately normally distributed.

$$
  \begin{align}
    CLT &\implies \bar{X}\stackrel{asymp}{\sim}N(\mu,\sigma^2/n)\\
    &\implies \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\rightarrow N(0,1)\\
  \end{align}
$$

We say 

$$
  \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\stackrel{\text{approx}}{\sim}N(0,1)\quad\text{for large n}
$$

If we put the standard normal between two standard normal critical values, that will give us the right area in the middle

$$
  -z_{\alpha/2}\lt\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\lt z_{\alpha/2}
$$

And we solve for $\mu$ "in the middle". But this involves $\sigma$, which is unknown! Consider instead using the sample variance.

$$
  S^2=\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{n-1}
$$

We have a [theorem](https://tinhnhatpham.github.io/likelihood%20estimation/2022/08/04/large-sample-properties-of-mle.html) that said that if a random variable is an unbiased estimator of some parameter $\theta$, and if the variance of that random variable is going to $0$ as $n$ goes to infinity, then that random variable is converging in probability to that parameter $\theta$. Or we can say that random variable is a consistent estimator of $\theta$.

We [know that the expected value of](https://tinhnhatpham.github.io/likelihood%20estimation/2022/07/27/multiple-paramenters-and-parameters-in-the-support-of-a-distribution.html) $S^2$ is $\sigma^2$

In the convergence of probability, that this unbiased estimator is converging in probability to $\sigma^2$.
$$
  S^2=\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{n-1}\stackrel{P}{\rightarrow}\sigma^2
$$

$S^2$ is approximately $\sigma^2$ in some sense. So

$$
  \frac{\bar{X}-\mu}{S/\sqrt{n}}\sim N(0,1)
$$

We can get an approximate $100(1-\alpha)\%$ confidence interval for $\mu$ is given by solving

$$
  -z_{\alpha/2}\lt\frac{\bar{X}-\mu}{S/\sqrt{n}}\lt z_{\alpha/2}
$$

for $\mu$ "in the middle". And we got

$$
  \bar{X}\pm z_{\alpha/2}\frac{S}{\sqrt{n}}
$$

**Note for Sample Variance**

This maybe computational simplier in some cases:

$$
  \begin{align}
    S^2&=\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{n-1}\\
    &=\frac{\sum_{i=1}^{n}X_i^2 - \frac{\big(\sum_{i=1}^{n}X_i\big)^2}{n}}{n-1}
  \end{align}
$$

---

Suppose that $X_1,X_2,...,X_n$ is a random sample from any distribution with mean $\mu$ and variance $\sigma^2$.

Suppose that $n$ is "small". $(n\le30)$

Suppose that $\mu$ and $\sigma^2$ are both unknown.

A $100(1-\alpha)\%$ Confidence Interval or approximate Confidence Interval for $\mu$?

We don't have the central limit theorem giving us the X-bar is approximately normal even though we didn't start with normals, and we don't have convergence in probability approximately holding, for the sample variance to the true variance. Small samples, you need to know more information. You need to know the underlying distribution. It could be normal and it could not be normal, but you need to know it.

---

Suppose that $X_1,X_2,...,X_n$ is a random sample **from the normal distribution** with mean $\mu$ and variance $\sigma^2$.

Suppose that $n$ is "small". $(n\le30)$

Suppose that $\mu$ and $\sigma^2$ are both unknown.

A $100(1-\alpha)\%$ Confidence Interval or approximate Confidence Interval for $\mu$?

* $\bar{X}$ has a normal distribution with mean $\mu$ and variance $\sigma^2/n$.
* $\sigma^2$ and hence $\sigma^2/n$, are unknown.
* Want to use $S^2$ in place of $\sigma^2$.
* Small sample means the approximation is not good!

In order to do it, we need to know the distribution of $\frac{\bar{X}-\mu}{S/\sqrt{n}}$.

We have

$$
  \begin{align}
    \frac{\bar{X}-\mu}{S/\sqrt{n}}&=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}.\frac{\sigma}{S}\\
    &=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\Bigg/\sqrt{\frac{S^2}{\sigma^2}}\\
    &=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\Bigg/\sqrt{\frac{\frac{(n-1)S^2}{\sigma^2}}{n-1}}\\
    &=\frac{Z}{\sqrt{W/(n-1)}}\sim t(n-1)
  \end{align}
$$

If we want to find the confidence interval, we're going to want to capture the right area in the middle.

![CI](\assets\images\notes\t-distribution-confidence-interval-1.png)


And we put the $t$ random variable between the two $t$ critical values and solve $\mu$ for the middle, we get

$$
  -t_{\alpha/2,n-1}\lt\frac{\bar{X}-\mu}{S/\sqrt{n}}\lt t_{\alpha/2,n-1}\\
  \Downarrow\\
  \bar{X}\pm t_{\alpha/2,n-1}\frac{S}{\sqrt{n}}
$$

---

### Example

 A small study is being conducted to test a new sensor for a continuous glucose monitoring system. Based on previous studies, it is believed that the lifetime of the sensors measured in days is approximately normally distributed. 
 
 We take a random sample of 20 patients who were fitted with this sensor, and we want to follow them until the sensor dies out on them, and look at the time that that takes in days. In our group of 20 people, it took on average 187 days for the sensors to wear out. The sample variance we saw among those 20 patients was 16.2 days. 
 
Find a 95% confidence interval for the true sensor mean lifetime.

We have

 * $n=20$
 * $\bar{x}=187$
 * $s^2=16.2$
 * $\alpha=0.05$

To calculate $t$ in R: ```qt(0.975,19)=2.093024```

$$
  \bar{X}\pm t_{\alpha/2,n-1}\frac{S}{\sqrt{n}} \implies 187\pm2.093\frac{\sqrt{16.2}}{\sqrt{20}}
$$

The 95% confidence interval for $\mu$ is (185.11, 188.88).
