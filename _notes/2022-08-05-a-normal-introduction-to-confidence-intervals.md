---
layout: post
comments: false
title: A Normal Introduction to Confidence Intervals
categories: [Confidence Intervals Involving the Normal Distribution]
---

### Introduction to Confidence Intervals

"A 95% confidence for the mean $\mu$ is given by (-2.14,30.7)."

This does **NOT** mean:

* You are 95% "confident" that the true mean $\mu$ is between -2.14 and 3.07.

* The true mean $\mu$ is between -2.14 and 3.07 with probability 0.95.

The randomness is in the sampling.

* Collect your sample.
* Estimate the parameter.
* Return a confidence interval.

If you did this again, you would **not** get the same results!

Multiple samples give multiple confidence intervals. If we keep doing it in the long run, **95% of them will correctly capture $\mu$**, 5% will miss that true mean $\mu$.

### From the Ground Up:

Suppose that $X_1,X_2,...,X_n$ is a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

Assume that $\sigma^2$ is known.

* $\bar{X}$ is an estimator of $\mu$.
* $\bar{X}$ is $N(\mu,\sigma^2/n)$.

We standardize $\bar{X}$ to a standard normal.

$$
  \frac{\bar{X}-\mu}{\sigma/\sqrt(n)}\sim N(0,1)
$$

Suppose that $Z\sim N(0,1)$. We can find 2 numbers that capture $z$ with probability 0.95. There is 2 numbers on this PDF that has area 0.95 in the middle.

If we have 0.95 is the area in the middle, then we have area 0.05/2 is the two tails left and right. So if we add the left tail 0.025 to the 0.95 to the middle. Then we want to find the number that has area 0.975 to the left. And the other number is symmetric to the mean $\mu$, so we have $n$ and $-n$ is the two numbers we need to find.

We use ```qnorm(0.975)``` in R, then it will return the number that cuts off area to the left. The result is ```qnorm(0.975 = 1.96)```.

$$
  P(-1.96 \lt Z \lt 1.96) = 0.95\\
  \implies P\Bigg(-1.96 \lt \frac{\bar{X}-\mu}{\sigma/\sqrt(n)}\lt 1.96\Bigg) = 0.95\\
  \implies P\Bigg(\bar{X}-1.96\frac{\sigma}{\sqrt{n}} \lt \mu \lt \bar{X}+1.96\frac{\sigma}{\sqrt{n}}\Bigg) = 0.95
$$

A 95% confidence interval for the mean $\mu$ of a normal distribution is given by

$$
  \Bigg(\bar{X}-1.96\frac{\sigma}{\sqrt{n}},\bar{X}+1.96\frac{\sigma}{\sqrt{n}}\Bigg)
$$

This can be written as

$$
  \bar{X}\pm1.96\frac{\sigma}{\sqrt{n}}
$$

The 1.96 is called a **critical value**. In general, the word "critical value" is used to denote any number that cuts off a certain area under the curve of a PDF.

### Summary

Suppose that $X_1,X_2,...,X_n$ is a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

A $100(1-\alpha)\%$ confidence interval for $\mu$ is given by

$$
  \bar{X}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}
$$

**Note**

Everything we did was based on the fact that $\bar{X}$ has a normal distribution.

CLT: For a more general distribution $\bar{X}$ has roughly a normal distribution for large sample $(n\gt30)$.

**An "Important Thing"**

Suppose that $X_1,X_2,...,X_n$ is a random sample from **any** distribution with mean $\mu$ and variance $\sigma^2$.

For **large n**, an **approximate** $100(1-\alpha)\%$ confidence interval for $\mu$ is given by

$$
  \bar{X}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}
$$

### Example

A supermarket chain is considering adding more
organic produce to its offerings in a certain region
of the country. They hired an external marketing
company which collected data for them.

Based on a random sample of 200 customers from
the region, they observed that the average amount
spent on organic produce, per person and per
month, was $36.

Based on past studies, it is believed that the
variance of the amount spent on produce,
organic or not, is 5 dollars.

Find a 90% confidence interval for the true
average dollar amount that all customers in the
region spent on organic produce each month.

We have:

* $n=200$
* $\bar{x}=36$
* $\sigma^2=5$

We would like to find the 2 numbers that cut off the area 0.9 in the middle.

$\alpha=1 - 0.9=0.1$, we can find the left area cut off number by using R ```qnorm(0.9 + 0.1/2)```, the result is approximately 1.645, and the other cut off number is -1.645.

Our formula: 

$$
  \bar{X}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\\
  \alpha=0.1 \quad z_{\alpha/2}=z_{0.05}=1.645
$$

$$
  36\pm1.645\frac{\sqrt{5}}{\sqrt{200}}\\
  \implies (35.74,36.26)
$$
