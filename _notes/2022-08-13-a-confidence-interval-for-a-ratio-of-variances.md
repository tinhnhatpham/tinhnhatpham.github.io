---
layout: post
comments: false
title: A Confidence Interval for a Ratio of Variances
categories: [Confidence Intervals Beyond the Normal Distribution]
---

### A Confidence Interval for a Ratio of Variances

Suppose that $X_1$ and $X_2$ are independent random variable with

$$
  X_1\sim\chi^2(n_1)\quad\text{and}\quad X_2\sim\chi^2(n_2)
$$

Define a new random variable

$$
  F = \frac{X_1/n_1}{X_2/n_2}
$$

This is called **F distribution**, it has two parameters $n_1$ and $n_2$, sometimes are called the numerator and denominator degrees of freedom. 

The F distribution can take a lot of shapes. The chi-squareds are both non-negative, so that ratio is non-negative. Here are some histograms for the F distribution:

![png](\assets\images\notes\a-confidence-interval-for-a-ratio-of-variances-1.png)

PDF of F distribution

$$
  \begin{align}
    &f(x;n_1,n_2)=\\
    &\frac{1}{B(n_1/2,n_2/2)}\bigg(\frac{n_1}{n_2}\bigg)^{n_1/2}x^{n_1/2-1}\bigg(1+\frac{n_1}{n_2}x\bigg)^{-(n_1+n_2)/2}\\
    \\
    &\text{for $x\gt0$}
  \end{align}
$$

Mean: 

$$\frac{n_2}{n_2-2}\quad \text{if}\quad n_2\gt2$$

Variance: 

$$
  \frac{2n_2^2(n_1+n_2-2)}{n_1(n_2-2)^2(n_2-4)}
$$

![png](\assets\images\notes\a-confidence-interval-for-a-ratio-of-variances-2.png)

**Proof for The Mean**

$$
  \begin{align}
    E[F]&=E\bigg[\frac{X_1/n_1}{X_2/n_2}\bigg]=\frac{n_2}{n_1}E\bigg[\frac{X_1}{X_2}\bigg]\\
    &\stackrel{indep}{=}\frac{n_2}{n_1}\underbrace{E[X_1]}_{n_1}.E\bigg[\frac{1}{X_2}\bigg]\\
    &=n_2E\bigg[\frac{1}{X_2}\bigg]\\
    &=n_2\int_{-\infty}^{\infty}\frac{1}{x}f_{X_2}(x)dx\\
    &=n_2\int_{0}^{\infty}\frac{1}{x}\frac{1}{\Gamma(n_2/2)}\bigg(\frac{1}{2}\bigg)^{n_2/2}x^{n_2/2-1}e^{-x/2}dx\\
    &=n_2\int_{0}^{\infty}\frac{1}{\Gamma(n_2/2)}\bigg(\frac{1}{2}\bigg)^{n_2/2}x^{n_2/2-2}e^{-x/2}dx\\
    &=n_2\frac{\Gamma(n_2/2-1)}{\Gamma(n_2/2)}\frac{1}{2}.\underbrace{\int_{0}^{\infty}\frac{1}{\Gamma(n_2/2-1)}\bigg(\frac{1}{2}\bigg)^{n_2/2-1}x^{n_2/2-2}e^{-x/2}dx}_{\text{intergrates to 1}}\\
    &= n_2\frac{\Gamma(n_2/2-1}{(n_2/2-1)\Gamma(n_2/2-1)}\frac{1}{2}\\
    &=\frac{n_2}{n_2-2}
  \end{align}
$$

And the point is...?

* Suppose that $X_{11},X_{12},...,X_{1,n_1}$ is a random sample of size $n_1$ from the $N(\mu_1,\sigma^2_1)$.
* Suppose that $X_{21},X_{22},...,X_{2,n_2}$ is a random sample of size $n_2$ from the $N(\mu_2,\sigma^2_2)$.

Find a $100(1-\alpha)\%$ confidence interval for the ration $\sigma_1^2/\sigma_2^2$.

Let $S_1^2$ and $S_2^2$ be the sample variances for the first and second samples, respectively.

We know that

$$
  \frac{(n_1-1)S_1^2}{\sigma_1^2}\sim\chi^2(n_1-1)\\
  \text{and}\\
  \frac{(n_2-1)S_2^2}{\sigma_2^2}\sim\chi^2(n_2-1)\\
  \text{are independent}
$$

So, we define an statistic F as

$$
  \begin{align}
    F&:=\frac{[(n_1-1)S_1^2/\sigma_1^2]/(n_1-1)}{[(n_2-1)S_2^2/\sigma_2^2]/(n_2-1)}\\
    &=\frac{\sigma_2^2}{\sigma_1^2}.\frac{S_1^2}{S_2^2}\\
  \end{align}
$$

Then

$$
  F\sim F(n_1-1,n_2-1)
$$

### Example:

Fifth grade students from two neighboring counties took a placement exam.

Group 1, from county A, consisted of 18 students. The sample mean score for these students was 77.2.

Group 2, from county B, consisted of 15 students. The sample mean score for these students was 75.3.

From the previous years of data, it is believed that the scores for both counties are normally distributed, and the variances of scores from Counties A and B, respectively, are 15.3 and 19.7.

You wish to create a confidence interval for $\mu_1-\mu_2$, the difference between the true population means.

You are thinking of using a pooled variance two-sample t-test, however this requires that the true population variances, $\sigma_1^2$ and $\sigma_2^2$ are the same.

Find a 99% confidence interval for the ration $\sigma_1^2/\sigma_2^2$. From your results, do you think it is plausible that $\sigma_1^2=\sigma_2^2$?

$$
  n_1=18,\quad s_1^2=15.3\\
  n_2=15,\quad s_2^2=19.7\\
  F:=\frac{\sigma_2^2}{\sigma_1^2}.\frac{S_1^2}{S_2^2}=\frac{15.3}{19.7}\frac{\sigma_2^2}{\sigma_1^2}
$$

Critical values:

$$
  F_{0.005,17,14}=4.1592\\
  F_{0.995,17,14}=0.2601\\
$$

Now we put our F statistic between thest numbers:

$$
  0.2601\lt\frac{15.3}{19.7}\frac{\sigma_2^2}{\sigma_1^2}\lt4.1592
$$

A 99% confidence interval for $\sigma_1^2/\sigma_2^2$ is given by $(0.18673,2.986)$.

Since this interval doesn't include $1$, it does not seem plausible that $\sigma_1^2=\sigma_2^2$ at the 99% level.

It instead seems that $\sigma_1^2\gt\sigma_2^2$.