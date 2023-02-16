---
layout: post
comments: false
title: One-Tailed Tests for the Mean of a Normal Distribution
categories: [Composite Test - Power Functions - and P-Values]
---

### **Simple versus Composite**

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Consider testing the simple versus simple hypotheses

$$
  H_0:\mu=\mu_0\qquad H_1:\mu\lt\mu_0
$$

where $\mu_0$ is fixed and known.

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if $\overline{X}\lt c$, where $c$ is to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=\underset{\mu=\mu_0}{\text{max}}P(\text{Type I Error})\\
    &=\underset{\mu=\mu_0}{\text{max}}P(\text{Reject }H_0;\mu)\\
    &=P(\text{Reject }H_0;\mu_0)\\
    &=P(\overline{X}\lt c;\mu_0)
  \end{align}
$$

We know that $\overline{X}\sim N(\mu_0,\sigma^2/n)$, we transform $overline{X}$ to Standard Normal distribution:

$$
  \begin{align}
    \alpha&=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\lt\frac{c-\mu_0}{\sigma/\sqrt{n}};\mu_0\Bigg)\\
    &=P\Bigg(Z\lt\frac{c-\mu_0}{\sigma/\sqrt{n}}\Bigg)
  \end{align}
$$

![png](\assets\images\notes\one-tailed-tests-for-the-mean-of-a-normal-distribution.png)

$$
  \implies \frac{c-\mu_0}{\sigma/\sqrt{n}} = z_{1-\alpha}\\
  \implies c=\mu_0 + z_{1-\alpha}\frac{\sigma}{\sqrt{n}}
$$

<font color='blue'><b>Step Four:</b></font>

Reject $H_0$, in favor of $H_1$, if 

<font color='darkred'>
$$
  \overline{X}\lt\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}
$$
</font>

---

**NOTE:**

$$
  \begin{align}
    \beta&=\underset{\mu\lt\mu0}{\text{max}}P(\text{Type II error)}\\
    &=\underset{\mu\in H_1}{\text{max}}P(\text{Fail to Reject H_0};\mu)\\
    &=\underset{\mu\lt\mu0}{\text{max}}P\Bigg(\overline{X}\ge\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}};\mu\Bigg)\\
    &=\underset{\mu\lt\mu0}{\text{max}}P\Bigg(Z\ge\frac{\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}-\mu}{\sigma/\sqrt{n}}\Bigg)\\
    &=\underset{\mu\lt\mu0}{\text{max}}\Bigg[1-\phi\Bigg(\frac{\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}-\mu}{\sigma/\sqrt{n}}\Bigg)\Bigg] \quad\text{(1)}
  \end{align}
$$

$(1)$ get increasing in $\mu$, so in order to maximize $(1)$, we maximize the largest possible in $\mu$. We are going to maximize overall $\mu\lt\mu_0$, which means we get to plug in $\mu_0$, we get:

$$
  \begin{align}
    \beta&=1-\phi\Bigg(\frac{\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}-\mu_0)}{\sigma/\sqrt{n}}\Bigg)\\
    &=1 - \phi(z_{1-\alpha})\\
    &=1-\alpha
  \end{align}
$$

---

### **Composite versus Composite**

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Consider testing the simple versus simple hypotheses

$$
  H_0:\mu\ge\mu_0\qquad H_1:\mu\lt\mu_0
$$

where $\mu_0$ is fixed and known.

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if $\overline{X}\lt c$, where $c$ is to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=\underset{\mu\ge\mu_0}{\text{max}}P(\text{Type I Error})\\
    &=\underset{\mu\ge\mu_0}{\text{max}}P(\text{Reject }H_0;\mu)\\
    &=\underset{\mu\ge\mu_0}{\text{max}}P(\overline{X}\lt c;\mu_0)\\
    &=\underset{\mu\ge\mu_0}{\text{max}}P\Bigg(Z\lt\frac{c-\mu}{\sigma/\sqrt{n}}\Bigg)\\
    &=\underset{\mu\ge\mu_0}{\text{max}}\quad\phi\Bigg(\underbrace{\frac{c-\mu}{\sigma/\sqrt{n}}}_{\text{decreasing in }\mu}\Bigg)\\
  \end{align}
$$

This is a decreasing function of $\mu$. It's going down as $\mu$ gets larger. If we want to be largest (max), we need to take the smallest $\mu$, so $\mu=\mu_0$. We have

$$
  \begin{align}
    &\alpha=\phi\Bigg(\frac{c-\mu_0}{\sigma/\sqrt{n}}\Bigg)\\
    &\implies \frac{c-\mu_0}{\sigma/\sqrt{n}}=z_{1-\alpha}\\
    &\implies c=\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}
  \end{align}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion:

Reject $H_0$, in favor of $H_1$, if

<font color='darkred'>
$$
  \overline{X}\lt\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}
$$
</font>

### **Example**

In 2019, the average health care annual premium for a family of 4 in the United States, was reported to be $6,015.

In a more recent survey, 100 randomly sampled families of 4 reported an average annual health care premium of $6,537.

Can we say that the true average is  currently greater that $6,015 for all families of 4?

Assume that annual health care premiums are normally distributed with a standard deviation of $814.

Let $\mu$ be the true average for all families of 4.

<font color='blue'><b>Step Zero:</b></font>

Set up the Hypotheses.

$$
  H_0:\mu=6015\qquad H_1:\mu\gt6015
$$

Decide on a level of significance.

$$
  \alpha=0.10
$$

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if 

$$\overline{X}\gt c$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=\underset{\mu=\mu_0}{\text{max}}P(\text{Type I Error};\mu)\\
    &=P(\text{Type I Error};\mu_0)\\
    &=P(\overline{X}\gt c;\mu_0)\\
    &=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\gt\frac{c-6015}{814/\sqrt{100}};\mu_0\Bigg)\\
    &=P\Bigg(Z\gt\frac{c-6015}{814/\sqrt{100}}\Bigg)
  \end{align}
$$

![png](\assets\images\notes\one-tailed-tests-for-the-mean-of-a-normal-distribution-1.png)

We can use ```qnorm(0.90)=1.28``` to calculate the critical value.

$$
  \begin{align}
    &\implies \frac{c-6015}{814/\sqrt{100}}=1.28\\
    &\implies c=6119.19
  \end{align}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if 

$$
  \overline{X}\gt6119.19
$$

From the data, where $\overline{x}=6537$, we reject $H_0$ in favor of $H_1$.

The data suggests that the true mean annual health care premium is greater than $6015.