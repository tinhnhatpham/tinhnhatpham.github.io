---
layout: post
comments: false
title: Composite Hypotheses and Level of Significance
categories: [Composite Test - Power Functions - and P-Values]
---

### Type II Error

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Consider testing the simple versus simple hypotheses

$$
  H_0:\mu=\mu_0\qquad H_1:\mu=\mu_1
$$

where $\mu_0$ and $\mu_1$ are fixed and known.

Suppose that <font color='red'>$\mu_0\lt\mu_1$</font>.

<u>The Test:</u>

Reject $H_0$, in favor of $H_1$ if

$$
  \bar{X} \gt \mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}}\quad\text{(Type I Error)}
$$

<u>Question:</u>

What about the Type II error?

![png](\assets\images\notes\errors-in-hypothesis-testing.png)

**Type II Error**

It is locked in!

$$
  \begin{align}
    \beta&=P(\text{Type II Error})\\
    &=P(\text{Fail to Reject $H_0$ when false})\\
    &=P\Bigg(\overline{X}\le\mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}}\text{ When $\mu=\mu_1$}\Bigg)\\
    &=P\Bigg(\overline{X}\le\mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}};\mu_1\Bigg)\\
  \end{align}
$$

We know that $\overline{X} \sim N(\mu_1, \sigma^2/\sqrt{n})$. We transform $\overline{X}$ to Standard Normal distribution:

$$
  \begin{align}
    \beta&=P\Bigg(\frac{\overline{X}-\mu_1}{\sigma/\sqrt{n}}\le\frac{\mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}}-\mu_1}{\sigma/\sqrt{n}};\mu_1\Bigg)\\
    &=P\Bigg(Z\le\underbrace{\frac{\mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}}-\mu_1}{\sigma/\sqrt{n}}}_{(1)}\Bigg)
  \end{align}
$$

(1) is a fixed number, so compute the probability and that's your $\beta$!

We could create the entire test starting from the "$\beta$ point of view" and then $\alpha$ would be locked in.

If we want to set both $\alpha$ and $\beta$ we would have to free up the sample size as another unknown. ($c$ and $n$)

**Note:** $\beta\neq1-\alpha$

---

### Composite vs Composite

$X_1,X_2,...,X_n\sim N(\mu,\sigma^2)$, $\sigma^2$ known

$$
  H_0:\mu\le\mu_0\quad\text{versus}\quad H_1:\mu\gt\mu_0
$$

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if $\overline{X}\gt c$, where $c$ is to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=P(\text{Type I Error})\\
    &=P(\text{Reject $H_0$ when true})\\
    &=P(\overline{X}\gt c\text{ when }\mu\le\mu_0)\\
    &= \quad\color{red}?
  \end{align}
$$

The definitions we have used for $\alpha$ and $\beta$ are for simple hypotheses only.

* The <font color='red'>level of significance</font> or "<font color='red'>size</font>" of a test is denoted by $\alpha$ and is defined by

$$
  \begin{align}
    \alpha&=\text{max}P(\text{Type I Error})\\
    &=\underset{\mu\in H_0}{\text{max}}P(\text{Reject $H_0$};\mu)\\
    \beta&=\text{max}P(\text{Type II Error})\\
    &=\underset{\mu\in H_1}{\text{max}}P(\text{Fail to Reject $H_0$};\mu)
  \end{align}
$$

<font color='green'><b>Definitions</b></font>

* $1-\beta$ is known as the <font color='red'>power of the test</font>

  $$
    \begin{align}
      1-\beta&=1-\underset{\mu\in H_1}{\text{max}}P(\text{Fail to Reject $H_0$};\mu)\\
      &= \underset{\mu\in H_1}{\text{min}}\bigg(1 - P(\text{Fail to Reject $H_0$};\mu)\bigg)\\
      &=\underset{\mu\in H_1}{\text{min}}P(\text{Reject }H_0;\mu)
    \end{align}
  $$

* $\beta$ is the probability making a **Type II error**, really the maximum. This mean the null hypothesis is false and we are failing to reject it. So $1 - P(\text{Fail to reject $H_0$})$ is its compliment, and that is the probability we do reject the hypothesis. That mean we want $1-\beta$ to be large ($\beta$ was a particular error, so we certainly want that to be small). and that $1-\beta$ is called <font color='red'>power of the test</font>. So we can say **high power is good!**
