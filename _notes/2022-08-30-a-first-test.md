---
layout: post
comments: false
title: A First Test
categories: [Fundamental Concepts of Hypothesis Testing]
---

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Consider testing the simple versus simple hypotheses

$$
  H_0:\mu=\mu_0\qquad H_2:\mu=\mu_1
$$

where $\mu_0$ and $\mu_1$ are fixed and known.

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Suppose that $\mu_0\lt\mu_1$.

Reject $H_0$, in favor of $H_1$ if $\overline{X}\gt c$, where $c$ is to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha &= P(\text{Type I Error})\\
    &=P(\text{Reject $H_0$ when true})\\
    &=P(\overline{X}\gt c \text{ when } \mu=\mu_0)\\
    &=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\lt\frac{c-\mu_0}{\sigma/\sqrt{n}}\text{ when }\mu=\mu_0\Bigg)
  \end{align}
$$

We have

$$
  \alpha=P\Bigg(Z\gt\frac{c-\mu_0}{\sigma/\sqrt{n}}\Bigg)\\
  \text{Where } Z\sim N(0,1)
$$

![png](\assets\images\notes\a-first-test.png)

$$
  \implies\frac{c-\mu_0}{\sigma/\sqrt{n}}=z_{\alpha}\\
  \implies c = \mu_0 + z_{\alpha}\frac{\sigma}{\sqrt{n}}
$$

<font color='blue'><b>Step Four:</b></font>

Give a conclusion!

Reject $H_0$, in favor of $H_1$ if 

<font color='red'><b>

$$
  \overline{X}\gt\mu_0+z_{\alpha}\frac{\sigma}{\sqrt{n}}
$$

</b></font>

**NOTES**

If we switch $\mu_0$ and $\mu_1$

$$
  \mu_0\gt\mu_1
$$

We have

Reject $H_0$, in favor of $H_1$ if 

<font color='red'><b>

$$
  \overline{X}\lt\mu_0+z_{1-\alpha}\frac{\sigma}{\sqrt{n}}
$$

</b></font>