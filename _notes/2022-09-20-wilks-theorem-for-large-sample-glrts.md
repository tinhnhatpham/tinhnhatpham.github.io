---
layout: post
comments: false
title: Wilks' Theorem for Large Sample GLRTs
categories: [Likelihood Ratio Tests and Chi-Squared Tests]
---

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from a distribution with pdf $f(x;\theta)$.

Let $\Theta$ be the parameter space. Assume that the parameter is one-dimensional.

Consider testing

$$
  H_0:\theta=\theta_0\quad\text{vs} H_1:\theta\ne\theta_0
$$

Let $\lambda(\stackrel{\rightharpoonup}{X})=L(\theta_0)/L(\widehat{\theta})$ be the GLR for this test.

In the parameter does not define the support for the pdf, for example as in the $\text{unif}(0,\theta)$, we have:

<font color='green'><b><u>Wilk's Theorem</u></b></font>

Under the assumption that $H_0$ is true

$$
  \color{red}{
    -2\ln\lambda(\stackrel{\rightharpoonup}{X})\stackrel{\text{d}}{\rightarrow}\chi^2(1)
  }
$$

Note that

$$
  \begin{align}
    \alpha&=P(\lambda(\stackrel{\rightharpoonup}{X})\le c;\theta_0)\\
    &=P(\ln\lambda(\stackrel{\rightharpoonup}{X})\le c_1;\theta_0)\\
    &=P(\underbrace{-2\ln\lambda(\stackrel{\rightharpoonup}{X})}_{\color{red}{\text{approximately $\chi^2(1)$}\\\text{for large sample size $n$}}}\le c_2;\theta_0)\\
    &\approx P(W\ge c_2;\theta_0)\\
    &\text{where }W\sim\chi^2(1)\\
  \end{align}
$$

And we want to solve

$$
    \alpha=P(W\gt c_2;\theta_0)\\
    \huge{\Downarrow}\\
    c_2=\chi^2_{\alpha,1}
$$

---

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from a distribution with pdf $f(x;\theta)$.

Suppose that $\theta$ is not involved in the support of $f$.

Suppose that $n$ is "large".

An approximate GLRT of size $\alpha$ for testing

$$
  H_0: \theta=\theta_0\quad\text{vs}\quad H_1:\theta\ne\theta_0
$$

is to reject $H_0$ if $\color{red}{-2\ln\lambda(\stackrel{\rightharpoonup}{X})\gt\chi^2_{\alpha,1}}$.

---

**Example**

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from the continous Pareto distribution with pdf

Fin an approximate large sample GLRT of size $\alpha=0.05$ for

$$
  H_0:\gamma=1.8\quad\text{vs}\quad H_1:\gamma\ne1.8
$$

A likelihood is

$$
  L(\gamma)=\frac{\gamma^n}{\left[\prod_{i=1}^{n}(1+x)\right]^{\gamma+1}}
$$

* The MLE for $\gamma$ is

  $$
    \widehat{\gamma}=\frac{n}{\sum_{i=1}^{n}\ln(1+X_i)}
  $$

* The restricted MLE for $\gamma$ is

  $$
    \widehat{\gamma}_0=1.8
  $$

Compute $\lambda(\stackrel{\rightharpoonup}{X})=L(\gamma_0)/L(\stackrel{\rightharpoonup}{X})$.

Compute $-2\ln\lambda(\stackrel{\rightharpoonup}{X})$.

Reject $H_0:\gamma=1.8$ if

$$
  -2\ln\lambda(\stackrel{\rightharpoonup}{X})\gt\chi^2_{\alpha,1} = 3.841459
$$

In R:

```R
qchisq(1-0.05,1)
```