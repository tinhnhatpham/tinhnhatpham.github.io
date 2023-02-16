---
layout: post
comments: false
title: The Generalized Likelihood Ratio Test
categories: [Likelihood Ratio Tests and Chi-Squared Tests]
---

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from a distribution with pdf $f(x;\theta)$.

Let $\Theta$ be the parameter space.

Consider testing

$$
  H_0:\theta\in\Theta_0\quad\text{vs}\quad H_1:\theta\in\Theta\backslash\Theta_0
$$

* Let $\widehat{\theta}$ be the maximum likelihood estimator for $\theta$.

* Let $\widehat{\theta}_0$ be <font color='red'><b>restricted MLE</b></font>.

  $\widehat{\theta}_0$ is the MLE for $\theta$ if we assume that $H_0$ is true.

Consider for these examples:

---

$$
  H_0:\theta\le\theta_0\quad\text{vs}\quad H_1:\theta\gt\theta_0
$$

Suppose that the likelihood looks like this:

![png](\assets\images\notes\the-generalized-likelihood-ratio-test.png)

* The maximum likelihood estimator $\widehat{\theta}$ is the value that maximizes the $L(\theta)$.

* The restricted likelihood is the value that maximizes over the region where the null hypothesis is true.

---

$$
  H_0:\theta\ge\theta_0\quad\text{vs}\quad H_1:\theta\lt\theta_0
$$

Suppose that the likelihood looks like this:

![png](\assets\images\notes\the-generalized-likelihood-ratio-test-1.png)

* Because the maximum likelihood estimator $\widehat{\theta}$ and the null hypothesis are in the same region. This is actually equivalent to the restricted MLE.

---

$$
  H_0:\theta\le\theta_0\quad\text{vs}\quad H_1:\theta\gt\theta_0
$$

Suppose that the likelihood looks like this:

![png](\assets\images\notes\the-generalized-likelihood-ratio-test-2.png)

* The null hypothesis is the region from $-\infty$ to $\theta_0$. In this case the restricted likelihood $\widehat{\theta}_0=\theta_0$.

---

$$
  H_0:\theta=\theta_0\quad\text{vs}\quad H_1:\theta\gt\theta_0
$$

Suppose that the likelihood looks like this:

![png](\assets\images\notes\the-generalized-likelihood-ratio-test-3.png)

* For the simple null hypothesis, we only looking for one point when the null hypothesis is true, and that is restricted MLE $\widehat{\theta}_0=\theta_0$.
 
---

<font color='green'><b>Definition:</b></font>

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from a distribution with pdf $f(x;\theta)$.

Consider testing

$$
  H_0:\theta=\theta_0\quad\text{vs}\quad H_1:\theta\ne\theta_0
$$

Let $L(\theta)$ be a likelihood function.

The <font color='red'><b>generalized likelihood ratio</b></font> (GLR) is

$$
  \lambda(\stackrel{\rightharpoonup}{X}) = \frac{L(\widehat{\theta}_0)}{L(\widehat{\theta})}
$$

The <font color='red'><b>generalized likelihood ratio test</b></font> (GLRT) says to reject $H_0$, in favor of $H_1$, if

$$
  \lambda(\stackrel{\rightharpoonup}{X})\le c
$$

---

**Example**

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from the continous Pareto distribution with pdf

$$
  f(x,\gamma)=\begin{cases}
    \begin{align}
      &\frac{\gamma}{(1+x)^{\gamma+1}}&&x\gt0\\
      &0&&\text{otherwise}\\
    \end{align}
  \end{cases}
$$

Here, $\gamma\gt0$ is a parameter.

Find the GLRT of size $\alpha$ for

$$
  H_0:\gamma=\gamma_0\quad\text{vs}\quad H_1:\gamma\ne\gamma_0
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
    \widehat{\gamma}_0=\gamma_0
  $$

The GLR is 

$$
  \lambda(\stackrel{\rightharpoonup}{X})=\frac{L(\widehat{\gamma}_0)}{L(\widehat{\gamma})}
$$

* The form of the test is to reject $H_0$, in favor of $H_1$, if $\lambda(\stackrel{\rightharpoonup}{X})\le c$, where $c$ is determined by solving

  $$
    P(\lambda(\stackrel{\rightharpoonup}{X})\le c;\gamma_0)=\alpha
  $$

  And we can make some simplifications or standardization, or something, and we turn it to some other function

  $$
    P(g(\stackrel{\rightharpoonup}{X})\quad?\quad c_1;\gamma_0)=\alpha
  $$