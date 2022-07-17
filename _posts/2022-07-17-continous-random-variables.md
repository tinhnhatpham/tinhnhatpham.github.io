---
layout: post
title: Continous Random Variables
categories: [Probability]
---

**Definition:** A random variable is **continous** if possible values comprise either a single interval on the number line or a union of disjoint intervals.

**Note:** If $X$ is continous, $P(X = x) = 0$ for any $x$!

**The Probability Density Function (PDF):** $y = f(x)$

* The probability that $x$ is between $a$ and $b$: 
$$
P(a \le X \le b) = \int_{a}^{b}f(x)dx
$$

**Cumulative Distribution Function (CDF):**
* **Definition**: The cumulative distribution function (cdf) for a continous rv $X$ is given by $F(x) = P(X \le x) = \int_{-\infty}^{x}f(t)dt$

* $0 \le F(x) \le 1$
* $\lim_{x \to -\infty}F(x) = 0 and \lim_{x \to \infty}F(x) = 1$ 
* $F'(x) = F(x)$ by the fundamental Theorum Calculus.
* $F(x)$ is always increasing.

**Uniform Random Variable**
* **Definition**: A random variable X has the **uniform distribution** on the interval $[a,b]$ if its density function is given by:
$$
  f(x)=\begin{cases}
    \frac{1}{b-a}, & \text{if $a \le x \le b$}.\\
    0, & \text{else}.
  \end{cases}
$$
* **Notation**: $X \sim U[a,b]$

* **CDF:** 
$$
F(x) = P(X \le x) = \int_{-\infty}^{x}f(t)dt \\
= \int_{a}^{x}\frac{1}{b - a}dt \quad a \le x \le b \\
= \begin{cases}
0 & \text{if} & x \lt a \\
\frac{x - a}{b - a} & \text{if} & a \le x \le b \\
1 & \text{if} & b \lt x
\end{cases}
$$
* Expectation and Variance for a continous rv X:

$$
E(X) = \int_{-\infty}^{\infty}xf(x)dx \\
V(X) = E(X^2) - (E(X))^2
$$
* Compute expectation and variance for $X \sim U[a,b]:$

$$
E(X) = \frac{b + a}{2} \\
E(X^2) = \frac{b^2 + ab + a^2}{3} \\
$$

$$
V(X) = E(X^2) - (E(X))^2 \\
V(X) = \frac{b^2 + ab + a^2}{3} - (\frac{b + a}{2})^2 \\
V(X) = \frac{(b - a)^2}{12}
$$
