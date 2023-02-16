---
layout: post
comments: false
title: The t and Chi-Squared Distribution
categories: [t-Tests and Two-Sample Tests]
---

### **A Few Continous Distributions:**

---

**The Normal Distribution**

Two Parameters:

* Mean: $-\infty\lt\mu\lt\infty$
* Variance: $\sigma^2\gt0$

The Probability Density Function:

$$
    f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2\sigma^2}(x-\mu)^2}
$$

$X\sim N(\mu, \sigma^2)$

![png](\assets\images\notes\the-t-and-chi-squared-distributions.png)

---

**The Exponential Distribution**

One parameter:

* Rate: $\lambda\gt0$

The Probability Density Function:

$$
    f(x) = \lambda e^{-\lambda x}\\
    \text{for $x\gt0$}
$$

* Mean:

    $$
        \begin{align}
            \mu=E[X]&=\int_{-\infty}^{infty}xf(x)dx\\
            &=\int_{0}^{\infty}x\lambda e^{-\lambda x}dx\\
            &=\frac{1}{\lambda}
        \end{align}
    $$

* Variance:

    $$
        \begin{align}
            \sigma^2=Var[X]&=E[(X-\mu)^2]\\
            &=E[X^2] - (E[x])^2\\
            &=\frac{1}{\lambda^2}
        \end{align}
    $$

 * Notation
 
    $$
        X\sim exp(\text{rate}=\lambda)\\
        f(x)=\lambda e^{-\lambda x}
    $$
    
    Or
    
    $$
        X\sim exp(\text{mean}=\lambda\\
        f(x)=\frac{1}{\lambda}e^{-x/\lambda}
    $$

![png](\assets\images\notes\the-t-and-chi-squared-distributions-1.png)

---

**The Gamma Distribution**

Two parameters:

* Shape: $\alpha\gt0$
* Inverse Scale: $\beta\gt0$

The Probability Density Function:

$$
    f(x)=\frac{1}{\Gamma(\alpha)}\beta^{\alpha}x^{\alpha-1}e^{-\beta x}\\
    \text{for $x\gt0$}
$$



![png](\assets\images\notes\the-t-and-chi-squared-distributions-2.png)

![png](\assets\images\notes\the-t-and-chi-squared-distributions-3.png)

![png](\assets\images\notes\the-t-and-chi-squared-distributions-4.png)

<u>The Gamma Function:</u>

$$
    \Gamma(\alpha)=\int_{0}^{\infty}x^{\alpha-1}e^{-x}dx
$$

Properties:

* $\Gamma(1)=1$

* $\Gamma(\alpha)=(\alpha-1)\Gamma(\alpha-1)\quad\text{for }\alpha\gt1$

* $\Gamma(n)=(n-1)!\quad\text{for an integer }n\gt1$

$$
    X\sim\Gamma(\alpha,\beta)
$$

* Mean:

    $$
        \begin{align}
            \mu&=E[X]=\int_{-\infty}^{\infty}xf(x)dx\\
            &=\int_{0}^{\infty}x\frac{1}{\Gamma(\alpha)}\beta^\alpha x^{\alpha-1}e^{-\beta x}dx\\
            &=\frac{\alpha}{\beta}
        \end{align}
    $$

* Variance:

    $$
        \begin{align}
            \sigma^2=Var[X]&=E[(X-\mu)^2]\\
            &=E[X^2]-(E[X])^2\\
            &=\frac{\alpha}{\beta^2}
        \end{align}
    $$
    
---

**The Chi-Squared Distribution**

One Parameter:

* Degrees of freedom: $n\ge1\quad\text{(n is an integer)}$

$$
    X\sim\chi^2(n)\\
    \text{is defined as}\\
    \Gamma\bigg(\frac{n}{2},\frac{1}{2}\bigg)
$$

* Mean:

    $$
        \mu=E[X]=n
    $$

* Variance:

    $$
            \sigma^2=Var[X]=2n
    $$
    
![png](\assets\images\notes\the-t-and-chi-squared-distributions-5.png)

**The t-distribution**

Let $Z\sim N(0,1)$ and $W\sim\chi^2(n)$ be independent random variables.

Define

$$
    T=\frac{Z}{\sqrt{W/n}}
$$

Then $T$ has pdf.

$$
  f(x)=\frac{\Gamma\big(\frac{n+1}{2}\big)}{\sqrt{n\pi}\Gamma\big(\frac{n}{2}\big)}
  \Bigg(1+\frac{x^2}{n}\Bigg)^{-(n+1)/2}\\
  (-\infty\lt x\lt\infty)
$$

We write $X\sim t(n)$

![png](\assets\images\notes\the-t-and-chi-squared-distributions-6.png)

For critical value

![png](\assets\images\notes\the-t-and-chi-squared-distributions-7.png)

Some facts:

* $Z\sim N(0,1)\implies Z^2\sim\chi^2(1)$
    
* $X_1,X_2,...,X_k$ independent with $X_i\sim\chi^2(n_i)
    
    $$
        \sum_{i=1}^{k}X_i\sim\chi^2(n_1+n_2+...+n_k)
    $$

    In particular, $X_1,X_2,...,X_n\stackrel{iid}{\sim}\chi^2(1)$
    
    $$
        \implies\sum_{i=1}^{n}X_i\sim\chi^2(n)
    $$
    
* $X\sim\Gamma(\alpha,\beta)$ and $c\gt0$
    
    $$
        \implies cX\sim\Gamma(\alpha,\beta/c)
    $$
    
* $X\sim\Gamma(\alpha,\beta)$

    $$
        f(x)=\frac{1}{\Gamma(\alpha)}\beta^\alpha x^{\alpha-1}e^{-\beta x}
    $$
    
    $\alpha=1\implies X\sim exp(\text{rate}=\beta)$

* $X_1,X_2,...,X_n\stackrel{iid}{\sim}exp(\text{rate}=\lambda)$

    $$
        \sum_{i=1}^{n}X_i\sim\Gamma(n,\lambda)
    $$
    
* $X_1,X_2,...,X_n\stackrel{iid}{\sim}\Gamma(\alpha,\beta)$

    $$
        \implies\sum_{i=1}^{n}X_i\sim\Gamma(n\alpha,\beta)
    $$
    
Things we now know

* $X_1,X_2,...,X_n\stackrel{iid}{\sim}exp(\text{rate}=\lambda)$

    $$
        \begin{align}
            \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\sim\Gamma(n, n\lambda)\\
            \implies 2n\lambda\overline{X}&=\Gamma\bigg(n,\frac{1}{2}\bigg)\\
            &=\Gamma\bigg(\frac{2n}{2},\frac{1}{2}\bigg)=\chi^2(2n)
        \end{align}     
    $$
    
* $X_1,X_2,...,X_n\stackrel{iid}{\sim}\Gamma(\alpha,\beta)$

    $$
        \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\sim\Gamma(n\alpha,n\beta)
    $$
