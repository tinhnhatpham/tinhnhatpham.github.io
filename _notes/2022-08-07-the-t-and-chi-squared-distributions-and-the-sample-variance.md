---
layout: post
comments: false
title: The t and Chi-Squared Distributions and The Sample Variance
categories: [Confidence Intervals Involving the Normal Distribution]
---

### The Chi-Squared Distribution

There is a special gamma distribution:

$$
  X\sim\Gamma\bigg(\frac{n}{2},\frac{1}{2}\bigg)
$$

$$
  f_X(x)=\frac{1}{\Gamma(n/2)}\bigg(\frac{1}{2}\bigg)^{n/2}x^{n/2-1}e^{-\frac{1}{2}x}I_{(0,\infty)}(x)
$$

We write $X\sim\chi^2(n)$.

> A chi-squared distribution with n **degree of freedom** parameter.

![Chi-Squared Distribution](\assets\images\notes\the-chi-squared.png)

**Thing About $\chi^2(n)$**

$$
  X\sim\Gamma(\alpha,\beta)\\
  \text{Moment generation:}\\
  \implies M_X(t) = \bigg(\frac{\beta}{\beta-t}\bigg)^{\alpha}
$$

So,

$$
  X\sim\chi^2(n)=\Gamma(n/2,1/2)\\
  \implies M_X(t) = \bigg(\frac{1/2}{1/2-t}\bigg)^{n/2}
$$

Suppose that $X_1,X_2,...,X_k$ are independent radom variables with $X_i\sim\chi^2(n_i)$.

Let $Y = \sum_{i=1}^{k}X_i$

We have the moment generating function for $Y$

$$
  \begin{align}
    M_Y(t)&=E[e^{tY}]=E\bigg[e^{t\sum_{i=1}^{k}X_i}\bigg]\\
    &=E\Bigg[\prod_{i=1}^{k}e^{tX_i}\Bigg]\\
    &\text{Because these rv are independent}\\
    &\stackrel{indep}{=}\prod_{i=1}^{k}E\Big[e^{tX_i}\Big]
  \end{align}
$$

$$
  \begin{align}
    M_Y(t)&=\prod_{i=1}^{k}E\Big[e^{tX_i}\Big]=\prod_{i=1}^{k}M_{X_i}(t)\\
    &=\prod_{i=1}^{k}\bigg(\frac{1/2}{1/2-t}\bigg)^{n_i/2}\\
    &=\bigg(\frac{1/2}{1/2-t}\bigg)^{\sum_{i=1}^{k}n_i/2}
  \end{align}
$$

If we add up $k$ independent $\chi^2$ random variable, we get another random variable that has a $\chi^2$ distribution as well.

$$
  \implies Y\sim\chi^2(n_1+n_2+...+n_k)
$$

$\implies$ The sum of chi-squareds is chi-squared!

### The Chi-Squared Normal Relationship

Let $X\sim N(0,1)$.

Let $Y=X^2$.

Then $Y\sim\chi^2(1)$.

The [pdf transformation](https://tinhnhatpham.github.io/point%20estimation/2022/07/25/transformations-of-distributions.html) $Y=g(x)$is

$$
  f_Y(y) = f_X(g^{-1}(y))\Bigg|\frac{d}{dy}g^{-1}(y)\Bigg|
$$

Suppose we have two random variabe $X_1$ and $X_2$ with a joint PDF $f_{X_1}$ and $f_{X_2}$.

There is a bivariate version of this to go from $X_1$ and $X_2$ to $Y_1=g_1(X_1,X_2)$ $Y_2=g_2(X_1,X_2)$.


In 2 dimensions, the absolute value of the pdf function above called a Jacobian, which is a determinant of a matrix of partial derivatives. 

$$
  f_{Y_1,Y_2}(y_1,y_2)=f_{X_1,X_2}(g_1^{-1}(y_1,y_2),g_2^{-1}(y_1,y_2)).\mid J\mid
$$

Where

$$
  J = \begin{vmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
\end{vmatrix} \quad \text{(determinant)}
$$

### The t-Distribution

Let $Z \sim N(0,1)$ and $W\sim\chi^2(n)$ be independent random variables.

Define

$$
  T=\frac{Z}{\sqrt{W/n}}
$$

If we form the ratios as the equation above, and name that new random variable $T$. We could use the Jacobian method to find the distribution of $T$.

Do the "Jacobian transformation" with

$X1=Z \quad Y_1=\frac{X1}{\sqrt{X2/n}}$

$X_2=W \quad Y_2=g_2(X_1,X_2)$

Can show that $T$ has pdf

$$
  f_T(t)=\frac{\Gamma\big(\frac{n+1}{2}\big)}{\Gamma\big(\frac{n}{2}\big)}
  \frac{1}{\sqrt{n\pi}}\Bigg(1+\frac{t^2}{n}\Bigg)^{-(n+1)/2}\\
$$

* Mean: $0$

* Variance: $\frac{n}{n-2}$

We write $T\sim t(n)$. "A t distribution with n **degrees of freedom**".

![t Distribution](\assets\images\notes\t-distribution.png)

If we have a $t$ random variable and it has n degrees of freedom, as n gets larger and larger, that $t$ gets closer and closer to a standard normal distribution.

Suppose that

$$
  X_1,X_2,...,X_n \stackrel{iid}{\sim}N(\mu,\sigma^2)
$$

$$
  \begin{align}
    \sum_{i=1}^{n}(X_i-\mu)^2 &= \sum_{i=1}^{n}(X_i-\bar{X}+\bar{X}-\mu)^2\\
    &=\sum_{i=1}^{n}(X_i-\bar{X})^2+2(\bar{X}-\mu)\sum_{i=1}^{n}(X_i-\bar{X}) + n(\bar{X}-\mu)^2\\
    &\text{We have }\sum_{i=1}^{n}(X_i-\bar{X})=\sum_{i=1}^{n}X_i-n\bar{X}\\
    &=\sum_{i=1}^{n}X_i-n\frac{1}{n}\sum_{i=1}^{n}X_i=0
  \end{align}
$$

Now we have 

$$
  \sum_{i=1}^{n}(X_i-\mu)^2=\sum_{i=1}^{n}(X_i-\bar{X})^2 + n(\bar{X}-\mu)^2
$$

Dividing through by $\sigma^2$

$$
  \underbrace{\sum_{i=1}^{n}\bigg(\frac{X_i-\mu}{\sigma}\bigg)^2}_{Y_1}=\underbrace{\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{\sigma^2}}_{Y_2} + \underbrace{\bigg(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\bigg)^2}_{Y_3} 
$$

We will call these terms above $Y_1,Y_2,Y_3$, they are all random variables. And we're gonna talk about these distributions

**$Y_1$ distribution:**

$$
  Y_1=\sum_{i=1}^{n}\bigg(\frac{X_i-\mu}{\sigma}\bigg)^2
$$

and we know $X_i \sim N(\mu,\sigma^2)$, so:

$$
  \frac{X_i-\mu}{\sigma}\sim N(0,1)\\
  \Downarrow \\
  \bigg(\frac{X_i-\mu}{\sigma}\bigg)^2 \sim\chi^2(1)\\
  \Downarrow \\
  \sum_{i=1}^{n}\bigg(\frac{X_i-\mu}{\sigma}\bigg)^2\sim\chi^2(n)
$$

**$Y_3$ distribution:**

$$
  Y_3=\bigg(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\bigg)^2
$$

and we know $\bar{X}\sim N(\mu,\sigma^2/n)$, so:

$$
  \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\sim N(0,1)\\
  \Downarrow \\
  \bigg(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\bigg)^2\sim\chi^2(1)
$$

**$Y_2$ distribution:**

$$
  Y_1=\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{\sigma^2}
$$

We have sample variance:

$$
  S^2=\frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{n-1}\\
  \Downarrow\\
  \frac{\sum_{i=1}^{n}(X_i-\bar{X})^2}{\sigma^2}=\frac{(n-1)S^2}{\sigma^2}
$$

Sum up, we have $Y_1=Y_2+Y_3$

* $Y_1 = \sum_{i=1}^{n}\bigg(\frac{X_i-\mu}{\sigma}\bigg)^2\sim\chi^2(n)$
* $Y_2 = \frac{(n-1)S^2}{\sigma^2}=?$
* $Y_3 = \bigg(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\bigg)^2\sim\chi^2(1)$

Find the moment generating function for $Y_1$

$$
  \begin{align}
    M_{Y_1}(t) &= M_{Y_2+Y_3}(t)\\
    &\stackrel{indep}{=}M_{Y_2}(t).M_{Y_3}(t)\\
    &\implies M_{Y_2}(t)=\frac{M_{Y_1}(t)}{M_{Y_3}(t)}
  \end{align}
$$

We know $Y_1,Y_3$ are $\chi^2$ distributions.

This is the moment generating function for a Chi-Squared random variable with $(n-1)$ degrees of freedom.
$$
  \begin{align}
    M_{Y_2}(t)&=\frac{M_{Y_1}(t)}{M_{Y_3}(t)}\\
    &=\frac{\bigg(\frac{1/2}{1/2-t}\bigg)^n}{\bigg(\frac{1/2}{1/2-t}\bigg)^1}\\
    &=\bigg(\frac{1/2}{1/2-t}\bigg)^{\frac{n-1}{2}}
  \end{align}
$$

$\implies Y_2=\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$ for $X_1,X_2,...,X_n\stackrel{iid}{\sim}N(\mu,\sigma^2)$.