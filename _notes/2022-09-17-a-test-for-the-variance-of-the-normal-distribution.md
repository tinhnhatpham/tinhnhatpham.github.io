---
layout: post
comments: false
title: A Test for the Variance of the Normal Distribution
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1,X_2,...,Xn$ is a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2$.

Derive a test of size/level $\alpha$ for

$$
  H_0:\sigma^2\ge\sigma_0^2\quad\text{vs.}\quad H_1:\sigma^2\lt\sigma_0^2
$$

<font color='blue'><b>Step One:</b></font>

Choose a statistic/estimator for $\sigma^2$.

$$
  S^2=\frac{\sum_{i=1}^{n}(X_i-\overline{X})^2}{n-1}
$$

<font color='blue'><b>Step Two:</b></font>

Give the form of the test.

Reject $H_0$, in favor of $H_1$, if 

$$
  S^2\lt c
$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$ using $\alpha$.

$$
  \begin{align}
    \alpha&=\text{max}P(\text{Type I Error})\\
    &=\underset{\sigma^2\ge\sigma_0^2}{\text{max}}P(\text{Reject }H_0;\sigma^2)\\
    &=\underset{\sigma^2\ge\sigma_0^2}{\text{max}}P(S^2\lt c;\sigma^2)\\
    &= P\Bigg(\underbrace{\frac{(n-1)S^2}{\sigma^2}}_{\color{red}{\chi^2(n-1)}}\lt\frac{(n-1)c}{\sigma^2};\sigma^2\Bigg)\\
    &= P\Bigg(W\lt\frac{(n-1)c}{\sigma^2};\sigma^2\Bigg)\\
    &\text{Where }W\sim\chi^2(n-1).
  \end{align}
$$

We have

$$
  \alpha=\underset{\sigma^2\ge\sigma_0^2}{\text{max}}\underbrace{P\Bigg(W\lt\underbrace{\frac{(n-1)c}{\sigma^2}}_{\color{red}{\text{decreasing in }\sigma^2}}\Bigg)}_{\color{red}{\text{decreasing in }\sigma^2}}\\
$$

We know the probability decreasing in $\sigma^2$, to maximize with all the $\sigma^2\ge\sigma_0^2$, we just plug in $\sigma^2=\sigma_0^2$.

$$
  \begin{align}
    \alpha&=P\Bigg(W\lt\frac{(n-1)c}{\sigma_0^2}\Bigg)\\
  \end{align}
$$

![png](\assets\images\notes\a-test-for-the-variance-of-the-normal-distribution.png)

$$
  \Downarrow\\
  \frac{(n-1)c}{\sigma_0^2}=\chi^2_{1-\alpha,n-1}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if

$$
  \color{red}{
  S^2\lt\frac{\sigma_0^2\chi^2_{1-\alpha,n-1}}{n-1}}
$$

---

### **Example**

A lawn care company has developed and wants to patent a new herbicide applicator spray nozzle.

For safety reasons, they need to ensure that the application is consistent and not highly variable.

The company selected a random sample of 10 nozzles and measured the application rate of the herbicide in gallons per acre.

The measurements were recorded as

$$
  0.213, 0.185, 0.207, 0.163, 0.179\\
  0.161, 0.208, 0.210, 0.188, 0.195\\
$$

Assuming that the application rates are normally distributed, test the following hypotheses at level $0.04$.

$$
  H_0:\sigma^2=0.01\quad H_1:\sigma^2>0.01
$$

Reject $H_0$, in favor of $H_1$, if $S^2\gt c$.

$$
  \begin{align}
  \alpha&=P(S^2\gt c;\sigma^2=0.01)\\
  &=P\Bigg(\frac{(n-1)S^2}{\sigma^2}\gt\frac{9c}{0.01};\sigma^2=0.01\Bigg)\\
  &=P\Bigg(W\gt\frac{9c}{0.01}\Bigg)\quad(\text{where }W\sim\chi^2(9))\\
  \end{align}
$$

So

$$
  0.04=P\Bigg(W\gt\frac{9c}{0.01}\Bigg)\\
  \huge{\Downarrow}\\
  \frac{9c}{0.01}=\chi^2_{0.04, 9} = 17.61
$$

In R:

```R
qchisq(1-0.04, 9)
```


```R
x <- c(0.213, 0.185, 0.207, 0.163, 0.179, 0.161, 0.208, 0.210, 0.188, 0.195)

# Compute variance
sigma_sq = var(x)

# Or
sigma_sq = (sum(x^2)-(sum(x)^2)/10)/9

sigma_sq
```


0.000364322222222219


Reject $H_0$, in favor of $H_1$, if $S^2\gt c$.

$$
    \begin{align}
        &c = (17.61)(0.01)/9\approx0.0196\\
        &s^2= 0.000364
    \end{align}
$$

Fail to reject $H_0$, in favor of $H_1$, at level $0.04$. There is not sufficient evidence in the data to suggest that $\sigma^2\gt0.01$.


```R
# install.packages("EnvStats")
# library(EnvStats)

varTest(x, alternative = "greater", conf.level = 1-0.04, 
    sigma.squared = 0.01, data.name = NULL)
```


    $statistic
    Chi-Squared 
        0.32789 
    
    $parameters
    df 
     9 
    
    $p.value
    [1] 0.9999951
    
    $estimate
        variance 
    0.0003643222 
    
    $null.value
    variance 
        0.01 
    
    $alternative
    [1] "greater"
    
    $method
    [1] "Chi-Squared Test on Variance"
    
    $data.name
    [1] "x"
    
    $conf.int
             LCL          UCL 
    0.0001862136          Inf 
    attr(,"conf.level")
    [1] 0.96
    
    attr(,"class")
    [1] "htestEnvStats"

