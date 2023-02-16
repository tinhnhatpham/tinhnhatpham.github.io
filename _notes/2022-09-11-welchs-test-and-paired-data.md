---
layout: post
comments: false
title: Welch's t-Test and Paired Data
categories: [t-Tests and Two-Sample Tests]
---

* Suppose that $X_{1,1},X_{1,2},...,X_{1,n_1}$ is a random sample of size $n_1$ from the normal distribution with mean $\mu_1$ and variance $\sigma_1^2$.

* Suppose that $X_{2,1},X_{2,2},...,X_{2,n_2}$ is a random sample of size $n_2$ from the normal distribution with mean $\mu_2$ and variance $\sigma_1^2$.

* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are <font color='red'><b>unknown</b></font> and that the samples are independent.

* <font color='red'><b>Don't assume</b></font> that $\sigma_1^2$ and $\sigma_2^2$ are <font color='red'><b>equal</b></font>!

* There is no known exact test.

* This is known as the <font color='red'><b>Behrens-Fisher problem</b></font>.

* The most popular approximates solution is given by <font color='red'><b>Welch's t-test</b></font>

Welch says that:

$$
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu2)}{\sqrt{\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}}}
$$

has an approximate t-distribution with r degrees of freedom where

$$
  r = \frac{S_1^2/n_1+S_2^2/n_2}{\frac{(S_1^2/n_1)^2}{n1-1}+\frac{(S_2^2/n_2)^2}{n2-1}}
$$

rounded down.

---

<b>Example: (In R)</b>


```R
x <- c(1.2, 3.2, 2.7, 1.6, 2.1)
y <- c(4.2, 0.8, 2.2, 2.3, 1.5, 3.0)

t.test(x,y)
```


    
    	Welch Two Sample t-test
    
    data:  x and y
    t = -0.28741, df = 8.742, p-value = 0.7805
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
     -1.543768  1.197102
    sample estimates:
    mean of x mean of y 
     2.160000  2.333333 




```R
# Calulate P-value
2*pt(-0.28741, 8.742)
```


0.780494693057154


---

<b>Example</b>

A random sample of 6 students' grades were recorded for Midterm 1 and Midterm 2.

Assuming exmam scores are nomally distributed, test whether the true (total population of students) average grade on Midterm 2 is greater than Midterm 1.

The Data

| Student | Midterm 1 Grade | Midterm 2 Grade |
|---------|-----------------|-----------------|
| 1       | 72              | 81              |
| 2       | 93              | 89              |
| 3       | 85              | 87              |
| 4       | 77              | 84              |
| 5       | 91              | 100             |
| 6       | 84              | 82              |

The data are <font color='red'><b>"paired"</b></font>.

First we compute the differences:

| Student | Midterm 1 Grade | Midterm 2 Grade | <font color='red'>Differences:<b></font></b><br>M1-M2 |
|---------|-----------------|-----------------|-------------------------------------------------------|
| 1       | 72              | 81              | 9                                                     |
| 2       | 93              | 89              | -4                                                    |
| 3       | 85              | 87              | 2                                                     |
| 4       | 77              | 84              | 7                                                     |
| 5       | 91              | 100             | 9                                                     |
| 6       | 84              | 82              | -2                                                    |

<u>The Hypotheses:</u>

Let $\mu$ be the true average difference for all students.

$$
  H_0:\mu=0\\
  H_1:\mu>0
$$

Data: $9, -4, 2, 7, 9, -2$

$$
  \sum x_i  = 23\qquad\sum x_i^2=267\qquad n=6
$$

This is simply a one sample t-test on the differences.

$\overline{x}=3.5$

$$
  s^2=\frac{\sum x_i^2-(\sum x_i)^2/n}{n-1}=32.3
$$


```R
# QQ Plot
data = c(9, -4, 2, 7, 9, -2)
qqnorm(data)
qqline(data)
```


    
![png](\assets\images\notes\welchs-test-and-paired-data.png)
    


The QQ Plot looks pretty linear!

$t_{\alpha,n-1}=t_{0.05,5}=2.01$

Reject $H_0$, in favor of $H_1$, if

$$
  \underbrace{\overline{X}}_{3.5}\gt\underbrace{\mu_0+t_{\alpha,n-1}\frac{S}{\sqrt{n}}}_{4.66}
$$

<u>Conclusion</u>

We fail to reject $H_0$, in favor of $H_1$, at 0.05 level of significance.

These data do not indicate that Midterm 2 scores are higher than Midterm 1 scores.
