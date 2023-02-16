---
layout: post
comments: false
title: Chi-Squared Goodness of Fit Test
categories: [Likelihood Ratio Tests and Chi-Squared Tests]
---

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from some distribution.

Consider testing

$$
  \begin{align}
    &H_0: {\small{\text{The sample comes from a particular,}}\\ \small{\text{specified distribution}}}\\ 
    &\qquad\quad\text{vs.}\\
    &H_1: `` \small{\text{Not }}H_0\text{"}
  \end{align}
$$

**Examples of $H_0$:**

<b>-</b> The sample comes from a binomial distribution with parameters $8$ and $0.2$.

<b>-</b> The sample comes from a $N(0,1)$ distribution.

<b>-</b> The sample comes from this distribution.


---

### **The sample comes from this distribution.**


Here some sample values:

$$
  1,1,2,0,1,1,1,1,3,3,\\
  1,2,0,3,1,1,3,3,1,2
$$
  
Here is a distribution

| $x$      | 0   | 1   | 2   | 3   |
|----------|-----|-----|-----|-----|
| $P(X=x)$ | 0.2 | 0.4 | 0.1 | 0.3 |

Test

$$
  \begin{align}
    &H_0: \small{\text{The sample comes from this distribution.}}\\
    &\qquad\text{vs.}\\
    &H_1: \small{\text{The sample does not come from this distribution.}}
  \end{align}
$$

Collect the observed counts:

$$
  O_0=2,O_1=10,O_2=3,O_3=5\quad\text{(total is $n=20$)}
$$

When $H_0$ is true, the expected counts are:

$$
  E_0=(20)(0.2)=4\\
  E_1=(20)(0.4)=8\\
  E_2=(20)(0.1)=2\\
  E_3=(20)(0.5)=6\\
$$

Consider the test statistic:

$$
  W:=\sum_{i=0}^{3}\frac{(O_i-E_i)^2}{E_i}
$$

**Claim:** Under $H_0$, $W$ has roughly a $\underbrace{\chi^2(3)}_{n-1}$ distribution. 

This is the result of

* The Central Limit Theorem which say that

  $$
    O_i=\sum_{i=j}^{n}I_{\{X_j=i\}}
  $$

  gets normal in the limit.

* The fact that a $N(0,1)$ random variable squared has a $\chi^2(1)$ distribution.

* The fact that a sum of $k$ independent $\chi^2(1)$ random variables has a $\chi^2(k)$ distribution.

<font color='red'><b>However</b></font>, it is complicated by the fact that $O_0+O_1+O_2+O_3=20$, so these 4 random variables are not independent.

In general, for $k$ categories and $n$ observations,

$$
  W:=\sum_{i=0}^{3}\frac{(O_i-E_i)^2}{E_i}\stackrel{\text{approx}}{\sim}\chi^2(k-1)
$$

for "large" sample sizes.

<b>-</b> "Large" $n$ is not quite enough.

<b>-</b> If you have a large sample but one of the true probabilities is very small

| $x$      | 0   | 1   | 2                | 3               |
|----------|-----|-----|------------------|-----------------|
| $P(X=x)$ | 0.2 | 0.4 | 1x10<sup>2</sup> | <b>the rest</b> |

then you still will have a difficult time getting observations of the outcom 2.

<b><u>Rule of Thumb:</u></b> Want the expected number (under $H_0$) in each category to be at least $5$.

---



<i>Back to the Original Example:</i>

| $x$      | 0   | 1   | 2   | 3   |
|----------|-----|-----|-----|-----|
| $P(X=x)$ | 0.2 | 0.4 | 0.1 | 0.3 |

When $H_0$ is true, the expected counts are:

$$
  E_0=(20)(0.2)=4\\
  E_1=(20)(0.4)=8\\
  E_2=(20)(0.1)=2\\
  E_3=(20)(0.5)=6\\
$$

By the **Rule of Thumb**, we have $E_0$ and $E_2$ are less than $5$, so we will need more data!

Increased sample size to 100 and observed

$$
  O_0=18, O_1=33, O_2=12, O_3=37
$$

$$
  W:=\sum_{i=0}^{3}\frac{(O_i-E_i)^2}{E_i}\approx2.458
$$

For a test of size $\alpha=0.05$, "large" means 

$$
  W\gt\chi^2_{0.05,3}\approx7.8147
$$

<u>Conclusion:</u>

We fail to reject $H_0$ at level $0.05$.

It appears that the data did come from the distribution

| $x$      | 0   | 1   | 2   | 3   |
|----------|-----|-----|-----|-----|
| $P(X=x)$ | 0.2 | 0.4 | 0.1 | 0.3 |



```R
data_prob = c(0.2, 0.4, 0.1, 0.3)
my_sample <- sample(c(0,1,2,3), 100, replace=T, prob=data_prob)

print(table(my_sample))
```

    my_sample
     0  1  2  3 
    24 35 13 28 
    


```R
obs <- as.integer(table(my_sample))  
exp <- 100*(data_prob)

W <- sum((obs - exp)^2/exp)
W
qchisq(1-0.05,3)

# P-value = P(W_bar > W)
p_value = 1 - pchisq(W, length(obs)-1)
p_value
```


2.45833333333333



7.81472790325118



0.482868265442832



```R
chisq.test(x=obs, p=exp, rescale.p = TRUE)
```


    
    	Chi-squared test for given probabilities
    
    data:  obs
    X-squared = 2.4583, df = 3, p-value = 0.4829
    


---

### **The sample comes from a binomial distribution with parameters $8$ and $0.2$.**

You have $n$ observations of 

$$
  0\text{'s},1\text{'s},\ldots,8\text{'s}.
$$

Count up observations:

$$
  O_0,O_1,\ldots,O_8
$$

Expected numbers:

$$
  E_i=np_i
$$

where

$$
  p_i=P(X=i)={n \choose x}0.2^i(1-0.2)^{n-i}
$$

---

### **The sample comes from a $N(0,1)$ distribution**

* Continuous data!
* Group data values into bins and do the test on this finite number of categories.
* Test can be sensitive to your choice of bins!
*Try a few different bin widths. Be leery of results if they are highly variable.
