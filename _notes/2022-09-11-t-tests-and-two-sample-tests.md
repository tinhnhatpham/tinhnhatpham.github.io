---
layout: post
comments: false
title: t-Tests and Two Sample Tests
categories: [t-Tests and Two-Sample Tests]
---

# t-Tests and Two Sample Tests

In this lab, we will take a close look at t-tests and two samples tests, in order to get an idea for how they can be applied to real data.

What is a t-test, and when do we use it? A t-test is used to compare the means of one or two samples, when the underlying population parameters of those samples (mean and standard deviation) are unknown. Like a z-test, the t-test assumes that the sample follows a normal distribution. In particular, this test is useful for when we have a small sample size, as we can not use the Central Limit Theorem to use a z-test.
 There are two kinds of t-tests:
 1. One Sample t-tests
 2. Two Sample t-tests

 We will go through both in this exercise. 
 
Before we test anything, we will need some data. In particular, some normal data. Let $\mu_{True}=10$ and $\sigma_{True}=3$ for the underlying normal population, then use the `rnorm()` function with your selected parameters to generate 20 samples from that particular normal distribution. Your final code will look something like `rnorm(20, mean=10, sd=3)`. If you want your code to be reproducable, make sure to set a seed as well using `set.seed()`.


```R
set.seed(0)
data = rnorm(20, mean=10, sd=3)
```

Now imagine that you where just given this data, without knowing the underlying parameters. How would you go about estimating the true mean of the population from the sample? We would need to do some hypothesis testing.

Let's start with a "reasonable" hypothesis that $\mu = 10$ (we know this is true, but suppose you don't for the sake of the test) and an alternate hypothesis that $\mu \ne 10$. To test this, we need to think about certian attributes of our sample:
1. Is the sample size "large" (n>30)? No, the sample size is n=20.
2. Do we know any of the underlying parameters? No, the true mean and standard deviation are unknown.
3. Is the sample approximately normally distributed? Hmm, let's check. Use the `hist()` function to plot your samples and check if they follow an approximately normal distribution.


```R
hist(data)
```


![png](\assets\images\notes\t-tests-and-two-sample-tests.png)


From there, we can see that we should use a t-test. To calculate a t-statistic from our data, it's just plugging values into the formula:

$$ \text{t-stat} = \frac{\bar{x}-\mu_0}{s/\sqrt{n}} $$

where $\bar{x}$ is the sample mean, $\mu_0$ is the true mean when assuming that the null is correct, $s$ is the sample standard deviation, and $n$ is the sample size. Then our t-statistic will follow a Student's t-distribution, which we can use to determine the probability of observing our data, given the null hypothesis. Use the `mean()` and `sd()` functions on your samples to solve for these values, and then calculate the t-statistic for your data.


```R
t_stat = (mean(data) - 10)/(sd(data)/sqrt(20))
t_stat
```


-0.00778711733273778


We have a test statistic, now we want to determine how likely it was that we observed our test statistic. We can calculate this with a p-value, just as we did with a z-test. However, there is an extra step with a t-test. 
 
The Student's t-distribution has a "Degrees of Freedom" parameter (typically annotated as $\nu$) which affects the spread of data values of the underlying distribution. The higher the degree of freedom, the less spread that will be observed. This parameter is based on the number of samples observed. 
 
Before we apply this to our sample, let's take a quick theoretical detour to see how different degrees of freedom affect the different shapes. Execute the cell below to plot the PDFs for different t-distributions, where the only difference is the degrees of freedom.


```R
x = seq(-4, 4, 0.05)

y.1 = dt(x, 1)
y.3 = dt(x, 3)
y.5 = dt(x, 5)
y.20 = dt(x, 20)

plot(0,0,xlim = c(-4,4),ylim = c(-0.01, 0.4),type = "n")
lines(x, y.1, col="blue")
lines(x, y.3, col="green")
lines(x, y.5, col="red")
lines(x, y.20, col="black")
legend("topleft", legend=c(1,3,5,20), col=c("blue", "green", "red", "black"), lty=c(1,1), title="Degrees of Freedom")
```


![png](\assets\images\notes\t-tests-and-two-sample-tests-1.png)


For a one sample t-test, the degrees of freedom is calculated as $\nu = n-1$. So, for our data, we get $\nu = 20 - 1 = 19$ degrees of freedom.

One of the ways to check the . Since we're using a two-tailed test (because our alternative hypothesis is $\mu_0 \ne 10$), we need to calculate the probability that the true mean is significantly above or below the observed data. We can solve this using the equation: $\text{p-value}= 2 F_t(-\mid t \mid, dof=n-1)$. Note that $F_t$ is the CDF of the t-distribution, which we can calculate in R using the `pt()` function, and that $t$ is the test statistic we calculated earlier.

Putting those all together, we can calculate the p-value using similar to `2*pt(-abs(test.stat), df=n-1)`.


```R
p_value = 2*pt(-abs(t_stat), 19)
p_value
```


0.993868024821185


Our null hypothesis was that $\mu=10$. Based on your results, and at a significance level of $\alpha=0.1$, does your data support or reject that null hypothesis?

Using the same data, try testing some "unreasonable" guess at the true mean, such as $\mu=100$. Does the test correctly reject the mean?


```R
# P-value with the "unreasonable" guess mu = 100
t_stat = (mean(data) - 100)/(sd(data)/sqrt(20))
print(paste0("P-value for mu = 100: ", 2*pt(-abs(t_stat), 19)))

# We have P-value is very low, so we reject the H_0: mu = 100, in favor of H_1, mu <> 100
```

    [1] "P-value for mu = 100: 1.41438476719693e-29"


The other use for t-tests are when you want to compare the mean of two different samples. This is a Two Sample t-Test.

Let your original sample be Sample A. Create a second sample (Sample B) that specifically has the *same* mean $\mu_B=10$ but a *different* standard deviation. Have 15 observations in Sample B.


```R
sample_A = data
sample_B = rnorm(15, mean = 10, sd = 5)
```

Let's test whether the two samples have the same underlying mean. In particular, we have the null hypothesis that $\mu_A - \mu_B = 0$ and the alternative hypothesis that $\mu_A - \mu_B \ne 0$.

To test this, we can calculate a test statistic that will follow the t-distribution using the equation:

$$\text{t-stat}=\frac{\bar{X}_A - \bar{X}_B}{\sqrt{s_A^2/n_A + s_B^2/n_B}}$$

Another difference from the One Sample test is the degrees of freedom, as it's based on both samples. For the Two Sample Test, we have $\nu = n_A + n_B - 2$.
 
Once we have our test statistic and degrees of freedom, we can calculate the p-value in the same was as we did for the One Sample t-test. Use the same functions as you did in the One Sample test to solve for this p-value. What are the results from the test at a significance level of $\alpha=0.1$?


```R
xbar_A = mean(sample_A)
sd_A = sd(sample_A)
n_A = 20

xbar_B = mean(sample_B)
sd_B = sd(sample_B)
n_B = 15

t_stat = (xbar_A - xbar_B)/(sqrt(sd_A^2/n_A) + sd_B^2/n_B)

p_value = 2*pt(-abs(t_stat), n_A + n_B - 2)
p_value

# With alpha = 0.1, we have a large P-value, so we fail to reject H_0, that's mean mu_A - m_B = 0 
# mu_A = mu_B
```


0.870986820082721

