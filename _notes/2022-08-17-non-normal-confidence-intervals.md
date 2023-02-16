---
layout: post
comments: false
title: Non-normal Confedence Intervals
categories: [Confidence Intervals Beyond the Normal Distribution]
---

# Non-Normal Confidence Intervals in R

During the lecture videos, we took a look at how we would construct a confidence interval for the rate parameter $\lambda$ of an exponential distribution. In particular, we determined how to construct confidence intervals using two different statistics: the mean $\bar{X}$ and the minimum value.

Let's use R to simulate an exponential distribution, and see if our confidence intervals actually work the way we think they do. 

Start by generating 15 samples from an $Exp(5)$ distribution. To do this, use the `rexp()` function with parameter `rate=5`.


```{r}
print(rexp(15, rate=5))
```

     [1] 0.135942420 0.307428097 0.102820149 0.226704191 0.339820451 0.264204984
     [7] 0.020370079 0.048234829 0.117016397 0.093518215 0.100688136 0.009782544
    [13] 0.051980649 0.146598687 0.156534183


That's a single sample. If we want to look at multiple confidence intervals, we're going to want multiple samples. Let's look at 100 samples in total.

We can calculate multiple samples at once, and store them into a matrix using the `matrix()` function. For our matrix, each sample will be a seperate row, and there will be 15 columns because we're generating 10 data points per sample, so our final function call will look like `matrix(rexp(15*100, rate=5), ncol=10)`. Use `set.seed(0)` to make sure you get the same values each time.


```{r}
set.seed(0)
data = matrix(rexp(15*100, rate=5), ncol=10)
print(dim(data))
```

    [1] 150  10


Let's start by looking at the confidence interval that we made with the mean. From the lectures, we eventually got the equation

$$ \dfrac{\chi^2_{0.025, 2n}}{2n\bar{X}} \le \lambda \le \dfrac{\chi^2_{0.975, 2n}}{2n\bar{X}} $$

If you don't remember how we got these equations, please go over your notes to make sure you understand. For now, we can use these equations to create our confidence intervals. For the $\chi^2$ values, we can use the `qchisq()` function with 15*2 degrees of freedom. For the lower bound, want `qchisq(0.025, 30)` and the upper bound we want `qchisq(0.975, 30)`.

We want to calculate the upper and lower bounds for each of our samples. We could use a for loop for this, but it would be more efficient to use the `apply()` function. To do that, we will need to define our calculations as functions. Fill in the functions below with the necesary calculations for each bound of the confidence interval.


```{r}
exp.conf.int.lower = function(sample){

    # Your Code Here
    qchisq(0.025, 30)/(30*mean(sample))
}

exp.conf.int.upper = function(sample){
    
    # Your Code Here
    qchisq(0.975, 30)/(30*mean(sample))
}
```

Now we have our two functions defined, we can use them to calculate the upper and lower bounds for the entire matrix. We will define the apply function as `apply(data, 1, exp.conf.int.lower)`.
* `data` is the variable containing your matrix of samples. It is the data that the function will be applied to.
* `1` means we are applying our function to each row of the data. This is the "margin" that we apply the function along. If we wanted to apply the function to each column, we would input `2`.
* `exp.conf.int.lower` is the function that we will apply to the data.

Use this function for both the `exp.conf.int.lower` and `exp.conf.int.upper` functions to get vectors of the numeric upper and lower bounds for each sample.


```{r}
lower.bound = apply(data, 1, exp.conf.int.lower)
upper.bound = apply(data, 1, exp.conf.int.upper)
```

How many of these confidence intervals contain the true parameter? Recall that you generated your data with a know rate $\lambda=5$.

We can perform an arithmetic comparison between a number and a vector in R, and R is smart enough to perform that operation between that number and each element of that vector. That's a complicated way of saying we can do `lower.bound < 5` and `5 < upper.bound`, and get vectors of boolean values for the "truth value" of each element's comparison. In total, want the condition `lower.bound < 5 < upper.bound`, so we can use a logical "and" to combine the two boolean vectors.

Combining all that, we get `(lower.bound < 5) & (5 < upper.bound)`. This gives us a vector of boolean values, where each boolean is whether or not $\lambda=5$ was within the confidence interval. To find the total number, we can calculate the sum of this vector. Does this value match the original confidence?


```{r}
sum((lower.bound < 5) & (5 < upper.bound))
```


134


Great! That's our first confidence interval done! Now let's move to the second confidence interval, which used the minimum value of the sample. Recall from the lectures that the 95% confidence interval was defined as:

$$ 0 \le \lambda \le \dfrac{-ln(0.05)}{nY_n} $$

where $Y_n = min(X_1,X_2, \dots, X_n)$.

Let's repeat the process above of defining a function for the bounds. Because we know that the lower bound is 0, we don't have to worry about defining anything for that. Complete the function below to solve for the upper bound of the confidence interval.

Note: We defined the lower bound at 0, for simplicity. This doesn't have to be the case. We could have a 95% confidence interval that had a lower bound at a different value, which would cause the upper bound to shift higher, as the confidence interval would have to continue containing 95% of the area.


```{r}
ci.upper.bound.min = function(sample){
    
    # Your Code Here
    -log(0.05)/(length(sample)*min(sample))
}
```

Again, use the `apply()` function to calculate the upper bound for the confidence intervals of each sample's data. Then compute whether `0 <= 5 <= upper.bound` for each sample. How many times the confidence interval contained the true rate? Does it match the 95% confidence level of the interval?


```{r}
upper.bound = apply(data, 1, ci.upper.bound.min)
sum(upper.bound >= 5)
```


143

