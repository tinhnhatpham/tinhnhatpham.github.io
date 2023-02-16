---
layout: post
comments: false
title: Mean square error, bias, and relative efficiency
categories: [Likelihood Estimation]
---

### Definition

**Variance, MSE and Bias**

Let $\hat{\theta}$ be an estimator of a parameter $\theta$. The **mean squared error** of $\hat{\theta}$ is denoted and defined by

$$
  MSE(\hat{\theta})=E[(\underbrace{\hat{\theta} - \theta}_\text{error})^2]
$$

Note: if $\hat{\theta}$ is an unbiased estimator of $\theta$, its **mean squared error** is siply the variance of $0$.

The **bias** of $\hat{\theta}$ is denoted and defined by

$$
  B(\hat{\theta}) = E[\hat{\theta}] - \theta
$$

An unbiased estimatorhas a bias of zero.

$$
  MSE(\hat{\theta})=Var[\hat{\theta}] + (B[\hat{\theta}])^2
$$

**Relative Efficiency**

Let $\hat{\theta_1}$ and $\hat{\theta_2}$ be two unbiased estimators of a parameter $\theta$. $\hat{\theta_1}$ is **more efficient** than $\hat{\theta_2}$ if 

$$
  Var[\hat{\theta_1}] \lt Var[\hat{\theta_2}]
$$

The **relative efficiency** of $\hat{\theta_1}$, relative to $\hat{\theta_2}$ is denoted/defined as

$$
  Eff(\hat{\theta_1},\hat{\theta_2}) = \frac{Var[\hat{\theta_1}]}{Var[\hat{\theta_2}]}
$$