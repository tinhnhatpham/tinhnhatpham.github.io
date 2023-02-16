---
layout: post
comments: false
title: Independence and Homogeneity
categories: [Likelihood Ratio Tests and Chi-Squared Tests]
---

Test for Independent

* Two categorical variables
* Are they related?

Test for Homogeniety

* One categorical variable
* Does it act the same for different populations?

---

### **Test for Independent**

Took a random sample of 300 students from some university.

For each student, recorded their status as an undergraduate student or a graduate student.

Also recorded whether the majority of their classes are given in-person, remotely, or in a hybrid manner

<i>Question:</i> Is student status independent of class delivery type?

|         | In-person | Remote | Hybrid | Total |
|---------|-----------|--------|--------|-------|
| Ungrads | 121       | 47     | 31     | 199   |
| Grads   | 33        | 57     | 11     | 101   |
| Total   | 154       | 104    | 42     | 300   |

* Estimated probability person in sample is an undergraduate is $\frac{199}{300}$.

* Estimated probability persion in sample is taking a majority of in-person courses is $\frac{154}{300}$.

* If delivery type and student type are independent, the probability that fall into "Ungrads - In-person" category should be $\frac{199}{300}.\frac{154}{300}$

* The expected number of people, out of the total of 300 sampled, who fall into "Ungrads - In-person" category should be

  $$
    \frac{199}{300}.\frac{154}{300}.300=\frac{(199)(154)}{300}\approx102.5
  $$

* Under the assumption of independence, the expected counts are

|         | In-person                | Remote                   | Hybrid                  | Total |
|---------|--------------------------|--------------------------|-------------------------|-------|
| Ungrads | $\frac{(199)(154)}{300}$ | $\frac{(199)(104)}{300}$ | $\frac{(199)(42)}{300}$ | 199   |
| Grads   | $\frac{(101)(154)}{300}$ | $\frac{(101)(104)}{300}$ | $\frac{(101)(42)}{300}$ | 101   |
| Total   | 154                      | 104                      | 42                      | 300   |

$$
  \begin{align}
    &H_0:\small{\text{Student status and class delivery type are independent}}\\
    &H_1:\small{\text{Student status and class delivery type are not independent}}\\
  \end{align}
$$

Test statistic

$$
  W:=\sum_{i}\frac{(O_i-E_i)^2}{E_i}\sim\chi^2(?)
$$

Degrees of freedom parameter is 

$$
  \color{red}{
    (\text{number of rows}-1)(\text{number of cols}-1)
  }
$$

Reject $H_0$, in favor of $H_1$ if $W$ is "large".




```R
obs <- matrix(c(121,47,31,33,57,11), nrow = 2, ncol = 3, byrow = T)

total <- sum(obs)

exp <- matrix(0, 2, 3)

n_rows <- dim(exp)[1]
n_cols <- dim(exp)[2]

for(i in 1:n_rows){
   for(j in 1:n_cols){
        exp[i,j] = sum(obs[i,])*sum(obs[,j])/total
   }
}

W <- sum((obs - exp)^2/exp)
W
```


32.1930892630927


For our example, the test statistic is

$$
  W\approx32.193
$$

The critical value $(\alpha=0.10)$ is

$$
  \chi^2_{0.10,2}=4.60517
$$



```R
dof = (n_rows - 1)*(n_cols - 1)
dof

qchisq(1-0.10, dof)
```


2



4.60517018598809


Reject $H_0$ because $W\gt\chi^2_{0.10,2}$.

There is sufficient evidence in the data to conclude that student status and class delivery type are dependent at level $0.10$.

---

### **Test for Homogeneity**

* Took a radom sample of 199 undergraduate students.
* Took an independent random sample of 101 graduate students.
* For each sample, recorded whether the majority of their classes are given in-person, remotely, or in a hybrid manner.

<i>Question:</i> Is the distribution of in-person, to remote, to hybrid the same for both groups?

|         | In-person | Remote | Hybrid | Total |
|---------|-----------|--------|--------|-------|
| Ungrads | 121       | 47     | 31     | 199   |
| Grads   | 33        | 57     | 11     | 101   |
| Total   | 154       | 104    | 42     | 300   |

$$
  \begin{align}
    &H_0:\small{\text{The distribution of class type is the same for undergraduate and graduate students.}}\\
    &H_1:\small{\text{Not }H_0.}\\
  \end{align}
$$

* Overall "in-person" probability is estimated to be $\frac{154}{300}$.
* Since there are 199 undergrads, the expected number in the in-person group under $H_0$ is $\frac{154}{300}.199$.

<u>Note:</u>

* Expected number under the assumption of independence:

$$
  \frac{199}{300}.\frac{154}{300}.300\quad\text{(1)}
$$

* Expected number under the assumption of homogeneity:

$$
  \frac{154}{300}.199\quad\text{(Same as (1))}
$$

Because we use the same data, so we will have the same result as the assumption of independence.

