---
title: A First Course in Bayesian Statistical Methods Chapter 3 . One parameter model, 2. Poisson Model & Exponential Families and Conjugate Priors
date: 2021-03-09T21:57:27+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 3. 파라미터가 하나인 모델(One-parameter models)

## 3.2 포아송 모델(Poisson model)

어떤 사람의 자식의 수, 친구의 수 같은 숫자를 세는데 있어 가장 간단한 확률 모델은 포아송 모델입니다. 그리고 이 때의 표본 공간은 $\mathcal{Y}$ = {0,1,2,...} 입니다.

### 포아송 분포

다음과 같은 분포를 평균이 $\theta$인 포아송 분포라고 합니다.

$$
Pr(Y=y|\theta) = \text{scipy.stats.poisson.pmf(y},\theta) = \frac{\theta^y e^{-\theta}}{y!} \ \ for \ y \in (0,1,2,...).
$$

그리고 이러한 확률 변수는

* $E[Y|\theta] = \theta$
* $Var[Y|\theta] = \theta$

입니다.

포아송과 같은 분포들은 평균이 큰 분포일 수록, 분산 또한 큽니다. 그래서 "평균-분산 관계"가 있다고 말합니다.

![IMG_C967558CC4F7-1](https://user-images.githubusercontent.com/57588650/110325291-7ac3db00-805a-11eb-90c2-87eafc893c8a.jpeg)

**그림. 3.7.** 포아송 분포입니다. 왼쪽 그래프의 까만 막대는 평균이 1.83인 포아송 분포이고, 회색 막대는 실제로 1990년대에 30세 여성이 가지는 자녀의 수의 분포입니다. 오른쪽의 그래프는 10개의 평균이 1.83인 i.i.d 포아송 확률을 더한 분포입니다. 이것은 평균이 18.3인 포아송 분포와 같습니다.

### 3.2.1 사후 분포 추론

이항 분포 예시와 같은 방식으로 포아송 샘플 데이터의 결합 pdf를 구해보면 다음과 같습니다.

$$
Pr(Y_1 = y_1, ..., Y_n = y_n | \theta) = \prod^n_{i-1} p(y_i|\theta)
$$

$$
= \prod^n_{i=1} \frac{1}{y_i!} \theta^{\Sigma y_i e^{\theta}}
$$

$$
= c(y_1, ..., y_n) \theta^{\Sigma y_i e^{-n \theta}}
$$

두 개의 $\theta$값 각각을 가지는 사후 확률을 비교해보면 

$$
\frac{p(\theta_a|y_1, ..., y_n)}{p(\theta_b|y_1, ..., y_n)} = \frac{c(y_1, ... , y_n)}{c(y_1, ..., y_n)} \frac{e^{-n\theta_a}}{e^{-n\theta_b}} \frac{\theta_a^{\Sigma y_i}}{\theta_b^{\Sigma y_i}} \frac{p(\theta_a)}{p(\theta_b)}
$$

$$
= \frac{e^{-n\theta_a}}{e^{-n\theta_b}} \frac{\theta_a^{\Sigma y_i}}{\theta_b^{\Sigma y_i}} \frac{p(\theta_a)}{p(\theta_b)}
$$

와 같습니다.

이항 모델과 마찬가지고, i.i.d인 포아송 모델에서 $\Sigma^n_{i=1} Y_i$는 그 데이터에서 얻을 수 있는 $\theta$에 대한 모든 정보를 가지고 있습니다(즉 충분 통계량입니다.). 즉

$$
\Sigma^n_{i=1} Y_i | \theta \sim \text{Poisson}(n\theta)
$$

입니다.

### 켤레 사전 분포(Conjugate Prior)

**그림. 3.7.** 포아송 분포입니다. 왼쪽 그래프의 까만 막대는 평균이 1.83인 포아송 분포이고, 회색 막대는 실제로 1990년대에 30세 여성이 가지는 자녀의 수의 분포입니다. 오른쪽의 그래프는 10개의 평균이 1.83인 i.i.d 포아송 확률을 더한 분포입니다. 이것은 평균이 18.3인 포아송 분포와 같습니다.

전 포스팅에서 배웠듯, 만약 어떠한 사전 분포의 클래스가 사후 분포의 클래스와 같다면 그 사전 분포를 표본 모델 $p(y_1, ..., y_n | \theta)$에 대해 켤레(conjugate)라고 합니다. 포아송 표본 모델에서 $\theta$에 대한 우리의 사후 분포는 다음과 같은 형태를 따릅니다:

$$
p(\theta|y_1,...,y_n) \propto p(\theta) \times p(y_1, ..., y_n|\theta)
$$

$$
\propto p(\theta) \times \theta^{\Sigma y_i} e^{-n\theta}
$$

이것은 켤레 사전분포가 무엇이든지 $\theta^{c_1} e^{-c_2\theta}$와 같은 꼴의 항을 가지고 있다는 것을 의미합니다. 이러한 형식을 가지고 있는 분포 중 가장 간단한 것은 이 항만을 가지고 있는 감마 분포족(the family of gamma distribution)입니다.

## 감마 분포

미지의 양수 $\theta$는 만약 다음과 같다면 gamma(a,b) 분포를 가집니다.

$$
p(\theta) = \text{scipy.stats.gamma.pdf}(\theta,a,scale = 1/b) = \frac{b^a}{\Gamma(a)} \theta^{a-1} e^{-b\theta}, \ \ for \ \theta, a, b > 0
$$

이러한 확률변수는 다음과 같은 통계량을 가집니다.

* $E[\theta] = a/b$
* $Var[\theta] = a/b^2$
* $mode[\theta] = \begin{cases} (a-1)/b \ \ \ \ if \ a>1 \newline 0 \ \ \ \ if \ a \leq 1 \end{cases}$


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(14,15))
import numpy as np
from scipy.stats import gamma

# Panel 1 : a=1 b=1

ax1 = plt.subplot(331)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 1, scale = 1)

plt.plot(theta, ptheta)
plt.title('a=1 b=1')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')

# Panel 2 : a=2 b=2

ax1 = plt.subplot(332)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 2, scale = 1/2)

plt.plot(theta, ptheta)
plt.title('a=2 b=2')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')

# Panel 3 : a=4 b=4

ax1 = plt.subplot(333)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 4, scale = 1/4)

plt.plot(theta, ptheta)
plt.title('a=4 b=4')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')


# Panel 4 : a=2 b=1

ax1 = plt.subplot(334)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 2, scale = 1)

plt.plot(theta, ptheta)
plt.title('a=2 b=1')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')

# Panel 5 : a=8 b=4

ax1 = plt.subplot(335)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 8, scale = 1/4)

plt.plot(theta, ptheta)
plt.title('a=8 b=4')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')


# Panel 6 : a=32 b=16

ax1 = plt.subplot(336)

theta = np.arange(0,11,0.01)
ptheta = gamma.pdf(theta, 32, scale = 1/16)

plt.plot(theta, ptheta)
plt.title('a=32 b=16')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')


plt.show()
```


    
![output_21_0](https://user-images.githubusercontent.com/57588650/110474328-e2455d80-8122-11eb-94a7-a6957d9f205b.png)
    


**그림. 3.8.** 감마 확률 밀도 함수

### $\theta$의 사후 분포

사전 분포 : p($\theta$) = $\text{scipy.stats.gamma.pdf}(\theta,a,scale = 1/b)$

데이터 : $Y_1, ..., Y_n|\theta \sim \text{i.i.d.} \ Poisson(\theta)$

라면 사후분포는

$$
\begin{align}
p(\theta|y_1, ..., y_n) = p(\theta) \times p(y_1, ..., y_n | \theta) / p(y_1, ..., y_n) \newline
=(\theta^{a-1} e^{-b\theta}) \times (\theta^{\Sigma y_i} e^{-n\theta}) \times c(y_1,...,y_n,a,b). \newline
= (\theta^{a + \Sigma y_i -1} e^{-(b+n)\theta}) \times c(y_1, ..., y_n, a, b)
\end{align}
$$

이는 명확하게 감마 분포의 형태이고, 다음과 같은 켤레성이 있다는 것을 확인할 수 있습니다.

$$
\begin{cases} \theta \sim gamma(a,b) \newline Y_1, ..., Y_n | \theta \sim Poisson(\theta) \end{cases}  \Rightarrow \theta|Y_1,...,Y_n \sim gamma(a+\sum^n_{i=1} Y_i, b+n).
$$

추정과 예측 과정은 이항 모델과 비슷합니다. 

$\theta$의 사후 기댓값은 사전 기댓값과 표본 평균의 [볼록 조합](https://ko.wikipedia.org/wiki/볼록_조합)입니다.:

$$
\begin{align}
E[\theta|y_1, ..., y_n] = \frac{a + \Sigma y_i}{b+n} \newline
= \frac{b}{b+n} \frac{a}{b} + \frac{n}{b+n} \frac{\Sigma y_i}{n}
\end{align}
$$

* b는 사전 관찰값들의 수입니다
* a는 b개의 사전 관찰값들이 가지고 있는 갯수의 합입니다.(예를 들면, 사전 관찰값에서 $\alpha$라는 사람의 자식이 2명, $\beta$라는 사람의 자식이 3명이라면 b = 2, a = 2 + 3 = 5)

$n$이 크다면, 데이터로 부터의 정보가 사전 정보를 압도합니다

$$
if \ \ \ n >> b \Rightarrow E[\theta|y_1, ..., y_n] \approx \bar{y},
Var[\theta|y_1, ..., y_n] \approx \bar{y}/n
$$

($n \rightarrow \infty$ 라고 생각해보세요)

추가적인 데이터가 어떤 값을 가질지에 대한 예측은 사후 예측 분포를 통해 구할 수 있습니다.

$$
p(\tilde{y}|y_1, ..., y_n) = \int^{\infty}_0 p(\tilde{y}|\theta, y_1, ..., y_n)p(\theta|y_1, ..., y_n) d\theta \ \ \ (\text{by 주변 확률의 법칙})
$$

$$
= \int p(\tilde{y}|\theta)p(\theta|y_1, ..., y_n) d\theta \ (\theta \text{가 주어졌을 때} \ y_1, ..., y_n \ \text{는 조건부 독립})
$$

$$
= \int dpois(\tilde{y}, \theta)dgamma(\theta, a+\Sigma y_i, b+n) d\theta 
$$

$$
= \int \bigg( \frac{1}{\tilde{y}!} \theta^{\tilde{y}} e^{-\theta} \bigg) \bigg( \frac{(b+n)^{a + \Sigma y_i}}{\Gamma(a+\Sigma y_i)} \theta^{a + \Sigma y_i -1} e^{-(b+n)\theta} \bigg) d\theta 
$$

$$
= \frac{(b+n)^{a + \Sigma y_i}}{\Gamma(\tilde{y} + 1) \Gamma(a+\Sigma y_i)} \int^{\infty}_{0} \theta^{a + \Sigma y_i + \tilde{y} - 1} e^{-(b+n+1)\theta} d\theta.
$$

맨 뒤의 적분은 복잡해보이지만, 베타 사후 분포 때와 마찬가지로 감마 pdf를 적분한 값이 1이라는 사실을 사용하면 됩니다.

$$
1 = \int^{\infty}_0 \frac{b^a}{\Gamma(a)}\theta^{a-1} e^{-b\theta} d\theta \ \ \text{for any values a,b}>0.
$$

즉

$$
\int^{\infty}_0 \theta^{a-1} e^{-b\theta} d\theta = \frac{\Gamma(a)}{b^a} \ \ \text{for any values a,b} >0
$$


이고, 여기에 a는 $a + \Sigma y_i + \tilde{y}$를, b에는 $b + n + 1$을 대입하면 사후 예측 분포의 적분 부분을 구할 수 있습니다.


$$
\int^{\infty}_{0} \theta^{a + \Sigma y_i + \tilde{y} -1} e^{-(b+n+1) \theta} d\theta = \frac{\Gamma(a + \Sigma y_i + \tilde{y})}{(b + n + 1)^{a+\Sigma y_i + \tilde{y}}}
$$

결과적으로 사후 예측 분포를 간단하게 표현하면 다음과 같습니다

$$
p(\tilde{y}|y_1, ..., y_n) = \frac{\Gamma(a+\Sigma y_i + \tilde{y})}{\Gamma(\tilde{y}+1)\Gamma(a + \Sigma y_i)} \bigg( \frac{b+n}{b+n+1} \bigg)^{a + \Sigma y_i} \bigg(\frac{1}{b+n+1} \bigg)^{\tilde{y}} \ \ \text{for} \ \tilde{y} \in (0,1,2,...)
$$

이것은 파라미터가 $(a + \Sigma y_i, b+n)$인 음이항분포(negative binomial)이고 따라서 다음과 같은 통계량을 가집니다.

$$
\begin{align}
E[\tilde{Y}|y_1, ..., y_n] = \frac{a+\Sigma y_i}{b+n} = E[\theta|y_1,...,y_n]; \newline
Var[\tilde{Y}|y_1, ..., y_n] = \frac{a + \Sigma y_i}{b+n} \frac{b+n+1}{b+n} = Var[\theta|y_1, ..., y_n] \times (b + n + 1) \newline
=E[\theta|y_1, ..., y_n] \times \frac{b+n+1}{b+n}
\end{align}
$$

예측 분산의 식에 대해 더 깊게 이해해봅시다. 예측 분산이란 모집단에서 뽑은 새로운 샘플 $\tilde{Y}$의 사후 불확실성의 정도를 확장한 것입니다. $\tilde{Y}$의 불확실성은 모집단에 대한 불확실성과 모집단에서 뽑은 표본의 변동성의 한 줄기입니다. 큰 n에 대해서 $\theta$에 대한 불확실성은 작습니다((b+n+1) / (b+n) $\approx$ 1) 그리고 $\tilde(Y)$가 가지고 있는 불확실성의 대부분은 표본의 변동성에서 옵니다. 포아송 모델에서 이것은 $\theta$입니다. 작은 n에 대해서는 $\tilde{Y}$의 불확실성은 $\theta$에 대한 불확실성까지 포함합니다. 그래서 전체 불확실성은 표본의 변동성보다 더 큽니다((b+n+1) / (b+n) > 1).

### 3.2.2 예시 : 출산율

![IMG_4EB359DDE5FD-1](https://user-images.githubusercontent.com/57588650/110417259-979ef380-80d8-11eb-9679-2c0743af7ecd.jpeg)

**그림. 3.9** 학사 학위 미만, 이상 두 그룹의 자녀의 수

이 예시에서는 학사 학위 미만의 학력을 가진 여성과, 이상을 가진 여성의 출산율 차이를 비교해볼 것입니다. 주어진 데이터는 다음과 같습니다.

* $Y_{1,1} , ... , Y_{n_1,1}$ : 학사 학위가 없는 여성의 자녀의 수
* $Y_{1,2} , ... , Y_{n_2,2}$ : 학사 학위가 있는 여성의 자녀의 수

그리고 이 데이터는 다음과 같은 표본 모델을 따른다고 하겠습니다

$$
Y_{1,1}, ... , Y_{n_1, 1} | \theta_1 \ \sim \ \text{i.i.d.} \ Poisson(\theta_1)
$$

$$
Y_{1,2}, ... , Y_{n_2, 2} | \theta_1 \ \sim \ \text{i.i.d.} \ Poisson(\theta_2)
$$

이 데이터에 포아송 모델을 쓰는 것이 적절한지는 다음 챕터에서 보도록 하겠습니다.

데이터의 경험적인 분포는 **그림. 3.9**에 나와있고, 그룹합과 평균은 다음과 같습니다.

학사 학위 미만 : $n_1 = 111, \sum_{i=1}^{n_1} Y_{i,1} = 217, \bar{Y}_1 = 1.95$

학사 학위 이상 : $n_2 = 44, \sum_{i=1}^{n_2} Y_{i,2} = 66, \bar{Y}_2 = 1.50$

{$\theta_1, \theta_2$} $\sim$ i.i.d. gamma(a=2, b=1)일 때, 다음과 같은 사후 분포를 가집니다.

$\theta_1|$ {$n_1 = 111, \Sigma Y_{i,1} = 217$} $\sim$ gamma(2+217, 1+111) = gamma(219, 112)


$\theta_2|$ {$n_2 = 44, \Sigma Y_{i,1} = 66$} $\sim$ gamma(2+66, 1+44) = gamma(68, 45)



 $\theta_1, \theta_2$의 사후 평균, 최빈값, 그리고 95% 분위수 기반 신뢰구간은 이 감마 사후 분포들로 구할 수 있습니다.


```python
from scipy.stats import gamma

## 사전 분포의 파라미터
a = 2
b = 1

## 데이터

### 그룹 1
n_1 = 111
sum_y1 = 217

### 그룹 2
n_2 = 44
sum_y2 = 66

## 그룹 1의 사후 평균, 최빈값, 95% 신뢰구간

### 평균
print("그룹 1의 사후 평균 = ", (a + sum_y1) / (b + n_1))

### 최빈값
print("그룹 1의 사후 최빈값 = ", (a + sum_y1 -1) / (b + n_1))

### 95% 신뢰구간
l, u = gamma.ppf([0.025, 0.975], a + sum_y1, scale = 1 / (b + n_1))
          
print("그룹 1의 95% 신뢰구간 = ", l, u)

print("-------------------------------------------------------------")

## 그룹 2의 사후 평균, 최빈값, 95% 신뢰구간

### 평균
print("그룹 2의 사후 평균 = ", (a + sum_y2) / (b + n_2))

### 최빈값
print("그룹 2의 사후 최빈값 = ", (a + sum_y2 -1) / (b + n_2))

### 95% 신뢰구간
l, u = gamma.ppf([0.025, 0.975], a + sum_y2, scale = 1 / (b + n_2))
          
print("그룹 2의 95% 신뢰구간 = ", l, u)


```

    그룹 1의 사후 평균 =  1.9553571428571428
    그룹 1의 사후 최빈값 =  1.9464285714285714
    그룹 1의 95% 신뢰구간 =  1.7049431489418194 2.2226790202451725
    -------------------------------------------------------------
    그룹 2의 사후 평균 =  1.511111111111111
    그룹 2의 사후 최빈값 =  1.488888888888889
    그룹 2의 95% 신뢰구간 =  1.1734369056138325 1.8908362583954745



```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import gamma, nbinom

## 사전 분포의 파라미터
a = 2
b = 1

## 데이터

### 그룹 1
n_1 = 111
sum_y1 = 217

### 그룹 2
n_2 = 44
sum_y2 = 66

# Panel 1 : 평균 출산율의 사후 분포

ax1 = plt.subplot(121)

## 그룹 1
theta = np.arange(0,5,0.01)
ptheta = gamma.pdf(theta, sum_y1 + a , scale = 1/(n_1 + b))

plt.plot(theta, ptheta, label = 'Group 1 : Less than bachelor\'s')


## 그룹 2

ptheta = gamma.pdf(theta, sum_y2 + a, scale = 1/(n_2 + b))

plt.plot(theta_1, ptheta, label = 'Group 2 : Bachelor\'s or higher')

## 사전 분포
ptheta = gamma.pdf(theta, a,b)
plt.plot(theta, ptheta ,linestyle = '--', label = 'Posterior : gamma(2,1)')

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|y_1, ..., y_n)$')
plt.legend(loc = 'upper right')


# Panel 2 : 사후 예측 분포

ax2 = plt.subplot(122)


## 그룹 1

newy = np.arange(0,13,1)
pnewy = nbinom.pmf(newy, a + sum_y1, (b+n_1)/(b+n_1+1))

plt.bar(newy-0.1, pnewy, width = 0.2, label = 'Less than bachelor\'s')
        

## 그룹 2
        
pnewy = nbinom.pmf(newy, a + sum_y2, (b+n_2)/(b+n_2+1))
        
plt.bar(newy+0.1, pnewy, width = 0.2, label = 'Bachelor\'s or higer')


plt.ylim(-0.01,0.35)
plt.xlabel(r'$y_{n+1}$')
plt.ylabel(r'$p(y_{n+1}|y_1, ..., y_n)$')
plt.legend(loc = 'upper right')


plt.show()
```


    
![output_42_0](https://user-images.githubusercontent.com/57588650/110474335-e40f2100-8122-11eb-9474-d5723190e9cf.png)
    


**그림. 3.10.** 왼쪽 그래프는 평균 출산율의 사후 분포를 나타내고(점선으로 나타난 부분은 공통 사전 분포), 오른쪽은 자식의 수의 사후 예측 분포를 나타냅니다.

이 그래프를 보면 알 수 있듯이 사후 분포는 $\theta_1 > \theta_2$라는 명확한 증거를 보여줍니다. 예를 들어 $Pr(\theta_1 > \theta_2 | \Sigma Y_{i,1} = 217, \Sigma Y_{i,2} = 66) = 0.97$로 $\theta_1$이 $\theta_2$보다 클 확률이 97%로 매우 높습니다. 이제 각각의 그룹별 모집단 두 명의 무작위로 추출된 샘플을 생각해봅시다. 과연 학위가 없는 그룹에서 뽑은 사람이 있는 그룹에서 뽑은 사람보다 자식의 수가 많을까요? 우리는 이 확률을 정확하게 구할 수 있습니다. $\tilde{Y_1}, \tilde{Y_2}$의 사후 예측 분포는 모두 음이항분포를 따르며 위 그림의 오른쪽 그래프와 같이 나타낼 수 있습니다.

그런데 왼쪽과 오른쪽 그래프의 차이가 보이시나요? 바로 오른쪽의 그래프가 왼쪽의 그래프보다 겹치는 부분이 더 많습니다. 예를 들어 $Pr(\tilde{Y_1} > \tilde{Y_2} | \Sigma Y_{i,1} = 217, \Sigma Y_{i,2} = 66) = 0.48$이고 $Pr(\tilde{Y_1} = \tilde{Y_2} | \Sigma Y_{i,1} = 217, \Sigma Y_{i,2} = 66) = 0.22$입니다. 왼쪽의 그래프에서 나타나는 방식의 {$\theta_1 > \theta_2$}와 오른쪽의 그래프에서 나타나는 방식인 {$\tilde{Y_1} > \tilde{Y_2}$} 간의 차이는 매우 중요합니다. 이것은 두 모집단이 그렇게 큰 차이를 보이진 않는다는 강한 증거가 됩니다. 

## 3.3 지수분포족(Exponential families) 과 켤레 사전 분포(Conjugate prior)

지금까지 배운 이항 모델과 포아송 모델은 모두 파라미터가 하나인 지수분포족의 예시입니다. 파라미터가 하나인 지수분포족은 다음과 같은 형태의 함수입니다. 

$$
p(y|\phi) = h(y)c(\phi)e^{\phi t(y)}, \ \ \text{이 때} \ \ \phi \text{는 미지수 이고, } \ t(y)\text{는 충분 통계량(sufficient statistic)입니다.}
$$

Diaconis와 Ylvisaker (1979)는 지수분포족에 모두 먹히는, 특히 사전 분포가 $p(\phi|n_0, t_0) = \kappa(n_0, t_0) c(\phi)^{n_0} e^{n_0 t_0 \phi}$ 형태에 일반적으로 적용되는 켤레 사전 분포에 대해 연구했습니다. 이러한 사전 분포와 $Y_1, ..., Y_n \sim \text{i.i.d.} \ p(y|\phi)$으로 부터 뽑힌 정보를 결합하면 다음과 같은 사후 분포를 얻을 수 있게 됩니다.

$$
p(\phi|y_1, ..., y_n) \propto p(\phi) p(y_1, ..., y_n | \phi) 
$$

![output](https://user-images.githubusercontent.com/57588650/110475813-aa3f1a00-8124-11eb-9b0a-e6261449f803.png)

이 때, $\kappa(n_0, t_0), h(y)$가 상수항이므로

$$
\propto \underbrace{\prod_{i=1}^n c(\phi) e^{\phi t(y_i)}}_{c(\phi)^n e^{\phi \sum_{i=1}^n t(y_i)}} \times c(\phi)^{n_0} e^{n_0 t_0 \phi}
$$

$$
= c(\phi)^{n_0+n} exp \bigg( \phi \times \bigg[ n_ot_0 + \sum_{i=1}^n t(y_i) \bigg] \bigg)
$$


이 식을 $p(\phi|n_0, t_0) = \kappa(n_0, t_0) c(\phi)^{n_0} e^{n_0 t_0 \phi}$ 이 식의 우항이라고 생각하고 적용하면 결과적으로 다음과 같이 정리할 수 있습니다.

$$
\propto p(\phi|n_0 + n, n_0 t_0 + n\bar{t}(\textbf{y})), \ \text{이 때} \ \bar{t}(y) = \Sigma t(y_i) / n 
$$

사후 분포와 사전 분포의 비슷한 부분에서 볼 수 있듯, $n_0$을 "사전 표본 크기"로, $t_0$은 $t(Y)$에 대한 "사전 추정값"으로 해석될 수 있습니다.

또한 이와같은 해석은 다음과 같이 더 정확하게 표현될 수 있습니다.

$$
E[t(y)] = E[E[t(Y)|\phi]] \ \ \ (\text{수통 시간에 배웠죠?})
$$

$$ 
= E[-c'(\phi) / c(\phi)] = t_0
$$

이때문에 $t_0$은 $t(Y)$의 사전 기댓값을 나타냅니다. 

파라미터 $n_0$는 그 사전 분포가 얼마나 많은 정보를 포함하고 있는지 포여주는 측도입니다. 

이 측도를 계량화하는 방법은 많지만, 가장 쉬운 것은 $\phi$에 대한 함수로서 $p(\phi|n_0, t_0)$이 $n_0$개의 "사전 관찰값" $\tilde{y_1}, ..., \tilde{y}_{n_0}$으로부터 온 우도(likelihood)인 $p(\tilde{y}_1, ..., \tilde{y}_{n_0}|\phi)$과 같은 모양을 가진다는 것을 사용하는 것입니다. 


이 때 $\Sigma t(\tilde{y}_i)/n_0 = t_0$입니다. 이것을 사용해서, 사전 분포 $p(\phi|n_0, t_0)$이 모집단으로 부터 독립적으로 추출한 $n_0$개의 샘플과 같은 양의 정보를 가지고 있다는 것을 알 수 있게됩니다. 

**예시 : 이항 모델**

이항 모델은 다음과 같은 방법으로 지수분포꼴의 모양으로 변환시킬 수 있습니다.


\begin{align}
p(y|\theta) = \theta^y(1-\theta)^{1-y} \newline
= \bigg(\frac{\theta}{1-\theta} \bigg)^y (1 - \theta) \newline
= e^{\phi y}(1+e^{\phi})^{-1}
\end{align}

이 때 $\phi = log[\theta / (1-\theta)]$이고, 이는 여러분이 알듯 log-odd입니다.

$\phi$에 대한 켤레 사전분포는 따라서 $p(\phi|n_0, t_0) \propto (1 + e^{\phi})^{-n_0} e^{n_0 t_0 \phi}$입니다. 이 때 $t_0$은 $t(y) = y$의 사전 기댓값 또는 $Y=1$이라는 우리의 사전 확률을 표현합니다. $\theta$에 대한 사전 분포인 $p(\theta|n_0, t_0) \propto \theta^{n_0t_0 - 1}(1-\theta)^{n_0(1-t_0)-1}$의 변수를 이리저리 변형해보면 $beta(n_0t_0, n_0(1-t_0))$라는 사전 분포를 얻을 수 있습니다. 약한 사전 분포(weakly informative prior distribution)는 $t_0$은 우리의 사전 기댓값으로, $n_0=1$로 설정함으로써 얻을 수 있습니다. 만약 우리의 사전 기댓값이 1/2라면, 결과로 나오는 사전 분포는 beta(1/2, 1/2)입니다. 그리고 이것은 이항 표본추출 모델의 Jefferys' prior distribution([여기서 알아봅시다](https://rooney-song.tistory.com/18))과 같습니다. 약한 $beta(t_0, (1-t_0))$ 사전 분포 하에서, 사후 분포는 {$\theta|y_1, ..., y_n$} $\sim beta(t_0 + \Sigma y_i, (1-t_0) + \Sigma(1-y_i))$입니다.

**예시 : 포아송 모델**

Poisson($\theta$) 모델은 다음을 활용해 지수분포족 모델로 보일 수 있습니다.

* t(y) = y;
* $\phi = log\theta$
* $c(\phi) = exp(e^{-\phi})$

$\phi$의 켤레 사전 분포는 따라서 $p(\phi|n_0, t_0) = exp(n_0 e^{-\phi}) e^{n_0 t_0 y}$ 이고, 이 때 $t_0$은 $Y$의 모평균의 사전 기댓값입니다. 이것을 $\theta$에 대한 사전 밀도 함수로 나타내면 $p(\theta|n_0, t_0) \propto \theta^{n_0 t_0 -1}e^{-n_0\theta}$이고, 이것은 $gamma(n_0t_0, n_0)$의 확률 밀도 함수입니다. 

약한 사전 분포는 $t_0$을 Y의 사전 기댓값으로 놓고 $n_0 = 1$로 놓음으로서 얻을 수 있고, 이는 $gamma(t_0, 1)$ 사전 분포를 제공합니다. 이러한 사전 분포 하에서의 사후 분포는 {$\theta|y_1, ..., y_n$} $\sim \ gamma(t_0 + \Sigma y_i, 1 + n)$입니다.




