---
title: A First Course In Bayesian Statistical Methods Chapter 6. Multivariate Normal Model, 1. Calculating Posterior Distribution Under Semiconjugate Prior
date: 2021-03-25T18:22:08+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 6. 다변량 정규 모델(Multivariate normal model)

이 챕터에서는 다변량 데이터를 다루는데 가장 효과적인 모델인 다변량 정규 모델에 대해 배워볼 것입니다. 다변량 정규 모델은 변수 모음의 모평균, 분산, 그리고 상관관계를 결합적으로 추정합니다. 이번 챕터의 과정은 다음과 같습니다.

1. semiconjugate 사전 분포 하에서 사후 분포 계산
2. 다변량 정규 모델을 사용한 무작위 결측치 imputation

## 6.1 다변량 정규 pdf

### 예시 : 독해 시험

22명의 아이들에게 특정한 교육 방식을 적용한 이전과 이후에 독해 시험 성적이 어떻게 변화하는지를 알아보고 싶습니다.

이 때 주어진 데이터는 다음과 같습니다.

$$
\textbf{Y} = \bigg({Y_{i,1} \atop Y_{i,2}} \bigg) = \bigg({\text{score on first test} \atop \text{score on second test}} \bigg)
$$

그리고 우리가 관심있는 것은 모평균 $\theta$와 모 공분산 행렬 $\Sigma$ 입니다.

$$
E[\textbf{Y}] = \bigg({E[Y_{i,1}] \atop E[Y_{i,2}]} \bigg) = \bigg({\theta_1 \atop \theta_2} \bigg)
$$

$$
\Sigma = Cov[\textbf{Y}] = \bigg( {E[Y_1^2] - E[Y_1]^2 \ \ \ \ \ E[Y_1Y_2] - E[Y_1]E[Y_2] 
\atop E[Y_1Y_2] - E[Y_1]E[Y_2] \ \ \ \ \ E[Y_2^2] - E[Y_2]^2} \bigg) = \bigg({\sigma_1^2 \ \ \sigma_{1,2} \atop \sigma_{1,2} \ \ \sigma_2^2} \bigg)
$$

만약 여기서 $\theta$와 $\Sigma$를 안다면, $\theta_2 - \theta_1$를 구해 교육 방식의 효과성을 평가하거나, 상관계수 $\rho_{1,2} = \rho_{1,2} / \sqrt{\sigma_1^2\sigma_2^2}$를 통해 독해 시험의 일관성을 추정할 수 있습니다.

### 다변량 정규 pdf

$\theta$와 $\Sigma$는 모두 모집단의 적률(moment)을 나타내는 함수입니다. 특히 이 둘은 1차 적률과 2차 적률 함수입니다.(적률은 수리통계학 시간에 배우는 "적률 생성 함수(moment generating funtion)"의 그 적률입니다!) 

1차 적률 : $E[Y_1], E[Y_2]$

2차 적률 : $E[Y_1^2], E[Y_1Y_2], E[Y_2^2]$

여러분이 알듯, 모수가 $(\theta, \sigma^2)$인 단변량 정규 모델에서 $E[Y] = \theta, E[Y^2] = Var[Y] + E[Y]^2 = \sigma^2 + \theta^2$입니다.

다변량 정규 데이터에서 1차, 2차 적률을 표현하는 이와 비슷한 모델은 "다변량 정규 모델"입니다. 우리는 p차원 벡터 $\textbf{Y}$가 다음과 같은 pdf를 가질 때 다변량 정규 분포를 가진다고 합니다.

$$
p(\mathbf{y}|\boldsymbol{\theta}, \Sigma) = (2\pi)^{-p/2}|\Sigma|^{-1/2} \text{exp}\{-(\mathbf{y}-\boldsymbol{\theta})^T\Sigma^{-1}(\mathbf{y}-\boldsymbol{\theta})/2\}
$$

이 때,

<img src = "https://user-images.githubusercontent.com/57588650/111948888-240ed480-8b23-11eb-9d13-0b1e3c9355bc.jpeg" width="400px">

입니다.


```python
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,4))


x, y = np.mgrid[20:90:.1, 20:90:1]
pos = np.dstack((x, y))


# sigma_{1,2} = -48
ax1 = plt.subplot(131)

rv = multivariate_normal([50,50], [[64, -48],[-48, 144]])

ax1.contourf(x, y, rv.pdf(pos))
plt.xlim([20, 80])
plt.ylim([20, 80])
plt.xlabel(r'$y_1$')
plt.ylabel(r'$y_2$')

# sigma_{1,2} = 0
ax2 = plt.subplot(132)

rv = multivariate_normal([50,50], [[64, 0],[0, 144]])

ax2.contourf(x, y, rv.pdf(pos))
plt.xlim([20, 80])
plt.ylim([20, 80])
plt.xlabel(r'$y_1$')
plt.ylabel(r'$y_2$')

# sigma_{1,2} = 48
ax3 = plt.subplot(133)

rv = multivariate_normal([50,50], [[64, 48],[48, 144]])

ax3.contourf(x, y, rv.pdf(pos))
plt.xlim([20, 80])
plt.ylim([20, 80])
plt.xlabel(r'$y_1$')
plt.ylabel(r'$y_2$')


plt.show();
```


    
    
![output_16_0](https://user-images.githubusercontent.com/57588650/112450243-87a13800-8d97-11eb-8930-57c2be5ce897.png)
    


**그림. 6.1** 다변량 정규 분포

**그림 6.1**은 각각 $\boldsymbol{\theta} = (50,50)^T$, $\sigma_1^2 = 64, \sigma_2^2 = 144$로 동일하고 $\sigma_{1,2}$만 왼쪽은 -45, 가운데는 0, 오른쪽은 45로 다른(각각 -0.5, 0, 0.5의 correlation을 줍니다) 2차원 다변량 정규 분포입니다. 다변량 정규 분포의 흥미로운 특징은 변수 $Y_j$ 각각의 주변 분포(marginal distribution)가 평균이 $\theta_j$이고 분산이 $\sigma_j^2$인 단변량 정규분포를 따른다는 점입니다. 

이것의 의미하는 점은 세 개의 모집단에서 $Y_1, Y_2$의 주변 분포는 모두 같다는 것입니다. 다른 것은 오직 공분산 파라미터 $\sigma_{1,2}$에 의해 통제받는 $Y_1, Y_2$사이의 관계입니다.

## 6.2 평균에 대한 semiconjugate 사전 분포

> Semiconjugate 사전분포란?

>파라미터 ($\theta, \sigma^2$)를 서로 독립이라고 가정했을 때 $\theta, \sigma^2$에 대한 사전 분포를 semiconjugate 사전분포라고 합니다. 이 때 각각은 다음과 같은 semiconjugate 사전 분포를 가집니다.
    
$$
\theta \sim \text{Normal}(\mu_0, \tau_0^2) \\
1/\sigma^2 \sim \text{Gamma}(\nu_0/2, \nu_0\sigma_0^2)
$$
    

i.i.d 샘플 $Y_1, ..., Y_n$이 단변량 정규 모집단에서 뽑혔다면, 모평균에 대한 편리한 켤레 사전 분포 또한 단변량 정규분포를 따릅니다. 이와 비슷하게 다변량 평균 $\boldsymbol{\theta}$의 편리한 사전 분포는 다변량 정규 분포를 따르고 다음과 같이 parameterize 할 수 있습니다.

$$
p(\boldsymbol{\theta}) = \text{MVN}(\mu_0, \Lambda_0)
$$

이 때 $\mu_0, \Lambda_0$은 각각 $\boldsymbol{\theta}$의 사전 평균과 분산입니다. 그렇다면 $\boldsymbol{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma$의 full conditional distribution은 뭘까요? 단변량 케이스에서 정규분포인 사전 분포, 표본 분포를 가진다면 모평균의 full conditional distribution 또한 결과적으로 정규분포가 됩니다. 이 결과가 다변량 케이스에서도 적용되는지 확인해봅시다. 우선 $\boldsymbol{\theta}$에 대한 함수로 사전 분포를 만들어봅시다.

> Full Conditional Distribution이란 자신을 제외한 모든 것이 given일 때의 조건부 분포입니다.

\begin{align}
p(\boldsymbol{\theta}) = (2\pi)^{-p/2} |\Lambda_0|^{-1/2}\text{exp} \{-\frac{1}{2}(\boldsymbol{\theta - \mu_0})^T\Lambda_0^{-1}(\boldsymbol{\theta - \mu_0}) \} \newline
    = (2\pi)^{-p/2}|\Lambda_0|^{-1/2} \text{exp}\{-\frac{1}{2}\boldsymbol{\theta^T\Lambda_0^{-1}\theta} + \boldsymbol{\theta^T \Lambda_0^{-1}\mu_0 -\frac{1}{2} \mu^T_0 \Lambda_0^{-1} \mu_0}\} \newline
\propto \text{exp} \{-\frac{1}{2} \boldsymbol{\theta^T} \Lambda_0^{-1} \boldsymbol{\theta}  + \boldsymbol{\theta^T} \Lambda_0^{-1} \boldsymbol{\mu_0}\} \newline
= \text{exp} \{-\frac{1}{2} \boldsymbol{\theta^T} A_0 \boldsymbol{\theta} + \boldsymbol{\theta}^Tb_0\}  \ \ \ \ \ \ \ (6.1)
\end{align}

이 때 $A_0 = \Lambda_0^{-1}, b_0 = \Lambda_0^{-1}\mu_0$입니다. 

이걸 거꾸로 말하면, 만약 확률 벡터(random vector)$\boldsymbol{\theta}$가 $\mathbb{R}^p$($p$차원 실수공간)에서 행렬 $\boldsymbol{A}$와 벡터 $\boldsymbol{b}$에 대해 $\text{exp} \{ -\boldsymbol{\theta^TA\theta}/2 + \boldsymbol{\theta^Tb}\}$에 비례하는 밀도함수를 가진다면, $\boldsymbol{\theta}$는 반드시 공분산이 $\boldsymbol{A}^{-1}$이고 평균이 $\boldsymbol{A^{-1}b}$인 다변량 정규 분포를 가져야 합니다.

만약 우리의 표본 모델이

$$
\boldsymbol{Y_1, ..., Y_n|\theta},\Sigma \ \overset{i.i.d}{\sim} \ MVN(\boldsymbol{\theta}, \Sigma)
$$

라면, 위와 비슷한 계산을 통해 관찰된 벡터인 $\mathbf{y_1}, ..., \mathbf{y_n}$의 결합 표본 분포를 구할 수 있습니다.

\begin{align}
p(\mathbf{y_1}, ..., \mathbf{y_n}|\boldsymbol{\theta}, \Sigma) = \prod^n_{i=1}(2\pi)^{-p/2}|\Sigma|^{-1/2} \text{exp} \{ -(\mathbf{y_i} - \boldsymbol{\theta})^T \Sigma^{-1} (\mathbf{y_i} - \boldsymbol{\theta})/2 \} \newline
= (2\pi)^{-np/2} |\Sigma|^{-n/2}\text{exp} \{ -\frac{1}{2} \sum_{i=1}^n (\mathbf{y_i} - \boldsymbol{\theta})^T \Sigma^{-1} (\mathbf{y_i} - \boldsymbol{\theta})\} \newline
\propto \text{exp} \{-\frac{1}{2} \boldsymbol{\theta^T A_1 \theta + \theta^T b_1} \} \ \ \ \ (6.2)
\end{align}

이 때 $\boldsymbol{A_1} = n\Sigma^{-1}, \mathbf{b_1} = n\Sigma^{-1}\boldsymbol{\bar{y}}$이고, $\boldsymbol{\bar{y}}$는 각 변수의 평균을 나타내는 벡터 $\boldsymbol{\bar{y}} = (\frac{1}{n}\Sigma_{i=1}^n y_{i,1}, ..., \frac{1}{n} \Sigma_{i=1}^n y_{i,p})^T$입니다. (6.1)식과 (6.2)식을 곱하면 $\boldsymbol{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma$의 full conditional distribution을 구할 수 있습니다.

$$
p(\boldsymbol{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma) \propto \text{exp} \{ -\frac{1}{2}\boldsymbol{\theta^T A_0 \theta + \theta^T b_0} \} \times \text{exp} \{ -\frac{1}{2}\boldsymbol{\theta^T A_1 \theta + \theta^T b_1} \} \\ 
= \text{exp} \{ -\frac{1}{2} \boldsymbol{\theta^T A_n \theta + \theta^T b_n}\}  \ \ \ \ \  (6.3)
$$

이 때 $\boldsymbol{A_n} = \boldsymbol{A_0} + \boldsymbol{A_1} = \Lambda_0^{-1} + n\Sigma^{-1} $이고 $\boldsymbol{b_n} = \boldsymbol{b_0} + \boldsymbol{b_1} = \Lambda_0^{-1} \boldsymbol{\mu_0} + n\Sigma^{-1}\mathbf{\bar{y}}$ 입니다.

즉 식 (7.3)은 $\boldsymbol{\theta}$의 조건부 분포가 반드시 공분산이 $\mathbf{A_n^{-1}}$이고 평균이 $\mathbf{A_n}\boldsymbol{b_n}$인 다변량 정규분포를 따른다는 것을 의미합니다. 즉 다음과 같은 통계량을 가집니다.

\begin{align}
\text{Cov}[\mathbf{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma] = \Lambda_n = (\Lambda_0^{-1} + n\Sigma^{-1})^{-1} \ \ \ \ (6.4) \newline
E[\mathbf{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma] = \boldsymbol{\mu_n} = (\Lambda_0^{-1} \boldsymbol{\mu_0} + n\Sigma^{-1}\mathbf{\bar{y}}) \ \ \ \ (6.5) \newline
p(\mathbf{\theta}|\mathbf{y_1}, ..., \mathbf{y_n}, \Sigma) = MVN(\mathbf{\mu_n}, \Lambda_n) \ \ \ \ (6.6)
\end{align}

약간 복잡해보이지만, 단변량 정규 분포 케이스일 때와 비교하면 더 쉽게 이해할 수 있습니다.

식 (6.4) : 단변량 때와 마찬가지로 사후 정확도(precision) 또는 분산의 역수가 사전 정확도와 데이터 정확도의 합으로 나타난다는 것을 보여줍니다. 

식 (6.5) : 사후 기댓값은 사전 기댓값과 표본 평균의 가중평균입니다.

주목해야할 점은, 표본 평균이 모평균에 대해 consistent하기 때문에 사후 평균 또한 실제 분포가 다변량 정규 분포가 아니라도 모평균에 consistent하다는 것입니다.

## 6.3 역 위샤트 분포(inverse-Wishart distribution)

분산 $\sigma^2$이 양수인 것 처럼, 분산-공분산 행렬인 $\Sigma$또한 반드시 "양의 정의(positive definite)"여야 합니다. 양의 정의란 다음을 의미합니다.

$$
\mathbf{x}'\Sigma\mathbf{x} > 0 \ \ \ \text{for all vectors} \ \ \mathbf{x}
$$

양의 정의를 만족하면, 모든 j에서 $\sigma_j^2$이 양수이고 모든 상관관계(correlation)가 -1과 1 사이에 있다는 것을 보장합니다. 공분산 행렬에 필요한 또 하나의 조건은 이것이 대칭행렬이어야 한다는 것입니다. 즉 $\sigma_{j,k} = \sigma_{k,j}$여야 합니다. 즉 어떠한 $\Sigma$에 적당한 사전 분포도 이 복잡한 대칭이고, 양의 정의를 만족하는 행렬에 모든 확률 밀도를 줘야 합니다. 어떻게 이러한 사전 분포를 만들 수 있을까요?

### 경험적 공분산 행렬(Empirical covariance matrices)

다변량 벡터 $\boldsymbol{z_1}, ..., \boldsymbol{z_n}$의 제곱합 행렬은 다음과 같습니다.

이 때 $\mathbf{Z}$는 i번째 행이 $z_i^T$인 $n\times p$행렬입니다. 행렬의 연산을 사용하면 $z_i$는 $p \times 1$행렬이고, $z_i z_i^T$는 다음과 같은 $p \times p$행렬입니다.

<img src = "https://user-images.githubusercontent.com/57588650/112110937-e67b7b80-8bf6-11eb-81d2-c408a282d7db.jpeg" width="400px">



만약 $z_i$들이 평균이 0인 모집단에서 뽑힌 표본들이라면, 행렬 $z_i z_i^T/n$을 벡터 $z_i$가 모든 관찰값의 공분산 행렬을 예측하는데 기여하는 것이라고 생각할 수 있습니다($Cov(Z) = \sum(z_i-\bar{z})^2 = \sum(z_i)^2 = z_i z_i^T$ if $\bar{z}$ = 0). 

그리고 이러한 평균이 0인 케이스에서는 $Z^T Z/n$가 모 공분산 행렬의 불편 추정량(unbiased estimator)인 표본 공분산 행렬과 같습니다.

$$
\frac{1}{n} [Z^T Z]_ {j,j}  = \frac{1}{n}\Sigma_{i=1}^{n} z^2_{i,j} = s_{j,j} = s^2_j
$$

$$
\frac{1}{n}[Z^T Z]_ {j,k} = \frac{1}{n}\Sigma^n_{i=1} z_{i,j} z_{i,k} = s_{j,k}
$$

만약 n>p이고, $\boldsymbol{z_i}$들이 선형 독립이라면, $\mathbf{Z^T Z}$는 양의 정의를 만족하고 대칭행렬이 됩니다. 이러한 가정은 다음과 같은 방식으로 "무작위" 공분산 행렬(random covariance matrix)을 만들 수 있게 합니다. 양의 정수 $\nu_0$과 $p \times p$ 공분산 행렬 $\Phi_0$이 주어졌을 때,

1. $\boldsymbol{z_1}, ..., \boldsymbol{z_n}$ $\overset{i.i.d}{\sim} MVN(0, \Phi_0)$에서 표본을 뽑습니다.

2. $\mathbf{Z^T Z} = \Sigma_{i=1}^{\nu_0} z_i z_i^T$ 를 계산합니다.

이 두 단계를 S번 반복하면 $\mathbf{Z_1^T Z_1}, ..., \mathbf{Z_S^T Z_S}$의 행렬들을 만들 수 있습니다. 이러한 제곱합 행렬들의 분포를 바로 파라미터가 ($\nu_0, \Phi_0$)인 "위샤트 분포(Wishart Distribution)"라고 하고 다음과 같은 특성을 가집니다.

* 만약 $\nu_0 > p$라면, $\boldsymbol{Z^T Z}$은 1의 확률로 양의 정의를 만족합니다.
* $\boldsymbol{Z^T Z}$는 1의 확률로 대칭행렬입니다.
* $E[Z^T Z] = \nu_0 \Phi_0$

위샤트 분포는 감마 분포의 다변량 버전이라고 생각하시면 됩니다(즉 만약 $z$가 평균이 0인 단변량 정규 확률 변수라면, $z^2$은 감마 확률 변수가 됩니다). 단변량 정규 모델에서 정확도 $1 / \sigma^2$에 대한 사전 분포는 감마 분포였습니다. 그리고 분산에 대한 full conditional distribution은 inverse-gamma 분포였습니다. 비슷한 방식으로 정확도 행렬 $\Sigma^{-1}$의 sem-conjugate 사전 분포는 위샤트 분포이고 또한 역 위샤트 분포는 공분산 행렬 $\Sigma$에 대한 semi-conjugate 사전 분포입니다. 약간의 재 파라미터화(reparameterization)을 거쳐, 역 위샤트 분포에서 공분산 행렬 $\Sigma$의 표본을 뽑기 위해서는 다음과 같은 단계를 밟으면 됩니다.

1. $\boldsymbol{z_1}, ..., \boldsymbol{z_{\nu_0}}$ $\overset{i.i.d}{\sim} MVN(0, S_0^{-1})$에서 표본을 뽑습니다.

2. $\boldsymbol{Z^T Z} = \Sigma_{i=1}^{\nu_0} z_i z_i^T$를 계산합니다

3. $\Sigma = (Z^T Z)^{-1}$ 로 설정합니다.

이 시뮬레이션 과정 하에서 정확도 행렬 $\Sigma^{-1}$와 공분산 행렬 $\Sigma$는 다음과 같은 분포를 가집니다.

$$
\Sigma^{-1} \sim \text{Wishart}(\nu_0, \boldsymbol{S_0^{-1}}) \\
\Sigma \sim \text{inverse-Wishart}(\nu_0, \boldsymbol{S_0^{-1}})
$$

그리고 기댓값은 다음과 같습니다.

$$
E[\Sigma^{-1}] = \nu_0 \boldsymbol{S_0^{-1}} \\
E[\Sigma] = \frac{1}{\nu_0 - p - 1}(\boldsymbol{S_0^{-1}})^{-1} = \frac{1}{\nu_0 - p - 1} \boldsymbol{S_0}
$$

만약 참 공분산 행렬이 $\Sigma_0$ 근처에 있다고 확신한다면, $\nu_0$은 크게 그리고 $\mathbf{S_0} = (\nu_0 - p - 1)\Sigma_0$ 놓으면 됩니다. 그렇게 하면 $\Sigma$의 분포는 $\Sigma_0$ 근처로 집중하게 됩니다. 다른 방식으로, $\nu_0 = p + 2$를 $\mathbf{S_0} = \Sigma_0$으로 놓는다면 $\Sigma$가 $\Sigma_0$를 중심으로 느슨하게 퍼져있을 것입니다.

### 공분산 행렬의 full conditional distributioin

$p(\Sigma) \sim \text{inverse-Wishart}(\nu_0, \mathbf{S_0^{-1}})$의 밀도는 다음과 같습니다.

$$
p(\Sigma) = \bigg[2^{\nu_0 p/2} \pi ^{({p \atop 2})/2} |\mathbf{S_0}|^{-\nu_0/2} \prod_{i=1}^p \Gamma([\nu_0 + 1 - j]/2) \bigg]^{-1} \times \\
|\Sigma|^{-(\nu_0 + p + 1)/2} \times \text{exp} \{ -\text{tr}(\mathbf{S_0}\Sigma^{-1})/2 \} \ \ \ \ \ (6.7)
$$

정규화 상수(normalizing constant)는 꽤 어렵습니다. 하지만 운좋게도 여기에서는 방정식의 두 번째 줄만 계산하면 됩니다. 이 때 "tr"은 trace를 의미하고 $p \times p$ 행렬 $\mathbf{A}$에서 tr($\mathbf{A}) = \sum_{j=1}^p a_{j,j}$, 즉 대각 성분의 합입니다.

### 표본 분포(sampling distribution)

데이터 $\mathbf{Y_1}, ..., \mathbf{Y_n}$의 표본 분포는 다음과 같습니다

$$
p(\mathbf{y_1}, ..., \mathbf{y_n}|\boldsymbol{\theta}, \Sigma) = (2\pi)^{-np/2}|\Sigma|^{-n/2} \text{exp} \{ -\sum^n_{i=1} ( \mathbf{y_i} - \boldsymbol{\theta} )^T \Sigma^{-1} ( \mathbf{y_i} - \boldsymbol{\theta} ) /2 \} \ \ \ \ (6.8)
$$

그런데 다음과 같은 흥미로운 결과가 있습니다.

$$
\sum_{k=1}^K \mathbf{b_k}^T \mathbf{A} \mathbf{b_k} = \text{tr}(\mathbf{B}^T\mathbf{B} \mathbf{A})
$$

이 때 $\mathbf{B}$는 $k$번째 행이 $\mathbf{b_k}^T$인 행렬입니다. 이것을 사용하면 (6.8)식의 expoenetial 안쪽 부분을 다음과 같이 표현할 수 있습니다.

$$
\sum_{i=1}^n (\mathbf{y_i} - \boldsymbol{\theta})^T \Sigma^{-1} (\mathbf{y_i} - \boldsymbol{\theta}) = \text{tr}(\mathbf{S_{\theta}} \Sigma^{-1})
$$

이 떄 $\boldsymbol{S_{\theta}} = \sum_{i=1}^n (\boldsymbol(y_i - \theta) (\boldsymbol(y_i - \theta) ^T$ 입니다.

행렬 $\boldsymbol{S_{\theta}}$는 모평균이 $\boldsymbol{\theta}$로 추정될 때 벡터들 $\mathbf{y_1}, ..., \mathbf{y_n}$의 잔차제곱합(residual sum of square) 행렬입니다.

$\frac{1}{n} \mathbf{S_{\theta}} | \boldsymbol{\theta}$는 참 공분산 행렬 Cov[$\mathbf{Y}$]의 불편추정량을 제공합니다(더 일반적으로 $\boldsymbol{\theta}$가 표본에 조건부로 주어지지 않았을 때, 공분산 행렬은 $\Sigma (\mathbf{y_i} - \mathbf{\bar{y}})(\mathbf{y_i} - \mathbf{\bar{y}})^T/(n-1)$ 이고 이것은 $\Sigma$의 불편추정량입니다.).

### $\Sigma$의 사후 분포

자 이제 (6.7), (6.8) 사용하면 다음과 같은 $\Sigma$의 조건부 분포를 얻을 수 있습니다.

$$
p(\Sigma| \mathbf{y_1}, ..., \mathbf{y_n}, \boldsymbol{\theta})  \\
\propto p(\Sigma) \times p(\mathbf{y_1}, ..., \mathbf{y_n}|\boldsymbol{\theta}, \Sigma) \\
\propto \bigg( |\Sigma|^{-(\nu_0 + p + 1)/2} \text{exp} \bigg(-\text{tr}(\mathbf{S_0}\Sigma^{-1})/2 \bigg) \bigg) \times \bigg(|\Sigma|^{-n/2} \text{exp} \bigg(-\text{tr}(\mathbf{S_{\theta}}\Sigma^{-1})/2 \bigg) \bigg) \\
= |\Sigma|^{-(\nu_0 + n + p + 1)/2} \text{exp} \{ -\text{tr}([\mathbf{S_0} + \mathbf{S_{\theta}}]\Sigma^{-1})/2 \}
$$

즉 다음과 같은 결과를 얻게 됩니다.

$$
\{\Sigma | \mathbf{y_1}, ..., \mathbf{y_n}, \boldsymbol{\theta} \} \sim \text{inverse-Wishart}(\nu_0 + n, [\mathbf{S_0} + \mathbf{S_{\theta}}]^{-1})
$$

이것은 다음과 같이 해석할 수 있습니다

$\nu_0 + n$ : "사전 표본 크기" $\nu_0$과 데이터 표본 크기 $n$의 합

$\mathbf{S_0} + \mathbf{S_{\theta}}$ : "사전" 잔차 제곱합 + 데이터 잔차 제곱합

추가적으로 모 공분산 행렬의 조건부 기댓값은

\begin{align}
E[\Sigma|\mathbf{y_1}, ..., \mathbf{y_n}, \boldsymbol{\theta}] = \frac{1}{\nu_0 + n - p - 1}(\mathbf{S_0} + \mathbf{S_{\theta}}) \newline
= \frac{\nu_0 - p - 1}{\nu_0 + n - p - 1} \frac{1}{\nu_0 - p - 1} \mathbf{S_0} + \frac{n}{\nu_0 + n - p - 1} \frac{1}{n} \mathbf{S_{\theta}}
\end{align}

이고, 이것을 사전 기댓값($\frac{1}{\nu_0 - p - 1} \mathbf{S_0}$)과 불편추정량($\frac{1}{n} \mathbf{S_{\theta}}$)의 가중 평균이라고 볼 수 있습니다.

그리고 $\mathbf{S_{\theta}}$가 참 모 공분산 행렬로 수렴한다는 것을 보일 수 있기 때문에, $\Sigma$의 사후 기댓값은 심지어 실제 모분포가 다변량 정규분포가 아니더라도 모 공분산의 consistent 추정량이 됩니다. 
