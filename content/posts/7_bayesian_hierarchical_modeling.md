---
title: A First Course In Bayesian Statistical Methods Chapter 7. Bayesian Hierarchical Modeling
date : 2021-04-18T18:32:23+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

# 7. Bayesian Hierarchical Modeling

이 챕터에서는 다음과 같은 두 가지를 배워볼 것입니다.

1. 그룹간 평균 비교 - 각 그룹의 평균과 그룹간 평균의 차이 비교
2. 정규 계층 모델(Normal Hierarchical Model) - 그룹 안의 변동성과 그룹간 변동성 측정, 그룹간 분산과 평균의 동질성 측정

## 7.1 그룹간 비교

 ![IMG_F43AFC193167-1](https://user-images.githubusercontent.com/57588650/113250916-5dea9280-92fc-11eb-8b20-d10332f82db6.jpeg)


**그림. 8.1.** 두 학교들의 수학 성적 표본을 나타낸 박스플롯과 두 학교의 모평균이 같다고 가정했을 때의 영분포(null distribution). 오른쪽 그림의 회색 선은 관찰값들의 t-통계량을 나타냅니다.

**그림. 8.1**의 첫 번째 패널은 두 개 학교의 수학 성적을 나타냅니다. 학교 1은 31명, 학교 2는 28명의 학생들이 약 600명의 전교생들 중에 무작위로 추출되었습니다. 

자 이제 학교 1의 모평균 $\theta_1$을 추정하고, 이를 학교 2의 모평균 $\theta_2$와 비교해보겠습니다. 표본 데이터에서 얻을 수 있는 값은 $\bar{y}_1 = 50.81, \bar{y}_2 = 46.15$이고 이는 $\theta_1$이 $\theta_2$보다 크다고 제안합니다. 그러나 만약 각각의 학교에서 다른 샘플이 추출된다면 $\bar{y}_2$가 $\bar{y}_1$보다 클 가능성도 배제할 수 없습니다. 이 때 관측된 평균의 차이 $\bar{y}_1 - \bar{y}_2 =.4.66$이 표본 변동성과 비교해 큰 것인지를 알아보기 위해서는 t-통계량을 계산하는 것이 일반적인 방법입니다. t-통계량이란 다음과 같은 관측된 차이와 추정된 표준 편차의 비율 입니다. 

$$
t(\mathcal{y_1, y_2}) = \frac{\bar{y}_1 - \bar{y}_2}{s_p\sqrt{1/n_1 + 1/n_2}}
$$

$$
= \frac{50.81 - 46.15}{10.44\sqrt{1/31 + 1/28}} = 1.74
$$

이 때 $s_p^2 = [(n_1 - 1)s_1^2 + (n_2 - 1) s_2^2] / (n_1 + n_2 -2)$는 두 그룹의 합동 모분산 추정량(pooled estimate of the population variance)입니다. 1.74은 큰 값일까요? 만약 두 학교가 모두 같은 평균과 분산을 가지는 정규 분포라면, t-통계량 $t(\mathcal{y_1, y_2})$의 표본 분포는 $n_1 + n_2 - 2 = 57$의 자유도를 가지는 t-분포입니다. 이 분포의 밀도는 그림 8.1의 두 번째 그림에 관측된 t-통계량과 함께 표현되어있습니다. 따라서 만약 두 모집단이 같은 정규 모집단을 따른다면, 실험 이전에 $t(\mathcal{y_1, y_2})$의 절댓값이 1.74보다 큰 데이터가 추출될 확률은 p = 0.087입니다.

여러분이 알듯 이 p값을 바로 (양측) p-value라고 합니다. 작은 p값이 일반적으로는 $\theta_1$과 $\theta_2$이 다르다는 증거가 되지만, p-value가 $\theta_1 = \theta_2$일 확률은 아니니 헷갈리지 않길 바랍니다. 두 집단간의 평균 차이라는 목적에서 이 방식이 통계적으로 완전하게 정의되지는 않지만, p-value는 파라미터 추정과 모델 선택에서 자주 쓰입니다. 예를 들어 다음은 두 그룹간의 모평균을 비교하는 일반적인 데이터 분석 과정입니다. 

### p-value에 기반한 모델 선택

    만일 p < 0.05라면,
   - 두 그룹이 같은 분포를 가지고 있다는 가설을 기각합니다.
   - $\theta_1 \neq \theta_2$라는 결론을 내립니다.
   - $\hat{\theta_1} = \bar{y}_1, \hat{\theta}_2 = \bar{y}_2$라는 추정량을 사용합니다.
    
    

    만일 p < 0.05라면
   - 두 그룹이 같은 분포를 가지고 있다는 가설을 채택합니다.
   - $\theta_1 = \theta_2$라는 결론을 내립니다
   - $\hat{\theta}_ 1 = \hat{\theta}_ 2 = (\Sigma y_{i,1} + \Sigma y_{i,2}) / (n_1 + n_2)$라는 추정량을 사용합니다.

이 데이터 분석 과정에 따르면 두 모집단은 완벽하게 구분된 것으로 생각하거나, 그들을 정확하게 동일한 것으로 생각합니다. 이러한 극단적으로 이분법적인 결과가 과연 옳은 것일까요? 수학 점수 데이터를 위의 과정으로 분석한 결과인 p = 0.087은 0.05보다 크므로 두 그룹을 수적으로(numerically) 동일하게 취급해야 한다고 말합니다. 둘이 다르다는 몇 가지 증거가 있음에도 불구하고 말이죠. 반대로 학교 1에서 더 많은 성적 높은 학생들이 뽑히고 학교 2에서 더 많은 성적 낮은 학생들이 뽑히는 시나리오를 어렵지 않게 상상할 수 있습니다. 이 경우에서 관찰된 p-value는 0.04나 0.05가 될 수 있겠죠. 후자의 케이스에서는 각각의 모집단을 구분하여 취급할 것입니다. 오직 학교 1의 데이터로만 $\theta_1$을 추정하고, 학교 2의 데이터로만 $\theta_2$를 추정하면서 말이죠. 이 후자의 방식은 불충분해 보입니다. 왜냐하면 두 표본이 모두 동질적인 학생들의 모집단에서 같은 것을 측정하기 때문이죠. 따라서 한 그룹에서의 정보가 다른 그룹의 평균을 추정하는데 도움을 줄 수 있다는 것을 이해할 수 있을것입니다.

위에서 설명한 p-value에 기반한 과정은 $\theta_1$을 $p < 0.05$일 때는 $w = 1$, $p > 0.05$일 때는 $w= n_1/(n_1+n_2)$인 $\hat{y} = w\bar{y}_1 + (1 - w)\bar{y}_2$로써 추정하는 것으로 다시 표현할 수 있습니다. 이러한 극단적인 과정을 사용하는 대신에, $w$의 값이 연속적으로 서로 다른 값을 가지며 상대적인 표본 크기 $n_1, n_2$와 표분 분산 $\sigma^2$, 그리고 두 모집단의 동질성에 대한 사전 정보에 의존하는 값이라고 생각하는 것이 더 말이 됩니다. 이것과 비슷한 추정량이 그룹간 공유되는 정보가 있다는 가정을 통한 베이지안 분석을 통해 만들어질 수 있습니다. 두 그룹으로 부터의 다음과 같은 표본 모델을 생각해봅시다.

$$
Y_{i,1} = \mu + \delta + \epsilon_{i,1}
$$

$$
Y_{i,2} = \mu - \delta + \epsilon_{i,2}
$$

$$
\epsilon_{i,j} \sim \text{i.i.d. Normal}(0, \sigma^2)
$$

이러한 $\theta_1 = \mu + \delta, \theta_2 = \mu - \delta$인 파라미터화(parameterization)를 사용하면 $\delta$가 두 모평균의 차이의 반 ($\theta_1 - \theta_2)/2$를 나타내고, $\mu$가 합동 평균(pooled average) ($\theta_1 + \theta_2)/2$를 나타낸다는 것을 알 수 있습니다. 이 때 미지의 모수에 대한 다음과 같은 켤레 사전 분포를 사용할 수 있습니다.

\begin{align}
p(\mu, \delta, \sigma^2) = p(\mu) \times p(\delta) \times p(\sigma^2) \newline
\mu \sim \text{normal}(\mu_0, \gamma_0^2) \newline
\delta \sim \text{normal}(\delta_0, \tau_0^2) \newline
\sigma^2 \sim \text{inv-gamma}(\nu_0/2, \nu_0 \sigma_0^2/2).
\end{align}

\begin{align}
\mu|\mathbf{y_1, y_2}, \delta, \sigma^2 \sim \text{Normal}(\mu_n, \gamma_n^2), \text{where} \newline
\mu_n = \gamma_n^2 \times [\mu_0 / \gamma_0^2 + \sum^{n_1}_ {i=1} (y_{i,1} - \delta) / \sigma^2 + \sum^{n_2}_ {i=1} (y_{i,1} + \delta) / \sigma^2] \newline
\gamma_n^2 = [1/\gamma_0^2 + (n_1 + n_2) / \sigma^2]^{-1}
\end{align}

\begin{align}
\delta|\mathcal{y_1, y_2}, \mu, \sigma^2 \sim \text{Normal}(\delta_n, \tau_n^2), \text{where} \newline
\delta_n = \tau_n^2 \times [\delta_0 / \tau_0^2 + \sum (y_{i,1} - \mu) / \sigma^2 - \sum (y_{i,2} - \mu) / \sigma^2] \newline
\tau_n^2 = [1/\tau_0^2 + (n_1 + n_2) / \sigma^2]^{-1}
\end{align}

\begin{align}
\sigma^2|\mathcal{y_1, y_2}, \mu, \delta \sim \text{inv-Gamma}(\nu_n/2, \nu_n \sigma_n^2/2), \text{where} \newline
\nu_n = \nu_0 + n_1 + n_2 \newline
\nu_n \sigma_n^2 = \nu_0 \sigma_0^2 + \Sigma(y_{i,1} - [\mu + \delta])^2 + \Sigma(y_{i,2} - [\mu - \delta])^2
\end{align}

이러한 식들은 복잡해서 어떤 것을 말하는지 이해가 잘 안될 수 있습니다. 그럴 땐 사전 파라미터에 극단적인 값을 넣어서 어떻게 되는지 확인해봅시다. 예를 들어 $\nu_0 = 0$이라면

$$
\sigma_n^2 = \frac{\Sigma(y_{i,1} - [\mu + \delta])^2 + \Sigma(y_{i,2} - [\mu - \delta])^2}{n_1 + n_2}
$$

이고, 이것은 만약 $\mu$와 $\delta$를 알 때의 통합 표본 분산의 추정값 입니다. 이와 비슷한 방식으로 만약 $\mu_0 = \delta_0 = 0$이고 $\gamma_0^2 = \tau^2_0 = \infty$라면 ($0/\infty = 0$이라고 정의합시다)

$$
\mu_n = \frac{\Sigma(y_{i,1} - \delta) + \Sigma(y_{i,2}+\delta)}{n_1 + n_2}, \ \ \delta_n = \frac{\Sigma(y_{i,1} - \mu) - \Sigma(y_{i,2} -\mu) }{n_1 + n_2}
$$

그리고 만약 $\mu$에 $\mu_n$을 넣고, $\delta$에 $\delta_n$을 넣으면 $\bar{y_1} = \mu_n + \delta_n$과 $\bar{y_2} = \mu_n - \delta_n$을 얻을 수 있습니다.

### 수학 성적 데이터 분석

이 수학 시험은 전국적으로 평균 50, 표준편차 10인 점수를 보이도록 표준화되었습니다. 만약 이 두 학교가 극단적인 예외 케이스로 판명난게 아니라면, 이 정보가 합리적인 사전 파라미터가 될 수 있습니다. 즉 학교간 표본 분산을 과대평가 할 수 있긴 하지만, 사전 파라미터로 $\mu_0 = 50, \sigma_0^2 = 10^2 = 100$을 설정하겠습니다. 이 사전 분포를 $\gamma_0^2 = 25^2 = 625$로 더 퍼지게 해보겠습니다. 이 $\delta$에 대한 사전분포를 $\delta_0 = 0$로 선택하는 것은 $\theta_1 > \theta_2$와 $\theta_2 > \theta_1$일 확률이 같다는 것을 표현합니다. 마지막으로 점수는 0과 100 사이에 있고, $\theta_1, \theta_2$의 차이의 절댓값의 절반은 50보다 작습니다. 따라서 $\tau_0^2 = 25^2 = 625$는 합리적으로 퍼져있는 것으로 보입니다. 

위에 주어진 full conditional distribution을 사용하면, 사후 분포 $p(\mu, \delta, \sigma^2 | \mathbf{y_1, y_2})$로 다음과 같은 코드를 통해 근사하는 깁스 샘플러를 만들 수 있습니다.


```python
import numpy as np
import pandas as pd
import random 

y1 = np.array([52.11, 57.65, 66.44, 44.68, 40.57, 35.04, 50.71, 66.17, 39.43, 
46.17, 58.76, 47.97, 39.18, 64.63, 69.38, 32.38, 29.98, 59.32, 
43.04, 57.83, 46.07, 47.74, 48.66, 40.8, 66.32, 53.7, 52.42, 
71.38, 59.66, 47.52, 39.51])

n1 = len(y1)

y2 = np.array([52.87, 50.03, 41.51, 37.42, 64.42, 45.44, 46.06, 46.37, 46.66, 
29.01, 35.69, 49.16, 55.9, 45.84, 35.44, 43.21, 48.36, 74.14, 
46.76, 36.97, 43.84, 43.24, 56.9, 47.64, 38.84, 42.96, 41.58, 
45.96])

n2 = len(y2)

### 앞에서 설정한 사전 파라미터

mu0 = 50 ; g02 = 625
delta0 = 0 ; t02 = 625
s20 = 100 ; nu0 = 1

###


### 시작 값 
mu = (np.mean(y1) + np.mean(y2)) / 2
delta = (np.mean(y1) - np.mean(y2)) / 2
###


### 깁스 샘플러 시작
MU = list()
DEL = list()
S2 = list()

random.seed(1)

for s in np.arange(0,5000):
    
    ## s2 업데이트
    s2 = 1/np.random.gamma((nu0 + n1 + n2)/2, ( nu0 * s20 + np.sum((y1 - mu - delta)**2) + np.sum((y2 - mu + delta)**2) )/2 ,1)
    ##
    
    ## mu 업데이트
    var_mu = 1 / (1/g02 + (n1+n2) / s2)
    mean_mu = var_mu * (mu0 / g02 + sum(y1 - delta)/s2 + sum(y2+delta)/s2)
    mu = np.random.normal(mean_mu, np.sqrt(var_mu), 1)
    ##
    
    ## del 업데이트
    var_del = 1 / (1/t02 + (n1+n2)/s2)
    mean_del = var_del * (delta0/t02 + sum(y1 - mu)/s2 - sum(y2-mu)/s2)
    delta = np.random.normal(mean_del, np.sqrt(var_del), 1)
    ##
    
    ## 파라미터 값 저장
    MU = np.append(MU, mu)
    DEL = np.append(DEL, delta)
    S2 = np.append(S2,s2)
    
    
```

**그림 8.2**는 $\mu$와 $\delta$의 주변 사후 분포와 그들이 사전 분포보다 얼마나 더 뾰족한지를 보여줍니다. 측히, $2 \times \delta$(두 학교 사이의 평균 점수 차이)의 사후 95% 분위수 기반 신뢰 구간은 (-0.61, 9.98)입니다. 이 구간이 0을 포함하긴 하지만 이러한 차이는 사전 그리고 사후 분포는 학교 1의 모평균이 학교 2보다 높다는 명확한 증거가 됩니다. 추가적으로 사후 확률 $\text{Pr}(\theta_1 > \theta_2 | \mathbf{y_1, y_2}) = \text{Pr}(\delta > 0 | \mathbf{y_1, y_2}) \approx 0.96$이고, 이에 대응하는 사전 확률은 $\text{Pr}(\delta > 0) = 0/50$이었습니다.

그러나 이 확률을 무작위로 선택한 학교 1의 학생이 학교 2에서 선택된 학생보다 점수가 높을 확률과는 다르다는 것을 헷갈리지 말아야 합니다. 그 확률은 이게 아니라 결합 사후 예측 분포를 사용해 $\text{Pr}(Y_1 > Y_2|\mathbf{y_1, y_2}) \approx 0.62$라고 구해야 합니다.

![IMG_E85A02BF0E05-1](https://user-images.githubusercontent.com/57588650/113506233-88e41900-957e-11eb-99be-5819141b0549.jpeg)

**그림. 8.2** $\mu, \delta$의 사전, 사후 분포

## 7.2 여러 그룹간 비교

위의 데이터는 전체 학교의 모집단과 각 학교의 모집단 데이터를 포함하고 있습니다. 이것과 같은 데이터는 중첩된 모집단의 계층을 가지고 있으며, "계층적(hierarchical)" 또는 "다층(multilevel)" 데이터라고 합니다. 이와 같은 데이터는 다음을 예로 들을 수 있습니다.

* 여러 병원들에 있는 환자들
* 동물 그룹 안의 유전자들
* 어떤 국가 안의 지역 안의 도시에 있는 사람들

가장 단순한 다층 데이터는 두 개의 층을 가지고 있습니다. 하나는 그룹의 데이터를 가지고 있고, 하나는 그룹 안의 유닛들의 데이터를 가지고 있죠. 이럴 때 $y_{i,j}$를 그룹 $j$의 $i$번째 유닛의 데이터라고 하겠습니다.

### 7.2.1 교환가능성과 계층적(hierarchical) 모델

확률 변수 $Y_1, Y_2, ... , Y_n$이 교환 가능하다는 것은 수열에 대한 우리의 정보를 표현하는 확률 밀도 함수가 모든 순열 $\pi$에 대해 $p(y_1, ..., y_n) = p(y_{\pi_1}, ..., y_{\pi_n})$을 만족한다는 것을 의미합니다. 교환 가능성은 확률 변수를 구별할 정보가 부족할 때, $p(y_1, ..., y_n)$에 대한 합리적인 특성입니다. 예를 들어 만약 $Y_1, ..., Y_n$이 특정 학교에서 $n$명의 무작위로 선택된 학생들의 수학 점수라면, 학생에 대한 다른 정보가 없을 때 그들의 수학 점수를 교환 가능한 것으로 취급할 것입니다. 만약 교환 가능성이 모든 $n$의 값에서 유지될 때, [de Finetti의 정리](https://en.wikipedia.org/wiki/De_Finetti's_theorem)에 따르면 우리의 정보와 일치하는 공식은 다음과 같습니다.

$$
\phi \sim p(\phi)
$$

$$
Y_1, ... , Y_n | \phi \sim \text{i.i.d.} \ p(y|\phi)
$$

다른 말로 하면, 확률변수는 몇몇의 알려지지 않았지만 고정된 값인 모집단의 특성 $\phi$에 의해 표현되는 모집단으로부터의 독립적인 표본들이라고 생각될 수 있습니다. 예를 들면 정규 모델에서 $\phi =$ {$\theta, \sigma^2$}이고, 이는 데이터를 조건부 i.i.d.인 normal($\theta, \sigma^2$)로 모델링 합니다. 

이제 $Y_j =${$Y_{1,j}, ..., Y_{n_j,j}$}일 때의 계층적 데이터 {$\mathbf{Y_1, ..., Y_m}$}에 대한 우리의 정보를 표현하는 모델에 대해 알아봅시다. 모델 $y(\mathbf{y_1, ..., y_m})$은 어떤 특성을 가져야 할까요? 우선 다음과 같이 하나의 그룹 $j$에 대한 주변 확률 밀도를 정의하겠습니다.

$$
p(\mathbf{y_j} = p(y_{1,j}, ..., y_{n,j})
$$

지금부터는 $Y_{1,j}, ... , Y_{n_j, j}$를 독립으로 취급하지 않아야 합니다. 예를 들어 $p(y_{n_j,j}|y_{1,j}, ..., y_{n_j-1,j}) = p(y_{n_j}, j)$, 즉 $Y_{1,j}, ... , Y_{n_j-1, j}$의 값은 $Y_{n_j, j}$에 대해 아무런 정보도 주지 않습니다. 그러나 $Y_{1,j}, ... , Y_{n_j, j}$에 대해 가지고 있는 유일한 정보가 이것들이 그룹 $j$에서 뽑힌 무작위 샘플인 것이라면, $Y_{1,j}, ... , Y_{n_j, j}$은 교환 가능합니다. 만약 그룹 $j$가 표본 사이즈 $n_j$와 비교해 크다면, de Finetti의 정리와 Diaconis and Freedman(1980)의 결과에서 그룹 $j$ 안에 있는 데이터를 몇가지의 그룹 특성 파라미터 $\phi_j$가 주어졌을 때 다음과 같은 조건부 i.i.d 모델로 만들 수 있습니다.

$$
Y_{1,j}, ..., Y_{n_j, j} | \phi_j \sim \text{i.i.d.} \ \ p(y|\phi_j)
$$

그렇다면 그룹 특성 파라미터인 $\phi_1, ..., \phi_m$에 대한 우리의 정보를 어떻게 표현할 수 있을까요? 이전에 말했듯, 우리는 이들을 독립된 파리미터로 취급하지 않을 것입니다. 왜냐하면 $\phi_1, ..., \phi_{m-1}$의 값을 안다고 해서 $\phi_m$에 대한 정보를 바꾸진 않을 것이기 때문입니다. 하지만 만약 그룹 그 자신들이 더 큰 모집단을 가지고 있는 그룹에서 뽑힌 샘플들이라면, 그룹 특성 파리미터들은 교환가능성을 가지고 있다는 것이 적절합니다. 두 번째로 de Finetti의 정리를 적용하면 다음과 같은 결과를 얻을 수 있습니다

표본 모델 $p(\phi|\psi)$와 미지의 파라미터 $\psi$일 때

$$
\phi_1, ..., \phi_m | \psi \sim \text{i.i.d.} \ \ p(\phi|\psi)
$$

이를 통해 다음과 같은 세 개의 확률 분포를 얻을 수 있습니다.

{$y_{1,j}, ..., y_{n_j,j}|\phi_j$} $\sim \text{i.i.d.} \ \ p(y|\theta_j)$ (그룹 내 표본 추출 변동성)

{$\phi_1, ..., \phi_m|\psi$} $\sim \text{i.i.d.} \ \ p(\theta|\psi)$ (그룹 간 표본 추출 편동성)

$\psi \sim p(\psi)$ (사전 분포)

여기서 중요한 것은 두 분포 모두 대상의 모집단 사이의 표본 추출 변동성을 나타내지만 그룹 안의 변동성을 뜻하는 분포 $p(y|\theta)$와 그룹 간의 변동성을 뜻하는 분포 $p(\theta|\psi)$를 구분하는 것입니다. 반대로 $p(\psi)$는 하나의 미지수지만 고정된 값에 대한 정보를 표현합니다. 이러한 이유 때문에 $p(y|\theta), p(\phi|\psi)$는 표본 분포이며, 사전 분포인 $p(\psi)$와는 개념적으로 구분됩니다. 특히 그룹 내, 그룹 간 표본 분포인 $p(y|\theta)$와 $p(\phi|\psi)$는 데이터를 사용해 추정되지만, 사전 분포인 $p(\psi)$는 데이터로 부터 추정되지 않습니다.

## 7.3 계층적 정규 모델(hierarchical normal model)

여러 모집단간의 평균 차이를 표현하는 모델 중 유명한 것이 다음과 같이 그룹 내, 그룹 간 표본 추출 모델이 모두 정규분포인 "계층적 정규 모델(hierachical normal model)"입니다.

$$
\phi_j = (\theta_j, \sigma^2), \ p(y|\phi_j) = \text{normal}(\theta_j, \sigma^2) \ \ \ \ (\text{within-group model}) \ (7.1)
$$

$$
\psi = (\mu, \tau^2), \ p(\theta_j|\psi) = \text{normal}(\mu, \tau^2) \ \ \ \ \text{(between-group model)} \ \ (7.2)
$$

![IMG_C90D0FB66C43-1](https://user-images.githubusercontent.com/57588650/113820298-7c95d100-97b5-11eb-92c5-3b3383642a58.jpeg)


**그림. 7.3.** 기본적인 계층적 정규 모델

위의 그림을 보면 (7.1), (7.2) 식이 의미하는 바를 더 쉽게 이해할 수 있을 것입니다. 이 때 중요한 것은 $p(\phi|\psi)$가 오직 그룹간 평균 간의 차이를 나타낼 뿐이지, 그룹 개개의 분산 차이를 나타내진 않는다는 점입니다. 사실 그룹 내 표본 변동성 $\sigma^2$은 그룹간 차이가 없는 것으로 가정됩니다. 이 챕터의 마지막에서 모델에 그룹 개개의 분산을 나타내는 요소를 추가함으로써 이 가정을 없에보도록 하겠습니다.

이 모델에서 고정됐지만 미지의 파라미터는 $\mu, \tau^2, \sigma^2$입니다. 편의를 위해 이 파라미터들의 표준 semiconjugate normal, inverse-gamma 사전 분포를 사용하도록 하겠습니다.

\begin{align}
1/\sigma^2 \sim \text{gamma}(\nu_0/2, \nu_0 \sigma_0^2/2) \newline
1/\tau^2 \sim \text{gamma}(\eta_0/2, \eta_0 \tau_0^2/2) \newline
\mu \sim \text{normal}(\mu_0, \gamma_0^2)
\end{align}

## 7.3.1 사후 추론

미지의 값들

1. {$\theta_1, ..., \theta_m$} : 그룹별 평균

2. $\sigma_2$ : 그룹 내 표본 추출 변동성

3. $\mu, \tau^2$ : 그룹별 평균의 모평균, 모분산

이러한 모수들의 결합 사후 추론 : 사후 분포 $p(\theta_1, ..., \theta_m, \mu, \tau^2, \sigma^2|\mathbf{y_1, ..., y_m})$을 근사하는 깁스 샘플러를 만들기

깁스 샘플러? : 연속적으로 각각의 모수를 그들의 full conditional distribution에서 추출함으로써 진행됨

이를 위해 full conditional distribution을 구하는 일은 이전의 4,5,6장에서 설명했습니다. 이 시점에서 이제 알아야 할 것은 단변량 정규 모델과 현재의 계층적 정규 모델링 사이에 어떤 유사점이 있는지 파악하는 것입니다. 다음의 인수분해가 그것을 위해 유용합니다.

\begin{align}
p(\theta_1, ..., \theta_m, \mu, \tau^2, \sigma^2 | \mathbf{y_1, ..., y_m}) \newline
\propto p(\mu, \tau^2, \sigma^2) \times p(\theta_1, ..., \theta_m | \mu, \tau^2, \sigma^2) \times p(\mathbf{y_1, ..., y_m}|\theta_1, ..., \theta_m, \mu, \tau^2, \sigma^2) \newline
= p(\mu) p(\tau^2) p(\sigma^2) \bigg( \prod^m_{j=1} p(\theta_j|\mu, \tau^2) \bigg) \bigg( \prod^m_{j=1} \prod^{n_j}_{i=1} p(y_{i,j} | \theta_j, \sigma^2) \bigg) \ \ \ \ \ (7.3)
\end{align}

마지막 큰 괄호 안에 있는 식은 우리 모델의 "조건부 독립" 특성에서 나온 결과입니다. {$\theta_1, ..., \theta_m, \mu, \tau^2$}가 주어졌을 때, 확률 변수 $Y_{1,j}, ..., Y_{n_j, j}$들은 오직 $\theta_j, \sigma^2$에만 의존하고 $\mu, \tau^2$에는 그렇지 않은 분포와 독립입니다. 이것을 **그림 7.3**을 보면서 이해하면 도움이 될 것입니다. 

($\mu, \tau^2$)에서 각각의 $\mathbf{Y_j}$로 가는 길은 $(\mu, \tau^2)$이 오직 $\theta_j$를 통해 간접적으로 $\mathbf{Y_j}$들에 정보를 제공한다는 것을 나타냅니다. 이것이 $\theta_j$를 따로 구분한 이유입니다.

### $\mu, \tau^2$의 Full conditional distribution

$\mu$와 $\tau^2$의 함수로써, (7.3)의 식은 다음에 비례합니다

$$
p(\mu) p(\tau^2) \prod^m_{j=1} p(\theta_j|\mu, \tau)
$$

그렇게 때문에 $\mu, \tau^2$의 full conditional distribution 역시 이 값에 비례합니다. 특히 다음과 같은 것을 의미합니다.

$$
p(\mu|\theta_1, ..., \theta_m, \tau^2, \sigma^2, \mathbf{y_1, ..., y_m}) \propto p(\mu) \prod p(\theta_j|\mu, \tau^2)
$$

$$
p(\tau^2 | \theta_1, ..., \theta_m, \mu, \sigma^2, \mathbf{y_1, ... , y_m}) \propto p(\tau^2) \prod p(\theta_j|\mu, \tau^2)
$$

이러한 분포는 명확하게 하나의 표본에서 정규 분포 문제의 full conditional distribution과 같습니다. 이전에 우리는 모평균과 정규 모집단에서의 분산의 full conditional distribution을 독립된 정규 분포와 inverse-gamma 분포에서 구했습니다. 지금의 이 상황에서 $\theta_1, ..., \theta_m$은 정규 모집단에서 뽑힌 i.i.d. 표본들이고, $\mu, \tau^2$은 모집단의 알려지지 않은 평균과 분산입니다. $y_1, ..., y_n$이 normal($\theta, \sigma^2)$에서 뽑힌 i.i.d. 값들이고 $\theta$는 정규 사전 분포를 가진다면, $\theta$의 조건부 분포 또한 정규 분포입니다.지금 이 상황과 정확히 일치합니다. 즉 $\theta_1, ..., \theta_m$이 i.i.d. normal($\mu, \tau^2$)을 따르고 $\mu$가 정규 사전 분포를 따르기 때문에 $\mu$의 조건부 분포 역시 정규 분포를 따르게 됩니다. 이와 같이 $\sigma^2$이 inverse-gamma 조건부 분포를 가졌던 것 처럼, $\tau^2$역시 현 상황에서 inverse-gamma 조건부 분포를 가지게 됩니다. 이를 식으로 나타내면 다음과 같습니다.

$$
\mu | \theta_1, ..., \theta_m, \tau^2 \sim \text{normal}(\frac{m\bar{\theta}/\tau^2 + \mu_0 / \gamma_0^2}{m/\tau^2 + 1/\gamma_0^2}, [m/\tau^2 + 1 / \gamma_0^2]^{-1})
$$

$$
1/\tau^2 | \theta_1, ..., \theta_m, \mu \sim \text{gamma}(\frac{\eta_0 + m}{2}, \frac{\eta_0 \tau_0^2 + \Sigma(\theta_j - \mu)^2}{2}).
$$

### $\theta_j$의 full conditional distribution

식 (7.3)에서 $\theta_j$에 의존하는 항들은 $\theta_j$의 full conditional distribution이 다음과 같은 비례 관계가 있다는 것을 보여줍니다.

$$
p(\theta_j|\mu, \tau^2, \sigma^2, \mathbf{y_1, ..., y_m}) \propto p(\theta_j|\mu, \tau^2) \prod^{n_j}_ {i=1} p(y_{i,j}|\theta_j, \sigma^2)
$$

이것은 {$\mu, \tau^2, \sigma^2, \mathbf{y_j}$}가 주어졌을 때, $\theta_j$의 조건부 확률은 다른 $\theta$들과 $j$가 아닌 그룹에서 뽑힌 데이터들에 모두 조건부 독립이란 것을 의미합니다. **그림 7.3**을 보면 더 쉽게 이해할 수 있습니다. 각각의 $\theta_j$에서 모든 다른 $\theta_k$로 가는 길이 있지만, 그 길은 ($\mu, \tau^2$) 또는 $\sigma^2$를 통해야만 합니다. 이것은 즉 $\theta$들은 서로에게 $\mu, \tau^2, \sigma^2$에 포함되어 있는 것을 넘어서는 정보를 주지 않는다는 것을 의미합니다. 

식 (7.3)은 $\theta_j$의 정규 밀도 함수($\prod^m_{j=1} p(\theta_j|\mu, \tau^2)$)에 $\theta_j$가 평균인 정규 밀도들의 곱($\prod^m_{j=1} \prod^{n_j}_{i=1} p(y_{i,j} | \theta_j, \sigma^2)$)을 곱한 값을 포함합니다. 수학적으로 이것은 하나의 샘플을 가지는 정규 모델의 설정에서 $\theta$의 표본 추출 모델 대신 $p(\theta_j|\mu, \tau^2)$이 사전 분포인 것과 정확히 일치합니다. 따라서 full conditional distribution은 다음과 같습니다.

{$\theta_j|y_{1,j}, ..., y_{n_j,j}, \sigma^2$} $\sim$ normal$ \bigg( \frac{n_j \bar{y}_j/\sigma^2 + 1/\tau^2}{n_j / \sigma^2 + 1 / \tau^2}, [n_j / \sigma^2 + 1 / \tau^2]^{-1} \bigg)$

### $\sigma^2$의 full conditional distribution

**그림 7.3**과 이전에 본 $\mu, \tau^2, \theta_j$의 full conditional distiribution을 구하는 과정을 보면, $\sigma^2$은 {$\mathbf{y_1, ..., y_m}, \theta_1, ..., \theta_m$}이 주어졌을 때 {$\mu, \tau^2$}과 조건부 독립이란 사실을 알 수 있습니다. $\sigma^2$의 full conditional distribution을 유도한 것은 이 경우엔 m개의 구분된 그룹으로부터 $\sigma^2$에 대한 정보를 가진다는 것만 빼면 하나의 샘플을 가지는 정규 모델과 비슷합니다.

\begin{align}
p(\sigma^2|\theta_1, ..., \theta_m, \mathbf{y_1, ..., y_m}) \propto p(\sigma^2) \prod^m_{j=1} \prod^{n_j}_{i=1} p(y_{i,j} | \theta_j, \sigma^2) \newline
\propto (\sigma^2)^{-\nu_0/2 + 1} e^{-\frac{\nu_0 \sigma_0^2}{2\sigma^2}}(\sigma^2)^{-\Sigma n_j / 2} e^{-\frac{\Sigma \Sigma (y_{i,j} - \theta_j)^2}{2\sigma^2}}
\end{align}

$\sigma^2$의 제곱수들을 더하고 e 안에 있는 항들을 모으면 이것이 다음과 같은 inverse-gamma 밀도 함수에 비례한다는 것을 알 수 있습니다.

$$
1/\sigma | \mathbf{\theta, y_1, ..., y_n} \sim \text{gamma}(\frac{1}{2}[\nu_0 + \sum^n_{j=1} n_j], \frac{1}{2}[\nu_0\sigma_0^2 + \sum^m_{j=1} \sum^{n_j}_{i=1}(y_{i,j} - \theta_j)^2])
$$

$\Sigma \Sigma (y_{i,j} - \theta_j)^2$이 그룹 내 평균이 주어졌을 때 전체 그룹들의 조건부 잔차 제곱합이고, 따라서 조건부 분포는 분산의 합동 표본(pooled-sample) 추정량 주변의 확률에 집중되어있다는 것에 주목합시다.
