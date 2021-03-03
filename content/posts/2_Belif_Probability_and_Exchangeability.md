---
title: A First Course in Bayesian Statistical Methods Chapter 2. Belif, Probability and Exchangeability
author : 오태환
date: '2021-03-02T20:09:32+09:00'
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 2. 믿음, 확률 그리고 교환가능성(Exchangeability)

이 챕터에서, 우리는 첫 번째로 합리적인 믿음 함수(belief function)이 어떤 특성을 가져야 하는지 그리고 이러한 특성들을 가지는 확률들을 보여줄 것입니다. 그리고 이산형, 연속형 확률 변수들과 확률 분포의 기본적인 메카니즘에 대해 복습 해볼것입니다. 마지막으로 우리는 독립성과 교환가능성 사이의 연결고리에 대해 알아볼 것입니다.

### 2.1 믿음 함수들과 확률들

지난 챕터를 시작하면서, 우리는 확률을 비율에 대한 믿음을 수적으로(numerically) 표현하는 방법이라고 정의했습니다. 여기에서 이 정의를 증명하진 않을 것입니다(자세한 내용은 Jaynes(2003)의 챕터2, 또는 Savage(1972)의 챕터 2, 3을 보세요.). 그러나 우리가 수적인 믿음들이 가지길 원하는 여러가지 특성들이 또한 확률의 특성이란 것을 보여줄 것입니다.

### *믿음 함수(Belief Functions)*

$\text{F}, \text{G}, \text{H}$를 세계에 대한 서로 겹칠 수도 있는 세 가지 상태라고 합시다. 예를 들어 다음과 같습니다.

$$
\text{F} = \{ \text{a person votes for a left-of-center candidate}\} \\ 
\text{G} = \{ \text{a person's income is in the lowest 10% of the population}\} \\
\text{H} = \{ \text{a person lives in a large city}\}
$$

$\text{Be}()$를 *믿음 함수*라고 합시다. 이것은 상태들에 믿음의 정도가 클 수록 더 큰 수를 할당하는 함수입니다. 몇몇 철학자들은 이 개념을 베팅의 선호도에 대한 믿음과 연결지어 더욱 단단하게 만들려고 시도해왔습니다. 

- $\text{Be}(\text{F})$ > $\text{Be}(\text{G})$는 우리가 $\text{G}$가 참이라기 보단 $\text{F}$가 참이란 것에 베팅하기를 선호한다는 것을 의미합니다.
우리는 또한 $\text{Be}()$가 특정한 상황 하에서 우리의 믿음을 설명하기를 원합니다.
- $\text{Be}(\text{F}|\text{H})$ > $\text{Be}(\text{G}|\text{H})$는 만약 우리가 $\text{H}$가 참이란 것을 알 때, $\text{G}$가 참이라기 보단 $\text{F}$가 참이란 것에 베팅하길 선호한다는 것을 의미합니다.
- $\text{Be}(\text{F}|\text{G})$ > $\text{Be}(\text{F} | \text{H})$는 만약 우리가 $\text{F}$에 베팅하도록 강요받았을 때, $\text{H}$가 참인 상황 보다는 $\text{G}$가 참인 상황 하에서 $\text{F}$에 베팅하는 것을 선호한다는 것을 의미합니다.

### *믿음들의 공리*

"수적으로 우리의 믿음들을 표현하기 위한 어떠한 함수도 다음과 같은 특성들을 가져야 한다"는 것은 많은 사람들에게 논의되어왔습니다.

$\textbf{B1}$ $\text{Be}(\text{not H} | \text{H}) \leq \text{Be}(\text{F}|\text{H}) \leq \text{Be}(\text{H}|\text{H})$ 

$\textbf{B2}$ $\text{Be}(\text{F or G} | \text{H}) \geq \text{max}\{\text{Be}(\text{F} | \text{H}) , \text{Be}(\text{G}|\text{H})\}$

$\textbf{B3}$ $\text{Be}(\text{F and G} | \text{H})$는 $\text{Be}(\text{G}|\text{H})$와 $\text{Be}(\text{F}|\text{G and H})$로부터 계산될 수 있다 

이러한 특성들을 어떻게 해석해야 할까요? 이들이 합리적인가요?

$\textbf{B1}$은 $\text{H}$가 주어졌을 때 $\text{F}$의 conditional belief인 $\text{Be}(\text{F}|\text{H})$에 우리가 할당한 숫자가 완벽한 불신($\text{Be}(\text{not H} | \text{H})$)에 대해 아래로 유계이고, 완벽한 믿음($\text{Be}(\text{H} | \text{H})$)에 대해 위로 유계라는 것을 의미합니다.

$\textbf{B2}$는 주어진 확률들의 집합 안에 진실이 놓여있다는 우리의 믿음이 확률들의 집합에 더해질 수록 더 줄어들어서는 안된다는 것을 의미합니다.

$\textbf{B3}$은 조금 어렵습니다. 왜 이것이 맞는지를 보기 위해 $\text{H}$가 참이라는 것을 알 때, 당신이 $\text{F}$가 참인지 $\text{G}$가 참인지를 정해야 한다고 상상해봅시다. 처음에 $\text{H}$가 주어졌을 때 $\text{G}$가 참인지 아닌지를 판단하고, 만약 그렇다면 $\text{G}$와 $\text{H}$가 주어졌을 때 $\text{F}$가 참인지 아닌지를 결정함으로써 이것을 구할 수 있습니다.

### *확률의 공리*

이제 $\textbf{B1, B2, B3}$을 기본적인 확률의 공리와 비교해보도록 하겠습니다. $\text{F} \cup \text{G}$는 "$\text{F or G}$"를, $\text{F} \cap \text{G}$는 "$\text{F and G}$"를, 그리고 $\emptyset$은 공집합을 의미한다는 것을 기억해봅시다.

$\textbf{P1}$ $0 = \text{Pr}(\text{not H} | \text{H}) \leq \text{Pr}(\text{F}|\text{H}) \leq \text{Pr}(\text{H}|\text{H}) = 1$

$\textbf{P2}$ $\text{Pr}(\text{F} \cup \text{G} | \text{H}) = \text{Pr}(\text{F} | \text{H}) + \text{Pr}(\text{G} | \text{H}) \ \  \text{if F} \cap \text{G} = \emptyset$

$\textbf{P3}$ $\text{Pr}(\text{F} \cap \text{G} | \text{H}) = \text{Pr}(\text{G} | \text{H})\text{Pr}(\text{F}|\text{G} \cap \text{H})$

당신은 한 확률 함수가 $\textbf{P1, P2, P3}$을 만족한다면 $\textbf{B1, B2, B3}$또한 만족한다는 것을 스스로 알 수 있을 것입니다. 즉 만일 우리가 우리의 믿음을 표현하기 위해 확률 함수를 사용한다면, 우리는 믿음의 공리를 만족시킨 것입니다.

### 2.2 사건, 분할 그리고 베이즈 법칙

**정의 1 (분할)** *집합들의 모음 $\{ \text{H}_{1}, ... , \text{H}_{k} \}$는 만약 다음과 같다면 또 다른 집합 $\mathcal{H}$의 분할(partition)입니다.*

1. 사건들은 연결되어있지 않다(이것을 $H_i \cap H_j = \emptyset \ for \ i \neq j ;$라고 씁니다)

2. 그 집합들의 합집합은 $\mathcal{H}$이다.(이것을 $\cup^K_{k=1} H_k = \mathcal{H}$라고 씁니다.)

어떤 여러 statement들이 참인지를 밝혀내는 과정 속에서, 만일 $\mathcal{H}$가 모든 가능한 참값들의 집합이고 $\{H_1, ..., H_K\}$가 $\mathcal{H}$의 분할이라면, 명확하게 $\{H_1, ..., H_K\}$ 중 하나는 참값을 포함하고 있을 것입니다.

### *Examples*

* $\mathcal{H}$를 어떤 사람의 종교적 기반이라고 합시다. 분할들은 다음과 같습니다.

    - {청교도, 카톨릭, 유대교, 기타, 무교}
    - {기독교, 비기독교}
    - {무신론자, 유일신론자, 다신론자}

* $\mathcal{H}$를 어떤 사람의 자식의 수라고 합시다. 분할들은 다음과 같습니다.
    - {0, 1, 2, 3 이상};
    - {0, 1, 2, 3, 4, 5, 6, ... }.
    
* $\mathcal{H}$를 주어진 모집단에서의 흡연과 고혈압 사이의 관계라고 합시다. 분할들은 다음과 같습니다.
    - {관계가 있다, 관계가 없다};
    - {음의 상관관계, 0의 상관관계, 양의 상관관계}

### *분할들과 확률*

$\{H_1, ..., H_K\}$를 $\mathcal{H}$의 분할이라고 가정합시다. $\text{Pr}(\mathcal{H}) = 1$이고 $E$는 특정한 사건입니다. 확률의 공리는 다음과 같은 것을 의미합니다:

### Rule of total probability : $\Sigma^K_{k=1} \text{Pr}(H_k) = 1$

### Rule of marginal probability : 

$\text{Pr}(E) = \Sigma^K_{k=1} \text{Pr}(E \cap H_k) = \Sigma^K_{k=1} \text{Pr} (E | H_k) Pr(H_k) $

### Bayes' rule : $\text{Pr}(H_j|E) = \frac{\text{Pr}(E|H_j) \text{Pr}(H_j) }{\text{Pr}(E)} = \frac{\text{Pr}(E|H_j) \text{Pr}(H_j)}{\Sigma^K_{k=1} \text{Pr}(E|H_k) \text{Pr}(H_k)}$ 

### *Example*

1996년 사회 총 조사의 한 부분은 30살 이상 남성의 교육 수준과 소득에 대한 표본 데이터를 가지고 있습니다. $\{H_1, H_2, H_3, H_4\}$를 이 표본에서 무작위로 선택된 사람들이라고 합시다. 그리고 각각에는 소득의 하위 25%, 차하위 25%, 차상위 25%, 상위 25%가 있다고 합시다. 정의에 따라 다음과 같이 적을 수 있습니다.

$$
\{Pr(H_1), Pr(H_2), Pr(H_3), Pr(H_4) \} = \{.25, .25, .25, .25\}
$$

$\{H_1, H_2, H_3, H_4 \}$가 분할이고 이러한 확률들의 합은 1이라는 점에 주목합시다. $\text{E}$가 설문조사에서 무작위로 선택된 사람이 대학 교육을 받았을 사건이라고 합시다. 설문조사 결과 다음과 같은 데이터가 나왔습니다.

$$
\{ \text{Pr}(E | H_1), \text{Pr}(E|H_2), \text{Pr}(E|H_3), \text{Pr}(E|H_4) \} = \{.11, .19, .31, .53 \} .
$$

 이 확률들은 다 더한 값이 1이 아닙니다. 이것들은 네 개의 각기 다른 소득 분위 $H_1, H_2, H_3, H_4$에서 대학 학위를 가지고 있는 사람의 비율을 뜻합니다. 자 이제 대학 교육을 받은 사람들의 소득 분포를 생각해봅시다. 베이즈의 법칙을 사용하면 다음과 같은 값을 얻을 수 있습니다.

$$
\{ \text{Pr}(H_1 | E), \text{Pr}(H_2 | E), \text{Pr}(H_3 | E), \text{Pr}(H_4|E) \} = \{.09, .17, .27, .47 \},
$$

이 값에서 우리는 대학 교육을 받은 사람들의 소득 분포가 전체 모집단의 {.25, .25, .25, .25} 와는 상당히 다르다는 것을 알 수 있습니다. 이 확률들은 모두 더하면 1이라는 사실에 주목합시다. 이들은 E가 주어졌을 때 분할 안에서 벌어진 사건의 조건부 확률입니다. 

   베이지안 추론에서, $ \{H_1, ..., H_K \} $는 보통 연결되지 않은(disjoint) 가설이나 자연 상태를 의미합니다. 그리고 E는 설문조사나, 연구 또는 실험의 결과를 나타냅니다. 실험 후의 가설과 비교하기 위해 우리는 보통 다음과 같은 비(ratio)를 계산하여 사용합니다.

$$
\frac{\text{Pr}(H_i| E)}{\text{Pr}(H_j | E)} = \frac{\text{Pr}(E|H_i)\text{Pr}(H_i) / \text{Pr}(E)}{\text{Pr}(E|H_j)\text{Pr}(H_j) / \text{Pr}(E)} \\
$$
$$
 \ \ \ \ \ \ \ = \frac{\text{Pr}(E | H_i) \text{Pr}(H_i)}{\text{Pr}(E|H_j) \text{Pr}(H_j)}
$$
$$
\ \ \ \ \ \ \ \ \ \ \ \ \ = \frac{\text{Pr}(E|H_i)}{\text{Pr}(E|H_j)} \times \frac{\text{Pr}(H_i)}{\text{Pr}(H_j)}
$$
$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \text{"Bayes factor"} \times \text{"prior beliefs"}
$$

이 식은 베이즈의 법칙이 데이터를 관찰한 이후 우리의 믿음이 어때야 하는지를 결정하진 않는다는 것을 상기시켜줍니다. 이것은 단지 데이터를 본 뒤 그들이 어떻게 바뀌여야 하는지를 말해줄 뿐입니다. 

### *Example*

우리가 어떤 공공 기관장의 특정 후보를 지지하는 비율에 관심을 가지고 있다고 가정합시다. 그리고 다음과 같은 상황들이 있습니다.

$$
\mathcal{H} = \{ \text{A후보를 지지하는 비율의 모든 경우의 수} \} ; \\
H_1 = \{\text{ 절반 초과의 유권자들이 A후보를 지지함 } \} ; \\
H_2 = \{ \text{절반 이하의 유권자들이 A후보를 지지함} \} ; \\
E = \{ \text{100명 중 54명이 설문조사에서 A후보를 지지한다고 말함 }\}.
$$

그렇다면 $\{ H_1, H_2 \}$ 가 $\mathcal{H}$의 분할입니다. 이 때 관심있는 것은 $\text{Pr}(H_1|E)$ 또는 $\text{Pr}(H_1|E)/\text{Pr}(H_2 | E) $입니다. 우리는 이러한 값들을 어떻게 구하는지 다음 챕터에서 배우도록 할 것입니다.

### 2.3 독립성(Independence)

**정의 2(독립성)** 두 사건 F와 G는 만약 $\text{Pr}(F \cap G | H ) = \text{Pr}(F|H)\text{Pr}(G|H) $ 라면 H가 주어졌을 때 조건부 독립이라고 합니다.

조건부 확률을 어떻게 해석할 수 있을까요? 공리 $\textbf{P3}$에 의하면, 다음은 항상 참입니다.

$$
\text{Pr} (F \cap G | H) = \text{Pr}(G|H) \text{Pr}(F | H \cap G) .
$$

만약 F와 G가 H가 주어졌을 때 조건부 독립이라면, 다음은 항상 참입니다.

$$
\text{Pr}(G|H) \text{Pr}(F | H \cap G) =^{\text{always}} \text{Pr}(F \cap G|H) =^{\text{independence}} \text{Pr}(F|H) \text{Pr}(G|H) \\
\text{Pr}(G|H)\text{Pr}(F|H \cap G) \ \ \ = \ \ \ \text{Pr}(F|H)\text{Pr}(G|H) \\
\text{Pr}(F|H \cap G) \ \ \ = \ \ \ \text{Pr}(F|H).
$$

조건부 독립은 따라서 $\text{Pr}(F|H \cap G) = \text{Pr}(F|H)$를 의미합니다. 다른 말로 하자면, 만일 우리가 H가 참이라는 것과 F와 G가 H가 주어졌을 때 조건부 독립이라는 것을 안다면, G를 아는 것은 F에 대한 우리의 믿음을 바꾸지 못합니다.

### *예시*

 

다음 두 가지 상황에서 H가 참이라고 가정했을 때, F와 G가 조건부 독립이라고 가정합시다.

F = { 한 환자가 흡연자이다. }

G = { 한 환자는 폐암 환자이다. }

H = { 흡연은 폐암을 유발한다. }

 

F = { 당신은 하트의 잭을 생각하는중이다 }

G = { 독심술사가 당신이 하트의 잭을 생각중이라고 말한다 }

H = { 그 독심술사는 초감각적인 통찰력을 가지고 있다 }

이러한 상황 둘 모두에서, H가 참이라는 것은 F와 G 사이의 관계를 의미합니다. 그렇다면 H가 사실이 아닐 때는 어떨까요?

### 2.4 확률 변수

베이지안 추론에서 확률변수는 우리가 확률의 상태에 관해서 알려지지 않은 수적인 양(numerical quantity)로 정의됩니다. 예를 들어 한 설문조사, 실험, 연구에서의 정량적인 산출값은 연구가 수행되기 전까지는 확률변수입니다. 추가적으로 고정되어 있지만 알려지지 않은 모집단의 파라미터 또한 확률변수입니다.

### **2.4.1 이산 확률 변수**

Y를 확률 변수라고 하고 $\mathcal{Y}$를 Y가 가질 수 있는 모든 가능한 값들의 집합이라고 합시다. 우리는 가능한 산출물들의 집합을 셀 수 있을(countable) 때 Y를 이산 확률 변수라고 합니다. 즉 $\mathcal{Y}$가 $\mathcal{Y} = \{y_1, y_2, ... \}$와 같이 표현될 수 있다는 것을 의미합니다. 

*Examples*

* Y = 모집단에서 뽑은 무작위 샘플에서 교회에 가는 사람의 수
* Y = 무작위로 뽑힌 사람의 자녀의 수
* Y = 무작위로 뽑힌 사람이 교육 받은 연수

*확률 분포와 확률 밀도함수*

우리의 설문조사의 결과 Y가 y라는 값을 가지는 사건은 {Y = y}와 같이 표현됩니다. 각각의 y $\in \mathcal{Y}$에게, $\text{Pr}(Y = y)$를 간단하게 쓰면 $p(y)$와 같이 쓸 수 있습니다. 이 y의 함수를 Y에 대한 확률 밀도 함수(*probability density function*, pdf)라고 부릅니다. 그리고 이것은 다음과 같은 특성을 가집니다.

1. 모든 $y \in \mathcal{Y}$에 대해 $0 \leq p(y) \leq 1$
2. $\Sigma_{y \in \mathcal{Y}}p(y)$ = 1.

일반적으로 Y에 대한 확률의 상태는 pdf로 부터 구할 수 있습니다. 예를 들어, $\text{Pr}(Y \in A) = \Sigma_{y \in A} p(y)$입니다. 만약 A와 B가 서로 연결되지 않은 $\mathcal{Y}$의 부분집합이라면 다음과 같은 식이 성립합니다.

$$
\text{Pr}( Y \in A \text{또는} Y \in B) \equiv \text{Pr}(Y \in A \cup B) = \text{Pr}(Y \in A) + \text{Pr}(Y \in B) \\
= \Sigma_{y \in A}p(y) + \Sigma_{y \in B}p(y)
$$

*Example : Binomial distribution*

몇몇 양의 정수 n에 대해 $\mathcal{Y} = {0,1,2,...,n}$라고 합시다. 불확실한 수량 Y $\in \mathcal{Y}$는 만약 다음과 같다면 *확률 $\theta$의 binomial distribution*라고 합니다.

$$
\text{Pr}(Y = y | \theta) = \text{dbinom}(y,n,\theta) = \binom n y \theta^y(1-\theta)^{n-y}.
$$

예를 들어 만약 $\theta = .25$이고 $n = 4$라면 다음과 같은 값을 계산할 수 있습니다.

$$
\text{Pr}(Y = 0| \theta = .25) = \binom 4 0 (.25)^0(.75)^4 = .316
$$

$$
\text{Pr}(Y = 1| \theta = .25) = \binom 4 1 (.25)^1(.75)^3 = .422
$$

$$
\text{Pr}(Y = 2| \theta = .25) = \binom 4 2 (.25)^2(.75)^2 = .211
$$

$$
\text{Pr}(Y = 3| \theta = .25) = \binom 4 3 (.25)^3(.75)^1 = .047
$$

$$
\text{Pr}(Y = 4| \theta = .25) = \binom 4 4 (.25)^4(.75)^0 = .004.
$$

*Example : 포아송 분포*

$\mathcal{Y} = {0,1,2,...}$라고 합시다. 불확실한 수량 Y $\in \mathcal{Y}$는 만약 다음과 같다면 *평균이 $\theta$인 포아송 분포*를 가진다고 합니다.

$$
\text{Pr}(Y = y|\theta) = dpois(y, \theta) = \theta^ye^{-\theta} / y!.
$$

예를 들어, 만약 $\theta = 2.1$이라면 다음과 같습니다. (2006년의 미국 출산율 데이터),

$$
\text{Pr}(Y = 0 | \theta = 2.1) = (2.1)^0e^{-2.1}/(0!) = .12
$$

$$
\text{Pr}(Y = 0 | \theta = 2.1) = (2.1)^0e^{-2.1}/(0!) = .12
$$

$$
\text{Pr}(Y = 0 | \theta = 2.1) = (2.1)^0e^{-2.1}/(0!) = .12 
$$

$$
\text{Pr}(Y = 0 | \theta = 2.1) = (2.1)^0e^{-2.1}/(0!) = .12
$$

$$
\vdots \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \vdots \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \vdots
$$


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import poisson

# Panel 1

ax1 = plt.subplot(121)

x1 = np.arange(0,11,1)

y1 = poisson(2.1).pmf(x)

ax1.bar(x1, y1, width = 0.1)
plt.ylabel(r'$p(y|\theta = 2.1)$')
plt.xlabel('y')
plt.ylim(-0.01,0.3)


# Panel 2

ax2 = plt.subplot(122)

x2 = np.arange(0,101,1)

y2 = poisson(21).pmf(x2)

ax2.bar(x2, y2, width = 0.6)
plt.ylabel(r'$p(y|\theta = 21)$')
plt.xlabel('y')
plt.ylim(-0.001, 0.09)

plt.show();
```




![output_67_0](https://user-images.githubusercontent.com/57588650/109639541-a7777e80-7b92-11eb-8364-2db244dc2d09.png)
    


$$
\textbf{그림. 2.1.} \ \ \  \text{평균이 2.1, 21일 때의 포아송 분포}
$$

### *2.4.2 연속 확률 변수*

표본 공간 $\mathcal{Y}$가 러프하게 모든 실수의 집합인 $\mathbb{R}$과 같다고 가정해봅시다. 총합이란 개념이 성립하지 않기 때문에 $\text{Pr}(Y \leq 5)$는 $\Sigma_{y \leq 5}p(y)$와 같다고 말할 수 없습니다(5보다 작거나 같은 실수의 집합은 "셀 수 없기(uncountable)"때문이죠.). 그래서 pdf $p(y)$의 방식으로 사건의 확률을 계산하는 대신에 수리 통계학 강의에서는 다음과 같은 cdf(*cumulative distribution function*)의 방식으로 확률 변수의 확률 분포를 정의합니다:

$$
F(y) = \text{Pr}(Y \leq y)
$$

$F(\infty) = 1, F(-\infty) = 0$이고 만약 $b < a$라면 $F(b) \leq F(a)$라는 것을 기억하세요. 다양한 사건들의 확률을 cdf를 통해 얻을 수 있습니다. 

* $\text{Pr}( Y > a) = 1 - F(a)$
* $\text{Pr}( a < Y \leq b) = F(b) - F(a)$

만약 $F$가 연속형이라면 (다른 말로 하면 어떠한 점프하는 구간이 없다면), 우리는 $Y$가 연속 확률 변수라고 말합니다. 수학의 theorem에 따르면 모든 연속형 cdf $F$에 대해 다음과 같은 양의 함수 $p(y)$가 존재합니다.

$$
F(a) = \int_{- \infty}^{a} p(y) dy.
$$

이 함수를 $Y$의 확률 밀도 함수라고 부릅니다. 그리고 이것의 특성은 이산 확률 변수에서 pdf의 특성과 비슷합니다.

1. 모든 $y \in \mathcal{Y}$에 대해 $0 \leq p(y)$ 
2. $\int_{y \in \mathbb{R}} p(y) dy = 1.$

이산형일 경우과 마찬가지로, $Y$에 대한 확률 상태(probability statement)는 pdf를 통해 구할 수 있습니다: $\text{Pr}(Y \in A) = \int_{y \in A} p(y) dy$ 이고, 만약 $A$와 $B$가 $\mathcal{Y}$의 연결되지 않은 부분집합이라면 다음이 성립합니다.

$$
\mathcal{Pr}( Y \in A \ or \ Y \in B) \equiv \text{Pr}(A \in A \cup B) = \text{Pr}(Y \in A) + \text{Pr}(Y \in B)
$$

$$
= \int_{y \in A} p(y) dy + \int_{y \in B} p(y) dy.
$$

이 성질들을 이산형 케이스에서의 비슷한 성질들과 비교해보면, 연속형 분포의 적분이 이산형 분포에서의 합계와 비슷하게 작용한다는 것을 볼 수 있습니다. 사실 적분은 표본 공간이 셀 수 없는 상황에서 합계의 일반화된 버전이라고 생각할 수 있습니다. 그러나 이산형 케이스에서의 pdf와는 다르게 연속 확률 변수에서의 pdf는 반드시 1보다 작을 필요는 없고, $p(y)$는 "$Y = y$일 확률"이 아닙니다. 그러나 만약 $p(y_1) > p(y_2)$라면 비공식적으로 $y_1$이 $y_2$보다 "더 높은 확률을 가졌다" 라고 할 수 있습니다.

*Example : 정규 분포*

우리가 $\mathcal{Y} = ( - \infty, \infty)$위에 있는 모집단에서 표본을 추출하고, 모집단의 평균이 $\mu$, 분산은 $\sigma^2$라는 것을 안다고 가정합시다. 평균이 $\mu$이고 분산이 $\sigma^2$인 모든 확률 분포 중, 가장 "넓게 퍼져있고" 또는 엔트로피라고 불리는 지표의 방식으로 말하면 "diffuse"인 것은 정규분포(normal($\mu, \sigma^2$))이고, 다음과 같은 cdf를 가집니다.

$$
\text{Pr}( Y \leq y | \mu, \sigma^2) = F(y) 
$$

$$
= \int_{- \infty}^{y} \frac{1}{\sqrt{2 \pi \sigma}} exp  \left(  - \frac{1}{2} \left( \frac{y - \mu}{\sigma} \right) ^2  \right) dy
$$

즉,

$$
p(y|\mu, \sigma^2) = dnorm(y,\mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma}} exp \left( - \frac{1}{2} \left(\frac{y - \mu}{\sigma} \right)^2 \right).
$$

입니다.

$\mu = 10.75, \sigma = .8(\sigma^2 = .64)$라고 하면, **그림 2.2.** 와 같은 cdf와 확률밀도함수가 나옵니다. 이 평균과 표준편차는 $e^Y$의 중위값을 약 46,630으로 만듭니다. 이는 실제 2005년 미국 가구 소득의 중위수와 비슷하죠. 추가적으로, $\text{Pr}(e^Y > 100000) = \text{Pr}( Y > \text{log}100000) = 0.17$은 러프하게 2005년의 가구 소득이 100,000$가 넘는 가구의 비율과 매치됩니다.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import norm

# Panel 1

ax1 = plt.subplot(121)

x1 = np.linspace(8,14,100)

y1 = norm(10.75, 0.8).cdf(x1)

ax1.plot(x1, y1)
ax1.axhline(1, color = 'red', linewidth = 0.3, linestyle = '--')
ax1.axhline(0.5, color = 'red', linewidth = 0.3, linestyle = '--')
ax1.axhline(0, color = 'red', linewidth = 0.3, linestyle = '--')
plt.ylabel(r'$F(y)$')
plt.xlabel('y')



# Panel 2

ax2 = plt.subplot(122)

x2 = np.linspace(8,14,100)

y2 = norm(10.75, 0.8).pdf(x2)

ax2.plot(x2, y2)
ax2.axvline(10.75, linewidth = 0.3, color = 'red', linestyle = '--')
plt.ylabel(r'$p(y)$')
plt.xlabel('y')

plt.show();
```


    
![output_82_0](https://user-images.githubusercontent.com/57588650/109639546-a8a8ab80-7b92-11eb-8e42-10c495009f2d.png)
    


$$
\textbf{그림. 2.2.} \text{평균이 10.75이고 표준편차가 0.8인 정규분포}
$$

### **2.4.3. 분포들의 표현**

미지의 수량 Y에 대한 평균 또는 기댓값은 다음과 같습니다.

$$
E[Y] = \Sigma_{y \in \mathcal{Y}} \ \ \text{if Y is discrete ;} 
$$

$$
E[Y] = \int_{y \in \mathcal{Y}} y p(y) \ dy \ \ \text{if Y is continuous.}
$$

평균이란 분포의 덩어리의 중앙입니다. 그러나 보통은 다음의 둘 각각과 같지 않습니다.

$$
\text{ 최빈값(mode) : "가장 높은 확률을 가지는 Y의 값" 또는}\\
\text{ 중위수(median) : "분포의 가운데에 위치한 Y의 값}
$$

특히, 한 쪽으로 치우친 분포에 대해서는(소득 분포와 같은) 평균은 "일반적인" 표본의 값과는 동떨어져있을 수 있습니다. 예를 들어 그림 2.3.을 봐봅시다. 여전히 평균히 아주 유명한 분포의 위치의 표현이긴 합니다. 평균에 대한 몇몇 보고서와 연구에서의 정의는 다음을 포함하고 있습니다.

1. $\{Y_1, ..., Y_n\}$의 평균은 전체 값을 비율로 만든 버전이고, 전체 값은 보통 관심있는 것의 수량입니다.
2. 당신이 Y의 값이 무었일지 예상하라고 압박받는다고 가정합시다. 그리고 당신은 $(Y - y_{guess})^2$의 값 만큼 패널티를 받습니다. 그렇다면 Y의 예측값을 $E[Y]$로 하는 것이 당신의 예상 패널티를 가장 작게 할 수 있습니다.
3. 지금부터 우리가 짧게 볼 몇몇의 간단한 모델에서는, 표본 평균이 데이터에서 얻어질 수 있는 모집단에 대한 모든 정보를 담고 있습니다.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import norm, lognorm

# Panel 1

ax1 = plt.subplot(121)

x1 = np.linspace(8,14,200)

y1 = norm(10.75, 0.8).pdf(x1)

ax1.plot(x1, y1)
ax1.axvline(10.75, linewidth = 0.3, color = 'red', linestyle = '--')
plt.ylabel(r'$p(y)$')
plt.xlabel('y')

# Panel 2

ax2 = plt.subplot(122)

x2 = np.linspace(0,300000,100)

y2 = lognorm.pdf(x2, s = 0.8, scale = np.exp(10.75)) * 10 ** 5

ax2.plot(x2, y2)
mean = lognorm.stats(s = 0.8, scale = np.exp(10.75), moments = 'm')
median = np.exp(10.75)
mode = np.exp(10.75 - 0.8**2)
ax2.axvline(mean, linewidth = 0.3, color = 'red', linestyle = '--')
ax2.axvline(median, linewidth = 0.3, color = 'green', linestyle = '--')
ax2.axvline(mode, linewidth = 0.3, color = 'black', linestyle = '--')
plt.ylabel(r'$10^5p(y)$')
plt.xlabel('y')
plt.text(mean, 1.5, 'mean', color = 'red')
plt.text(median, 1.55, 'median', color = 'green')
plt.text(mode, 1.6, 'mode', color = 'black')
plt.xticks([0, 50000, 150000, 250000])

plt.show();
```


    
![output_88_0](https://user-images.githubusercontent.com/57588650/109639551-a9414200-7b92-11eb-9ee7-3a2d89a741f7.png)
    


$$
\textbf{그림. 2.3.} \text{파라미터가} \mu = 10.75, \sigma = 0.8 \\
\text{인 normal distribution과 lognormal distribution의 최빈값, 중위수, 평균}
$$

분포의 위치에 더해 우리가 자주 관심있는 값은 그것이 얼마나 퍼져있냐 입니다. 가장 유명한 퍼진 정도의 척도는 분포의 "분산(variance)"입니다.

$$
\text{Var}[Y] = \text{E}[(Y - E[Y])^2] \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = E[Y^2 - 2YE[Y] + E[Y]^2] \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = E[Y^2] - 2E[Y]^2 + E[Y]^2 \\
\ \ \ \ \ \ \ \ \ \ \  \ \ = E[Y^2] - E[Y]^2
$$

분산은 표본의 값 $Y$와 모평균 $E[Y]$사이 거리의 제곱의 평균입니다. 표준 편차는 분산의 제곱근이고, $Y$와 같은 스케일입니다.

 퍼진 정도의 또 다른 척도는 분위수(quantiles)를 사용하는 것입니다. 연속이고 단조 증가하는 cdf F에 대해 $F(y_a) \equiv Pr(Y \leq y_{\alpha}) = \alpha$인 값 $y_{\alpha}$를 $\alpha$-quantile이라고 합니다. 분포의 *interquantile range*는 구간 $(y_{.25}, y_{.75})$이고, 이것은 분포의 50%를 포함하고 있습니다. 비슷하게 구간 ($y_{.025}, y_{.975}$)은 분포의 95%를 포함하고 있습니다.

### *2.5 결합 분포(Joint distribution)*

### *이산 분포*

다음을 정의하겠습니다.

* $\mathcal{Y}_1, \mathcal{Y}_2$ 는 두 개의 셀 수 있는(countable) 표본 공간입니다.;
* $Y_1, Y_2$는 각각 $\mathcal{Y}_1, \mathcal{Y}_2$안에서 값을 가지는 두 개의 확률 변수입니다.

$Y_1$과 $Y_2$에 대한 결합된 믿음은 확률로 나타낼 수 있습니다. 예를 들어 부분집합들인 $A \subset \mathcal{Y}_1$과 $B \subset \mathcal{Y}_2$에 대해 $Y_1$이 A 안에 있고, $Y_2$가 B 안에 있을 때의 우리의 믿음을 $\text{Pr}( \{ Y_1 \in A \} \cap \{ Y_2 \in B \})$로 나타낼 수 있습니다. $Y_1$과 $Y_2$의 결합 pdf 또는 결합 밀도는 다음과 같이 정의됩니다.

$$
p_{Y_1 Y_2}(y_1,y_2) = Pr( \{Y_1 = y_1 \} \cup \{Y2 = y_2 \}), \ for \ y_1 \in \mathcal{Y_1}, \ y_2 \in \mathcal{Y_2}
$$

$Y_1$의 주변 밀도(*marginal density*)는 결합 밀도에서 계산될 수 있습니다.

$$
p_{Y_1}(y_1) \equiv Pr(Y_1 = y_1) \\
$$

$$
= \sum_{y_2 \in \mathcal{Y_2}} Pr( \{ Y_1 = y_1 \} \cap \{ Y_2 = y_2 \} ) 
$$

$$
\equiv \sum_{y_2 \in \mathcal{Y_2}} p_{Y_1 Y_2}(y_1, y_2)
$$

{$Y_1 = y_1$}이 주어졌을 때 $Y_2$의 조건부 밀도(*conditional density*)는 결합 밀도와 주변 밀도를 사용해 다음과 같이 계산할 수 있습니다:

$$
p_{Y_2 | Y_1}(y_2 | y_1) = \frac{Pr ( \{Y_1 = y_1 \} \cap \{Y_2 = y_2 \}) }{Pr(Y_1 = y_1)} \\
= \frac{p_{Y_1 Y_2}(y_1, y_2)}{p_{Y_1}(y_1)}
$$

또한 다음과 같은 것을 스스로 구해볼 수 있을 것입니다.

{$p_{Y_1}, p_{Y_2 | Y_1}$}은 $p_{Y_1 Y_2}$를 통해 구할 수 있습니다.

{$p_{Y_2}, p_{Y_1 | Y_2}$}은 $p_{Y_1 Y_2}$를 통해 구할 수 있습니다.

$p_{Y_1 Y_2}$은 {$p_{Y_1}, p_{Y_1 | Y_1}$}를 통해 구할 수 있습니다.

$p_{Y_1 Y_2}$은 {$p_{Y_2}, p_{Y_1 | Y_2}$}를 통해 구할 수 있습니다.

그러나

$p_{Y_1 Y_2}$은 {$p_{Y_1}, p_{Y_2}$}로부터 구할 수 없습니다.

밀도 함수의 아랫첨자는 밀도 함수의 타입이 함수의 매개변수(argument)에 의해 정해지는 케이스에서는 보통 쓰지 않습니다. 즉 $p(y_1)$은 $p_{Y_1}(y_1)$을, $p(y_1, y_2)$는 $p_{Y_1 Y_2}(y_1, y_2)$를, $p(y_1 | y_2)$는 $p_{Y_1 | Y_2}(y_1 | y_2)$를 나타냅니다.

*예시 : 사회 이동*

로건 (1983)은 다음과 같은 아빠와 아들의 직업 분류의 결합 분포를 발표했습니다.

<img width="551" alt="스크린샷 2021-02-24 오후 8 43 48" src="https://user-images.githubusercontent.com/57588650/108996044-43ac0c00-76e1-11eb-9943-83a18f14ee99.png">

우리가 이 모집단에서 하나의 아빠-아들 쌍을 추출한다고 가정합시다. $Y_1$을 아빠의 직업이라고 하고, $Y_2$를 아들의 직업이라고 하면 다음과 같은 값을 계산할 수 있습니다.

$$
Pr(Y_2 = \text{professional} | Y_1 = \text{farm})
= \frac{Pr(Y_2 = \text{professional} \cap Y_1 = \text{farm})}{Pr(Y_1 = \text{farm})} \\
$$

$$
= \frac{.018}{.018 + .035 + .031 + .008 + .018} \\ 
$$

$$
= .164.
$$

### *연속 결합 분포*

만약 $Y_1, Y_2$가 연속이면 우리는 cdf(cumulative distribution function)을 사용합니다. 연속 결합 cdf $F_{Y_1 Y_2}(a,b) \equiv Pr(\{Y_1 \leq a \} \cap \{Y_2 \leq b \})$이 주어졌을 때, 다음과 같은 함수 $p_{Y_1 Y_2}$가 존재합니다.

$$
F_{Y_1 Y_2} (a,b) = \int_{-\infty}^a \int_{-\infty}^b p_{Y_1 Y_2} (y_1, y_2) dy_1y_2 .
$$

함수 $p_{Y_1 Y_2}$는 $Y_1, Y_2$의 결합 밀도 함수입니다. 이산형 케이스일 때와 같이, 우리는 다음을 구할 수 있습니다.

* $p_{Y_1}(y_1) = \int^{\infty}_{-\infty}(y_2, y_2) dy_2$;
* $p_{Y_2 | Y_2}(y_2|y_1) = p_{Y_1 Y_2} (y_1, y_2) / p_{Y_1}(y_1)$.

당신은 $p_{Y_2 | Y_1}(y_2|y_1)$이 실제 확률 밀도라는 것에 스스로 확신을 가져야 합니다. 예를 들어 각각의 $y_1$의 값에 대해 $p_{Y_2 | Y_1}(y_2|y_1)$는 $Y_2$의 확률 밀도입니다.

### *연속형과 이산형 변수가 결합된 경우*

$Y_1$이 이산형이고 $Y_2$가 연속형이라고 합시다. 예를 들어 $Y_1$은 직업의 범주이고 $Y_2$는 개인 소득입니다. 다음과 같은 것을 정의한다고 가정합시다.

* 우리의 믿음 $Pr(Y_1 = y_1)$으로 부터의 $p_{Y_1}$의 주변밀도
* $Pr(Y_2 \leq y_2|Y_1 = y_1) \equiv F_{Y_2 | Y_1}(y_2|y_1)$으로 부터의 위와 같은 조건부 밀도 $p_{Y_2|Y_1}(y_2|y_1)$

$Y_1, Y_2$의 조건부 밀도는 그렇다면 

$$
p_{Y_1 Y_2}(y_1,y_2) = p_{Y_1}(y_1) \times p_{Y_2|Y_1}(y_2|y_1),
$$

이고 다음과 같은 성질을 가집니다.

$$
Pr(Y_1 \in A, Y_2 \in B) = \int_{y_2 \in B} \bigg \{ \sum_{y_1 \in A} p_{Y_1 Y_2}(y_1,y_2) \bigg \} dy_2
$$

### *베이즈의 법칙과 파라미터 추정*

다음을 정의합시다.

$\theta$ = 커다란 모집단에서 특정한 성격을 가지고 있는 사람의 비율

$Y$ = 작은 랜덤 샘플에서 그 성격을 가지고 있는 사람의 수

그러면 우리는 $\theta$는 연속형으로 $Y$는 이산형으로 취급해야 합니다. $\theta$의 베이지안 추정량은 $y$가 $Y$의 관찰된 값일 때 $p(\theta | y)$를 계산함으로써 구할 수 있습니다. 이 계산이 처음으로 필요로 하는 것은 $\theta$에 대해 가지고 있는 우리의 믿음과 설문조사의 결과 $Y$를 표현하는 결합 밀도 $p(y, \theta)$입니다. 보통 이 결합 밀도를 다음으로 부터 만드는 것이 자연스럽습니다.

* $p(\theta)$, $\theta$에 대한 믿음
* $p(y|\theta)$, $\theta$의 각각의 값 $Y$에 대한 믿음

$\{Y = y \}$를 관찰했을 때, 우리는 $\theta$에 대한 업데이트된 믿음을 계산할 필요가 있습니다.

$$
p(\theta | y) = p(\theta, y) / p(y) = p(\theta)p(y|\theta) / p(y).
$$

이 조건부 밀도를 $\theta$의 **사후 밀도(posterior density)** 라고 부릅니다. $\theta_a$와 $\theta_b$이 $\theta$의 실제 값이 가질 수 있는 두 개의 가능한 수적인(numerical) 값이라고 가정합시다. $\theta_a$의 $Y = y$가 주어졌을 때 $\theta_b$에 상대적인 사후 확률(밀도)는 다음과 같습니다.

$$
\frac{p(\theta_a |y)}{p(\theta_b|y)} = \frac{p(\theta_a) p(y|\theta_a) / p(y)}{p(\theta_b)p(y|\theta_b) / p(y)} 
$$

$$
= \frac{p(\theta_a)p(y|\theta_a)}{p(\theta_b)p(y|\theta_b)}
$$

이것은 $\theta_a$와 $\theta_b$의 상대 사후 확률을 구하기 위해서 $p(y)$의 값을 구할 필요가 없다는 것을 의미합니다. 이것을 생각해내는 또 다른 방법은 다음과 같은 $\theta$에 대한 함수를 통해서입니다.

$$
p(\theta|y) \propto p(\theta) p(y|\theta).
$$

이 비례식에서의 상수는 $1/p(\theta)$이고, 다음으로 부터 계산될 수 있습니다.

$$
p(y) = \int_{\Theta} p(y, \theta) d\theta = \int_{\Theta}p(y|\theta)p(\theta)d\theta
$$

즉 다음과 같은 식을 구할 수 있습니다

$$
p(\theta|y) = \frac{p(\theta)p(y|\theta)}{\int_{\Theta}p(y|\theta)p(\theta)d\theta}
$$

다음 장에서 볼 수 있겠지만, 분자가 중요한 부분입니다.

### *2.6 독립 확률 변수(Independent random variables)*

$Y_1, ..., Y_n$이 확률변수이고 $\theta$가 확률변수가 생성된 조건을 기술하는 파라미터라고 가정합시다. 우리는 만일 모든 n개의 집합들 {$A_1, ..., A_n$}이 다음을 만족한다면 $Y_1, ..., Y_n$이 $\theta$가 주어졌을 때 조건부 독립이라고 부릅니다.

$$
Pr(Y_1 \in A_1, ..., Y_n \in A_n | \theta) = Pr(Y_1 \in A_1 | \theta) \times ... \times Pr(Y_n \in A_n|\theta).
$$

이 독립 확률 변수들의 정의가 우리가 이전에 다룬 각각의 {$Y_j \in A_j$}가 하나의 사건일 때 독립 사건의 정의에 기반한다는 것에 주목합시다. 이전에 했던 계산에서, 만약 독립 조건을 만족한다면 다음과 같습니다.

$$
Pr(Y_i \in A_i | \theta, Y_j \in A_j) = Pr(Y_i \in A_i | \theta)
$$

그래서 조건부 독립의 의미를 $Y_j$는 $\theta$를 알 때 더이상 $Y_i$에 대한 추가적인 정보를 줄 수 없다는 것으로 해석할 수 있습니다. 더 확장해보자면, 독립 조건 하에서 결합 밀도 함수는 다음과 같이 주변 밀도(marginal densities)들의 곱으로 나타낼 수 있습니다.

$$
p(y_1, ..., y_n | \theta) = p_{Y_1}(y_1 | \theta) \times ... \times p_{Y_n}(y_n | \theta) = \prod^n_{i=1}pY_i(y_i|\theta),
$$

Y_1, ..., Y_n이 같은 과정을 통해 비슷한 방법으로 생성됐다고 가정해봅시다. 예를 들어 그들은 모두 같은 모집단에서 나온 샘플들일 수 있고, 비슷한 조건 하에서 수행된 실험일 수도 있습니다. 이것은 주변 밀도(marginal density)들이 모두 다음과 같은 공통의 밀도와 같다고 제안합니다.

$$
p(y_1, ..., y_n|\theta) = \prod^n_{i=1}p(y_i|\theta).
$$

이 케이스에서, $Y_1, ..., Y_n$은 조건부 독립이고 동일하게 분포한다고(conditionally independent and identically distributied, i.i.d.) 말할 수 있습니다. 이것을 수식으로 짧게 나타내면 다음과 같습니다.

$$
Y_1, ..., Y_n | \theta \sim \text{i.i.d.} p(y|\theta).
$$
 

### *교환가능성(Exchangeability)*

*Example : 행복*

1998년의 사회 총 조사에서, 참가자들은 그들이 보통 행복한지 아닌지에 대한 질문을 받았습니다. $Y_i$는 이 질문과 관련된 확률 번수라고 합시다. 즉

$$
Y_i =  \begin{cases} 1 \ \ \text{참가자 i가 보통 행복하다고 답했을 때} \\ 0 \ \ \text{그 외} \end{cases}
$$

이 섹션에서 우리는 $Y_1, ..., Y_10$, 즉 무작위로 선택된 10명의 설문조사 참여자들의 답안에 대한 결합 믿음(joint beliefs)의 구조를 생각해볼것입니다. 이전과 같이, $p(y_1, ..., y_10)$이 각각의 $y_i$가 0 또는 1 둘 중 하나일 때, $Pr(Y_1 = y_1, ..., Y_{10} = y_{10})$의 간단한 버전의 노테이션이라고 합시다.

*교환가능성(Exchangeability)*

우리가 세 개의 각각 다른 결과에 확률을 할당하라는 요청을 받았다고 가정합시다.

$$
p(1,0,0,1,0,1,1,0,1,1) \ = \ ? \\
p(1,0,1,0,1,1,0,1,1,0) \ = \ ? \\
p(1,1,0,0,1,1,0,0,1,1) \ = \ ? 
$$

이들에게 같은 수적인(numerical) 값을 할당하는 것에 논란이 있을까요? 각각의 배열(sequence)이 6개의 1과 4개의 0을 포함한다는 것에 주목합시다.

### 정의 3 (교환가능, Exchangeable)

$p(y_1, ..., y_n)$이 $Y_1, ... , Y_n$의 결합 밀도 함수라고 합시다. 만약 모든 {$1, ..., n$}의 순열(permutation) $\pi$에 대해 $p(y_1, ..., y_n) = p(y_{\pi_1}, ... , y_{\pi_n})$이라면, $Y_1, ..., Y_n$은 교환가능(exchangeable)하다고 합니다.

대충 말하자면, $Y_1, ..., Y_n$은 아래첨자의 라벨들이 결과에 대해 아무런 정보도 주지 못한다면 교환가능합니다. 

*독립 vs 의존(Independence versus dependence)*

다음과 같은 두 개의 확률이 할당되었다고 생각합시다.

$$
Pr(Y_{10} = 1) = a \\
Pr(Y_{10} = 1 | Y_1 = Y_2 = ... = Y_8 = Y_9 = 1) = b
$$

a < b, a = b, a > b 중 어떤 쪽인지 알 수 있나요? 만약 $a \neq b$라면 $Y_10$은 $Y_1, ..., Y_9$에 독립이 아닙니다.

### *조건부 독립(Conditional independence)*

누군가가 당신에게 1,272명의 답변자들 중 행복하다는 비율의 수적인(numerical) 값 $\theta$를 말해줬다고 가정합시다. 다음과 같은 확률 할당이 합리적이라고 생각하시나요?

$$
Pr(Y_{10} = 1 | \theta) \approx^{?} \theta \\
Pr(Y_{10} = 1 | Y_1 = y_1, ... , Y_9 = y_9, \theta) \approx^{?} \theta \\
Pr(Y_9 = 1 | Y_1 = y_1, ..., Y_8 = y_8 , Y_{10} = y_{10}, \theta) \approx^{?} \theta
$$

만약 이러한 확률들이 합리적이라면, 우리는 $Y_i$들을 $\theta$가 주어졌을 때 조건부 i.i.d이거나 최소한 근사적으로 그렇다고 생각할 수 있습니다. 모집단의 크기인 1,272는 샘플 크기인 10보다 훨씬 큽니다. 이러한 케이스에서는 비복원 추출은 근사적으로 i.i.d. 복원 추출과 같습니다. 조건부 독립을 가정하면 다음과 같은 식을 구할 수 있습니다.

$$
Pr(Y_i = y_i | \theta, Y_j = y_j, j \neq i) = \theta^{y_i}(1 - \theta)^{1-y_i} \\
Pr(Y_1 = y_1, ... , Y_{10} = y_{10} | \theta) = \prod^{10}_{i = 1} \theta^{y_i}(1 - \theta) ^ {1 - y_i} \\
= \theta^{\Sigma y_i} (1 - \theta)^{10 - \Sigma y_i}
$$

만약 $\theta$가 확실히 정해지지 않았다면, 그것에 대한 믿음을 사전 분포인 $p(\theta)$로 표현합니다. $Y_1, ..., Y_{10}$의 주변 결합 분포(marginal joint distribution)은 다음과 같습니다.

$$
p(y_1, ..., y_{10}) = \int^1_0 p(y_1, ..., y_{10} | \theta) p(\theta) d\theta = \int^1_0 \theta^{\Sigma y_i}(1-\theta)^{10 - \Sigma y_i} p(\theta) d\theta.
$$

이제 위에서 주어진 세 개의 이진 수열들에 대한 우리들의 확률을 생각해봅시다.

$$
p(1,0,0,1,0,1,1,0,1,1) = \int \theta^6(1-\theta)^4 p(\theta) d\theta \\
p(1,0,1,0,1,1,0,1,1,0) = \int \theta^6(1-\theta)^4 p(\theta) d\theta \\
p(1,1,0,0,1,1,0,0,1,1) = \int \theta^6(1-\theta)^4 p(\theta) d\theta 
$$

이와 같은 믿음의 모델에서는 $Y_1, ..., Y_n$이 교환 가능한 것으로 보입니다.

*Claim:*

만약 $\theta \sim p(\theta)$이고 $Y_1, ..., Y_n$이 $\theta$가 주어졌을 때 조건부 i.i.d.라면, 주변적으로(marginally, $\theta$에 조건부가 아니게), $Y_1, ..., Y_n$은 교환 가능합니다.

*Proof:*

$Y_1, ..., Y_n$이 미지의 파라미터 $\theta$가 주어졌을 때 조건부 i.i.d라고 가정합시다. 그렇다면 어떠한 {1,...,n}의 순열(permutation) $\pi$와 어떠한 값들의 집합 $(y_1, ..., y_n) \in \mathcal{Y}$들은 모두 다음을 따릅니다.

$$
p(y_1, ..., y_n) = \int p(y_1, ..., y_n | \theta) p(\theta) d\theta \ \ \ \  (\text{주변 확률(marginal probability)의 정의}) 
$$

$$
= \int \bigg \{ \prod^n_{i=1} p(y_i|\theta) \bigg \} p(\theta) d\theta \ \ \ \ (Y_i\text{들은 조건부 i.i.d.})
$$

$$
= \int \bigg \{ \prod^n_{i=1} p(y_{\pi_i} | \theta) \bigg \} p(\theta) d\theta \ \ \ \ (\text{확률의 곱은 순서에 의존하지 않는다})
$$

$$
= p(y_{\pi_1}, ..., y_{\pi_n}) \ \ \ \ (\text{주변 확률(marginal probability)의 정의}) 
$$

### *2.8 de Finetti의 정리*

우리는 $Y_1, ..., Y_n$이 $\theta$가 주어졌을 때 i.i.d. 이고, $\theta \sim p(\theta)$이면 $\Rightarrow$ $Y_1, ..., Y_n$이 교환 가능하다는 것을 배웠습니다.



화살표의 방향이 다르다면 어떨까요? {$Y_1, Y_2, ...$}가 모두 동일한 표본 공간 $\mathcal{Y}$를 가지는 확률 변수들의 잠재적인 무한 수열이라고 합시다.

### **정의 1 (de Finetti)**

$Y_i \in \mathcal{Y} \text{for all i} \in \{ 1,2,...\}.$라고 합시다. 어떠한 n에도 $Y1, ..., Y_n$에 대한 우리의 확률 모델이 교환 가능하다고 가정합시다. 즉 {1, ..., n}의 모든 순열 $\pi$에 대해

$$
p(y_1, ..., y_n) = p(y_{\pi_1}, ..., y_{\pi_n})
$$

입니다. 그렇다면 우리의 모델은 다음과 같이 쓸 수 있습니다. 

몇몇의 파라미터 $\theta$, $\theta$에 대한 사전 분포와 표본 추출 모델 $p(y|\theta)$에 대해

$$
p(y_i, ..., y_n) = \int \bigg \{ \prod^n_1 p(y_i | \theta) \bigg \} p(\theta) d\theta
$$

이 때, 사전 분포와 표본 추출 모델은 믿음 모델 $p(y_1, ..., y_n)$의 형태에 의존합니다.

확률 분포 $p(\theta)$는 우리의 믿음 모델 $p(y_1, ..., y_n)$로 부터 추론된 {$Y_1, Y_2, ...$}의 결과에 대한 우리의 믿음을 나타냅니다. 더 정확하게 표현하자면

$$
p(\theta) \text{는 이진(binary) 케이스에서 우리의 lim}_{n \rightarrow \infty} \Sigma Y_i / n \text{에 대한 믿음을 나타냅니다} \\
p(\theta) \text{는 일반적인(general) 케이스에서 우리의 각각의 c에 대한 lim}_{n \rightarrow \infty} \Sigma (Y_i \leq c) / n  \text{에 관한 믿음을 나타냅니다}
$$

이것과 이전 섹션에서의 주요 아이디어는 다음과 같이 요약될 수 있습니다.

$$
Y_1, ..., Y_n | \theta \text{가 i.i.d. 이고, } \theta \sim p(\theta) \Leftrightarrow Y_1, ..., Y_n \text{은 모든} n\text{에 대하여 교환 가능}
$$

어떤 조건에서 "$Y_1, ..., Y_n$이 모든 $n$에 대하여 교환 가능하다" 라는 명제가 합리적일까요? 이 조건을 만족하려면, 우리는 교환 가능성과 반복 가능성을 만족해야합니다. 교환 가능성은 만약 라벨들이 아무런 정보가 없다면 만족합니다. 반복 가능성이 합리적인 상황은 다음과 같습니다. 

$$
Y_1, ... , Y_n \text{은 반복 가능한 실험의 결과이다.} \\ 
Y_1, ... , Y_n \text{은 유한한 모집단에서 복원 추출된 것이다.} \\
Y_1, ... , Y_n \text{은 무한한 모집단에서 비복원 추출된 것이다.}
$$


만약 $Y_1, ... , Y_n$가 교환 가능하고 크기가 N >> n인 유한한 모집단에서 비복원추출되었다면, 그들은 근사적으로 조건부 i.i.d.라고 모델링 될 수 있습니다.(Diaconis and Freedman, 1980).

### *2.9 시사점과 참고자료*

일관적인 도박 전략에 관한 주관적인 확률의 개념은 de Finetti의 정리(de Finetti, 1931, 1937)를 만든 de Finetti에 의해 개발되었습니다. 이러한 주제들은 모두 Savage(Savage, 1954; Hewitt and Savage, 1955)를 포함한 많은 다른 사람들에게 연구되었습니다.

 교환 가능성의 개념은 de Finetti의 정리에서 다뤄진 무한하게 교환 가능한 수열의 개념을 넘어섭니다. Diaconis and Freedman(1980)은 유한한 모집단 또는 수열에서의 교환 가능성을 다뤘고, Diaconis(1988)은 몇몇의 다른 버전의 교환가능성을 다뤘습니다. Bernardo and Smith(1994)의 Chapter 4는 다양한 타입의 교환 가능성에 기반한 통계적 모델을 만드는 가이드를 제공합니다. 교환 가능성에 대한 아주 포괄적이고 수학적인 리뷰는 Aldous(1985)에 적혀있습니다. 이것은 특히  무작위 행렬에 적용되는 교환 가능성에 대한 훌륭한 연구를 제공합니다.
