---
layout: single
title:  "회귀(Regression) 및 추천(Recommendation)"
categories: Supervised_Learning
tag: [python, SupervisedLearning, MachineLearning, Regression, Recommendation]
toc: true
author_profile: false
use_math: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 지도학습 - 2. 회귀(Regression) 및 추천(Recommend)


## 학습 내용

- 회귀 문제의 특성에 대해서 알아본다



- 로지스틱 회귀 문제의 특성에 대해서 알아본다 (실제로는 분류에 사용되는 기법이다)



- 편향-분산 트레이드오프에 대해 알아본다



- 추천 문제의 특성과 전략에 대해서 알아본다


## 회귀

### 회귀(regression)

- 학습 데이터에 부합되는 출력값이 실수인 함수를 찾는 문제



- 입력값과 출력값이 실수인 힘수를 찾는 지도학습



- 학습 데이터가 주어졌을 때 데이터를 잘 나타낼 수 있는 함수를 찾아내고 그 함수들을 통해서 학습 데이터에 있지 않은 데이터의 입력에 대해서 출력값에 해당하는 값을 낼 수 있는 함수 f를 찾는 것



- 수치형 데이터 (x, y)가 주어질 때 x를 y로 대응시키는 함수를 찾음



- 함수의 형태는 주어지고(1차식, 2차식, 3차식, 지수함수 등) 파라미터들을 결정하게 되는데, x에 대한 참값(실제 대응값)과 기댓값의 차이(=오차)가 적어야 함


$ f^{*}(x) = argf^{*}(x) = argmin_f\sum_{i=1}^{n}(y_i-f(x_i))^2 $
f∗(x)=argf∗(x)=argminf∑ni=1(yi−f(xi))2


오차의 제곱들의 합을 최소화하는 함수를 찾는 것이 회귀의 목표이다.  

argmin은 오차 제곱의 합을 작게 하는 파라미터 아규먼트를 찾는 의미

![ErrorFunction](/assets/images/regression/errorfunction.png)


#### - 성능

  - 오차 : 예측값과 실제값의 차이

    - 테스트 데이터들에 대한 $(예측값 $f(x)$ - 실제값 $y$)^2$의 평균 또는 평균의 제곱근

    - 오차의 제곱의 평균처럼 정해놓고 씀

    

    - $ E = \frac{1}{n}\sum_{i=1}^{n}(y_i - f(x_i))^2 $

    - 모델의 종류(함수의 종류)에 영향을 받음

    ![diffoffunc](/assets/images/regression/diffodfunc.PNG)


### 경사 하강법(Gradient descent method)

- 오차 함수 E의 그레디언트 반대 방향으로 조금씩 움직여가며 최적의 파라미터를 찾으려는 방법 (요약)



- 회귀에서 함수의 형태를 결정하기 위해서 함수의 종류(1차식, 2차식 등)가 결정되면 함수의 형태는 거기있는 계수와 같은 파라미터들에 의해서 결정이 됨



- 예를 들어 y = ax + b인 경우에 a, b를 파라미터parameter(= 가중치weight)라고 하는데 이를 어떻게 결정할 것인가 -> 오차 함수 E의 값을 최소로 하는 파라미터를 찾음



- 일반적으로 오차 함수 E를 나타내는 함수의 형태는 복잡해서 계산으로 찾을 수가 없음. 그래서 그레디언트(각 파라미터별로 편미분을 한 벡터)를 이용하여 파라미터의 변화량에 대한 E의 변화량을 알아내고, E가 가장 작아지는 파라미터를 찾기 위해서 그레디언트 반대 방향으로 조금씩 움직이는 것이 경사 하강법



- 그레디언트(gradient) : 각 파라미터에 대해 편미분한 벡터    

$E = \frac{1}{n}\sum_{i=1}^{n}(y_i - f(x_i))^2$  

$f(x) = ax + b$  

$delE = (\frac{\partial E}{\partial a}, \frac{\partial E}{\partial b})$  





- 데이터의 입력과 출력을 이용하여 각 파라미터에 대한 그레디언트를 계산하여 파라미터를 반복적으로 조금씩 조정함  

$a \leftarrow a - \frac{\partial E}{\partial a}  $  

$b \leftarrow b - \frac{\partial E}{\partial b}$  

- 학습 데이터에 부합되는 출력값이 되도록 파라미터를 변경하는 일

$E = \sum_{i}^{}(y_i - f(x_i))^2$

![gradient](/assets/images/regression/gradient.PNG)

$\triangledown E = (\frac{\partial E}{\partial a}, \frac{\partial E}{\partial b})$

![errorfunction2](/assets/images/regression/errorfunction2.PNG)


### 최대 경사법(경사 하강법) 기반의 학습 알고리즘

![algorithms](/assets/images/regression/algorithms.png)


### [실습] 선형회귀

- 입력 : 배달거리(delivery distance)  /  출력 : 배달 시간(delivery time)

- 파라미터에 대한 1차 방정식을 사용한 회귀



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.array([[30, 12], [150, 25], [300, 35], [400, 48], [130, 21], [240, 33], 
                [350, 46], [200, 41], [100, 20], [110, 23], [190, 32], [120, 24],
                [130, 19], [270, 37], [255, 24]])
plt.scatter(data[:, 0], data[:, 1])  # 데이터 위치의 산포도 출력
plt.title("Linear Regression")
plt.xlabel("Delivery Distance")
plt.ylabel("Delivery Time")
plt.axis([0, 420, 0, 50])  # x의 범위 0~ 420 / y의 범위 0 ~ 50

# x = data[:, 0]
# y = data[:, 1]

# reshape를 해주는 이유 : 그래프가 2차원이기 때문에 1차원인 데이터를 2차원으로 변형시켜야 함
x = data[:, 0].reshape(-1, 1)  # 입력  한 열 나머지 숫자 행으로 트랜스폼
y = data[:, 1].reshape(-1, 1)  # 출력

model = LinearRegression()
model.fit(x, y)  # 모델 학습

y_pred = model.predict(x)  # 예측값 계산
plt.plot(x, y_pred, color = 'r')
plt.show()
```



### 회귀의 과적합(overfitting)과 부적합(underfitting)



- 과적합 : 지나치게 복잡한 모델(함수) 사용 -> 학습 데이터에 오류가 있을 개연성이 높다



- 부적합 : 지나치게 단순한 모델(함수) 사용 -> 학습이 제대로 되지 않음



- 회귀 모델을 만들 때에는 함수의 복잡도에 대해 고려하는 것이 중요하다

![somefittings](/assets/images/regression/somefitting.png)


### 회귀의 과적합 대응 방법



- 모델의 복잡도(model complexity)를 성능 평가에 반영



- 너무 복잡한 모델에 대해서는 패널티를 제공



- 목적함수가 최소가 되는 값을 찾는 것 -> 복잡도가 더해지게 되므로 너무 복잡한 모델은 선택되지 않음



- $ 목적함수 = 오차의 합(보통 오차의 제곱의 합을 사용) + (가중치)*(모델 복잡도(=패널티)) $


### 로지스틱 회귀(logistic regression)



- 회귀 함수를 찾아내는데 입력과 출력에서 출력에 해당하는 값이 0 아니면 1인 이진 출력을 학습 데이터로 사용



- 학습 데이터 : $\begin{Bmatrix}(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)

\end{Bmatrix}, y_i \in \begin{Bmatrix}0, 1\end{Bmatrix} : 이진 출력$



![logistic](/assets/images/regression/logistic.png)

- 출력값이 {0, 1}인 경우에 선형 회귀에서 제대로 표현하기 어려움



- 이를 해결하기 위해 로지스틱 회귀에선 출력을 만들어낼 때 s자 모양의 함수를 만들어내서 0과 1에 가까운 값이 나오도록 함



- 로지스틱 회귀 모델을 로지스틱 함수 logistic function 또는 시그모이드 함수 sigmoid function 이라고 함



- 참고로 시그모이드 함수는 신경망에서 자주 쓰임



- 이진 분류(binary classification) 문제에 적용 



- 의료분야, 생물학 분야에서 많이 사용되는 모델



- 고객 거래 정보와 신용도에 관한 데이터 분석을 했을 때 출력값이 0, 1이었는데 그때 선형 회귀를 쓰고 성능이 제대로 나오지 않았음. 로지스틱 회귀를 썼어야 적당했을 것 같음.



- 로지스틱 함수를 이용하여 함수 근사

![logistic2](/assets/images/regression/logistic2.PNG)

- 입력 : θ_1*x_1 + θ_2*x_2 + ... 



- 파라미터 : θ_1, θ_2, ...  우리가 찾아야 하는 것이 파라미터



- 로지스틱 회귀의 오류 함수 E는 오차 제곱의 합을 이용하는 선형 회귀와 다르게 확률 개념을 사용함





#### - 가능성(likelihood)

  - (예측값^참값 * (1 - 예측값)^1-참값) 들의 곱 -> 참값이 1이라면 예측값이 클수록 좋은 것 / 참값이 0이라면 예측값이 작을수록 좋은 것

  - $ P(X) = \prod_{i=1}^{N}f(x_i)^{y_i} \times (1-f(x_i))^{1-y_i} $

  - $ LogP = - \frac{1}{N} logP(X) = - \frac{1}{N} \sum_{i=i}^{N}(y_i logf(x_i) + (1-f(x_i))log(i-y_i)) $

  - 위의 LogP를 사용해서 경사하강법으로 학습하여 θ를 찾게 되는 것이 로지스틱 회귀


### [실습] 로지스틱 회귀



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('User_Data.csv')

x = dataset.iloc[:, [2, 3]].values  # 입력 정보 : Age, EstimatedSalary
y = dataset.iloc[:, 4].values       # 출력 정보 : Purchased
# 출력 정보 0과 1의 비율은 65:35
# 그럼 어느 정도의 비율까지 데이터 불균형 문제로 삼지 않을까?

from sklearn.model_selection import train_test_split  # 학습, 테스트 데이터 분할
# 학습 데이터와 테스트 데이터의 비율은 3:1 / 각 데이터는 랜덤하게 분할
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)

# 학습 데이터 정규화 : 단위가 다른 Age와 EstimatedSalary를 정규화함
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
# 정규화한 xtrain값 확인
print(xtrain[0:10, :])

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
# 학습 데이터를 학습
classifier.fit(xtrain, ytrain)
# 학습을 기반으로 예측
y_pred = classifier.predict(xtest)

# 테스트 데이터와 예측 데이터를 비교 : 이진 분류기 성능평가
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print("혼동행렬 : \n", cm)

# 이진 분류기 성능 평가 : 정확도
from sklearn.metrics import accuracy_score
print("정확도 : ", accuracy_score(ytest, y_pred))

# 시각화 (두가지 요인에 대해 비교할 때 유용한 툴인 듯. 필요할 때 복붙하기...) 
from matplotlib.colors import ListedColormap
X_set, y_set = xtest, ytest

# 메쉬그리드(격자)를 넣을 위치를 만듦
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                              stop = X_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = X_set[:, 1].min() - 1,
                             stop = X_set[:, 1].max() + 1, step = 0.01))

# 컨투어 펑션(등고선 그리기) 적용
plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 산점도
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
               c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Classifier (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 빨간색 영역에 있는 것은 purchased가 0, 초록색은 1
# 빨간색 영역에 초록색 원 / 초록색 영역에 빨간색 원은 오차를 의미함
```


<pre>
[[ 0.58164944 -0.88670699]
 [-0.60673761  1.46173768]
 [-0.01254409 -0.5677824 ]
 [-0.60673761  1.89663484]
 [ 1.37390747 -1.40858358]
 [ 1.47293972  0.99784738]
 [ 0.08648817 -0.79972756]
 [-0.01254409 -0.24885782]
 [-0.21060859 -0.5677824 ]
 [-0.21060859 -0.19087153]]
혼동행렬 : 
 [[65  3]
 [ 8 24]]
정확도 :  0.89
</pre>


### 편향-분산 트레이드오프(bias-variance tradeoff) : 편향과 분산은 반비례 관계

#### - 편향 (bias) : 모델을 잘못 선택했을 때 발생



  - 학습 알고리즘에서 잘못된 가정을 할 때 발생하는 오차



  - 큰 편향값은 부적합 문제 초래

  



#### - 분산 (variance) : 오차가 있는 데이터를 학습했을 때 발생



  - 학습 데이터에 내재된 작은 변동(fluctuation) 때문에 발생하는 오차



  - 큰 분산값은 큰 잡음까지 학습하는 과적합 문제 초래 -> 오차가 있는 데이터를 학습 = 과적합 초래



- 편향과 분산이 작을수록 좋지만 반비례 관계이므로 중간을 맞추는 것이 현실적

![bias1](/assets/images/regression/bias1.png)

![bias2](/assets/images/regression/bias2.png)


### 편향-분산 분해

예측모델 f(x)의 평균을 이용해 복잡한 계산을 해보면



$bias\left(편향\right)는 \overline {F}\left(x_0\right)-f^*\left(x_0\right),\ \ variance\left(분산\right)는\ E\left\lceil {\overline {F}\left(x_0\right)-f\left(x_0\right)^2}\right\rceil , $σ는피할 수없는 잡음(noise)에 해당한다.

#### 회귀 모델에서 오차는 편향(bias), 분산(variance), 잡음(noise) 성분으로 구별할 수 있다.

![bias3](/assets/images/regression/bias3.PNG)



```python
# 데이터의 분포가 곡선으로 나타날 때 다항회귀 사용
# 다항회귀 : 데이터들간의 형태가 비선형일 때 데이터에 각 특성의 제곱을 추가해줘서
# 특성이 추가된 비선형 데이터를 선형 회귀 모델로 훈련시키는 방법
from sklearn.preprocessing import PolynomialFeatures
# 선형회귀
from sklearn.linear_model import LinearRegression
# 데이터 변환 과정과 머신러닝을 연결해주는 파이프라인
# make_pipeline을 통해 PolynomailFeatures와 LinearRegression의 과정이 
# 한번으로 통합된 모델을 생성함
from sklearn.pipeline import make_pipeline
import numpy as np

np.random.seed(1)
# 40*1 쉐이프의 랜덤값이 할당된 2차원 행렬 X 생성
X = np.random.rand(40, 1) ** 2
# X값에 맞춘 40개의 1차원 행렬 생성
# .ravel() : 행렬을 1차원 행렬로 바꿈
y = (10-1./(X.ravel() + 0.1)) + np.random.randn(40)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
X_test = np.linspace(-0.1, 1.1, 500).reshape(-1, 1)
fig = plt.figure(figsize = (12, 10))
# i는 1부터 1씩 늘어나고 degree는 enumerate([1, 3, 5, 10])순서 할당
for i, degree in enumerate([1, 3, 5, 10], start = 1):
    ax = fig.add_subplot(2, 2, i)
    # s는 마커 사이즈
    ax.scatter(X.ravel(), y, s = 15)
    # PolynomialFeatures(degree) : 현재 데이터를 degree에 따른 다항식 형태로 변형(1, 3, 5, 10)
    # LinearRegression() : 선형 회귀
    # make_pipeline을 통해 다항 회귀와 선형 회귀가 통합된 모델을 생성해
    # 학습 데이터를 학습시키고 테스트 데이터에 대해서 예측값을 생성
    y_test = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(X, y).predict(X_test)
    ax.plot(X_test.ravel(), y_test, label = 'degree = {0}'.format(degree))
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-2, 12)
    ax.legend(loc = 'best')
```



※ 그래프에 편향과 분산도 추가하고 싶었지만 다시 내용을 읽어본 결과 편향과 분산은 f(x)의 평균에 대한 계산이기 때문에 하나의 수치에 대한 내용이라 계산값을 나타낼 수는 있지만 그래프는 그릴 수 없다는 것을 깨달았다.


## 추천

### 추천 (recommendation)



- 개인별로 맞춤형 정보를 제공하려는 기술



- 사용자에게 맞춤형 정보를 제공하여 정보 검색의 부하를 줄여주는 역할



- 사용자가 정보 검색을 쉽게 할 수 있음 / 서비스를 제공하는 측에서는 사용자에게 맞춤형 정보를 제공할 수 있음



- 많이 연구되는 분야. 비즈니스 측면에서 효과가 크다.


### - 추천 데이터



  - 희소 행렬(sparse matrix) 형태



    - 많은 원소가 비어있음



    - 비어있는 부분을 채우는 것이 추천에 해당



    - 비어있는 부분을 채운 값 중에서 가장 큰 값이 나온 대상을 추천함



    - 채워진 부분은 학습 데이터를 준 것 / 비어있는 곳이 테스트 데이터라고 볼 수 있음

    ![recommend_data](/assets/images/regression/recommend_data.png)


### - 추천 기법 : 크게 3가지가 있음



####   - 내용 기반 추천(content-based recommendation)



    - 고객이 이전에 높게 평가했던 것과 유사한 내용을 갖는 대상을 추천



####   - 협력 필터링(collaborative filtering)



#####    - 사용자간 협력 필터링(user-user collaborative filtering)



      - 추천 대상 사용자와 비슷한 평가를 한 사용자 집합 이용



      - 쉽게 말하자면, 추천을 할 대상자가 내려왔던 평가와 비슷한 경향을 가진 사용자 집합에서 높은 평가를 받은 것을 추천에 이용



#####    - 항목간 협력 필터링(item-item collaborative filtering)



      - 항목간의 유사도를 구하여 유사 항목을 선택



      - 추천했던 대상과 유사한 특징을 가진 대상을 추천



####  - 은닉 요소 모델(latent factor model)



    - 추천 데이터 행렬을 다른 행렬들에 대한 곱으로 나타낸다고 가정



    - 아래 행렬을 순서대로 M, A, B라고 한다면 M = A * B인 것으로 가정



    - A행렬과 B행렬의 요소들의 값을 어떻게 결정하는가? -> M행렬에 채워져있는 값을 가지고 결정



    - 그렇게 구해진 A, B행렬을 곱해서 비어져있는 공간이 채워진 행렬 M이 생성됨. 그 채워진 값을 가지고 추천함. (그럼 A, B행렬을 구하는 것도 지도학습인가?)



    - 행렬 분해에 기반한 방법   

![recommend_data2](/assets/images/regression/recommend_data2.png)

