---
layout: single
title:  "분류 Classification"
categories: Supervised_Learning
tag: [python, SupervisedLearning, MachineLearning, Classification]
toc: true
author_profile: false
use_math: true
typora-copy-images-to: ..\images
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


# 지도학습 - 1. 분류(Classification)

지도학습은 분류, 회귀, 추천으로 나뉘는데 이번 포스트에서는  

분류 - 과적합 학습의 문제 / 학습 데이터가 적은 경우의 성능 평가 / 불균형 데이터 문제 / 이진 분류기의 성능 평가에 대해서 공부한 내용을 기록한다.


## 학습 내용

- 지도학습에 속하는 분류와 회귀 문제에 데이터의 특성을 알아본다



- 학습 모델의 과적합과 부적합에 대해서 알아본다



- 학습 모델의 성능 평가 방법을 알아본다



- 불균형 부류 데이터 문제와 해결 방법에 대해서 알아본다



- 이진 분류기(두 개의 범주만 있는 경우)의 성능 평가 방법에 대해서 알아본다



## 지도학습

### 지도학습(supervised learning)

- 주어진 (입력, 출력)에 대한 데이터 이용 : 학습 데이터(training data)

$$
{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}
$$

- 지도학습의 목적 : 새로운 입력이 있을 때 결과를 결정할 수 있도록 하는 방법을 찾아내는 것.

  x1이 들어갈 때 y1이 나오고 x2가 나올 때 y2가 나오는 함수를 찾는 것이 지도학습의 목적



 - $y\ =\ f\left(x\right)$

- 분류(Classification) : 출력값에 해당하는 것이 부류(class), 범주(category) 중의 하나로 결정



- 회귀(Regression) : 출력이 연속인 영역(continuous domain)의 값 결정



- 추천도 지도학습의 일종


## 분류

### 분류(classification) 

- 출력이 정해진 부류(Class)가 있을 때 입력을 각 클래스에 대응시키는 문제  

![결정경계](/assets/images/classification/결정경계.PNG)

- 학습 데이터가 주어질 때, 분류 문제에서는 데이터들을 분류하는 경계(결정 경계 decision boundary)를 나타내는 함수를 찾는 것이 목표

- 학습을 끝내고 새로운 데이터가 들어온다면 결정 경계를 나타내는 함수에 의해 분류됨  





- 분류 문제의 학습

  - 학습 데이터를 잘 분류할 수 있는 함수를 찾는 것

  - 함수의 형태는 수학적 함수일 수도 있고, 규칙일 수도 있음  

  

  

- 분류기(classifier)

  - 학습된 함수를 이용하여 데이터를 분류하는 프로그램


### 분류 학습 데이터와 학습 결과의 예

#### 범주형 속성(playTennis의 예)   

![playTennis데이터](/assets/images/classification/playTennis_data.png)

![playTennis'_decisiontree](/assets/images/classification/playTennis'_decisiontree.PNG)

- 입력 데이터 : Day, Outlook, Temperature, Humidity, Wind 

- 출력 데이터 : playTennis(테니스 여부)

- 이러한 분류를 할 수 있는 데이터에 대해서 찾아낸 지식의 한 형태가 결정 트리

- 결정 트리 : 분류에 대한 학습 데이터가 주어질 때 사용될 수 있는 방법 중의 하나


#### 수치형 속성(iris data의 예)

![iris_data](/assets/images/classification/iris_data.PNG)

![iris_knn](/assets/images/classification/iris_knn.PNG)

![iris_decisiontree](/assets/images/classification/iris_decisiontree.PNG)

- 입력 데이터 : 수치형

- 출력 데이터 : 범주형

* 입력 데이터가 수치형이고 출력 데이터가 수치형인 경우도 물론 있음


### [실습] Iris DecisionTree



```python
import matplotlib.pyplot as plt
# iris 데이터를 불러오기 위함
from sklearn.datasets import load_iris
# 결정트리, 트리를 그리기 위한 임포트
from sklearn.tree import DecisionTreeClassifier, plot_tree

# iris 데이터를 읽어서 iris에 저장
iris = load_iris() 
# entropy : 내부 노드에 어떤 속성을 써서 결정을 할 것인지 정하는 기법
# 결정트리 분류기의 객체를 생성. 
# criterion = "entropy" : entropy를 사용해서 속성을 정함
# random_state = 0 : 초기값은 무작위
# max_depth = 3 : 트리의 depth는 3으로
decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 0, max_depth = 3)
# fit 메소드를 써서 학습 / iris.data가 입력, iris.target이 출력
decision_tree = decision_tree.fit(iris.data, iris.target)

# tree를 plot할 수 있는 도화지 생성
plt.figure()
# 결정트리를 그림. plt.show()를 안하면 내부에 그림을 그리는 것이라 보이지 않음
plot_tree(decision_tree, filled = True)
# 그림 출력
plt.show()
```


### 분류기 학습 알고리즘

- 결정트리(decision tree) 알고리즘

- KNN(K-nearest neighbor) 알고리즘

- 다층 퍼셉트론 신경망

- 딥러닝 신경망

- 서포트 벡터 머신(SVM)

- 에이다부스트(AdaBoost)

- 랜덤 포레스트(Random forest)

- 확률 그래프 모델(Probabilistic graphical model)


### 이상적인 분류기

- 학습에 사용되지 않은 데이터에 대해서 분류를 잘 하는 것 = 일반화(generalization) 능력이 좋은 것

- 데이터의 오류, 과적합 등 다양한 이유로 학습 데이터를 100% 반영하는 분류기는 좋은 분류기가 아님


### 데이터의 구분

#### - 학습 데이터(training data)

  - 분류기를 학습하는데 사용하는 데이터 집합

  - 학습 데이터가 많을수록 유리  

  

  

#### - 테스트 데이터(test data)

  - 학습된 모델의 성능을 평가하는데 사용하는 데이터 집합

  - 검증에 사용되지 않은 데이터

  

  

#### - 검증 데이터(Validation data)

  - 학습 과정에서 학습을 중단할 시점을 결정하기 위해 사용하는 데이터 집합 

  - 과적합을 피하기 위함


### 과적합(overfitting)과 부적합(underfitting)

#### - 과적합

  - 학습 데이터에 대해서 지나치게 잘 학습된 상태

  - 데이터는 오류나 잡음을 포함할 개연성이 크기 때문에 학습 데이터에 대해 매우 높은 성능을 보이더라도 학습되지 않은 데이터에 대해 좋지 않은 성능을 보일 수 있음





#### - 부적합

  - 학습 데이터를 충분히 학습하지 않은 상태

  

  

![overfitting_underfitting](/assets/images/classification/overfitting_underfitting.png)


### 과적합 회피 방법

#### - 학습 데이터에 대한 성능

  - 학습을 진행할수록 오류 개선 경향

  - 지나치게 학습이 진행되면 과적합 발생

  

  

#### - 학습 과정에서 별도의 검증 데이터에 대한 성능 평가

  - 검증 데이터에 대한 오류가 감소하다가 증가하는 시점에 학습 중단

  - 검증 데이터는 과적합을 피하기 위해 사용하는 데이터

  

![timetostoplearning](/assets/images/classification/timetostoplearning.png)


### 분류기의 성능 평가

#### - 정확도(Accuracy)

  - 얼마나 정확하게 분류하는가

  - 정확도 = 옳게 분류한 데이터의 개수 / 전체 데이터 개수

  - 테스트 데이터에 대한 정확도를 분류기의 정확도로 사용

  - 정확도가 높은 분류기를 학습하기 위해서는 많은 학습 데이터를 사용하는 것이 유리

  - 학습 데이터와 테스트 데이터는 겹치지 않도록 해야 함


### 데이터가 부족한 경우 성능평가

- 데이터가 부족한 상태로 데이터를 분할하면 학습 데이터가 부족해서 부적합이 발생함 -> 테스트 데이터는 성능만 알면 됨 -> K-겹 교차 검증

- 별도로 테스트 데이터를 확보하면 비효율적

- 가능하면 많은 데이터를 학습에 사용하면서, 성능 평가하는 방법이 필요



#### - K-겹 교차 검증(k-fold cross_validation) 사용

  - 전체 데이터를 k 등분

  - 각 등분을 한번씩 테스트 데이터로 사용하여, 성능 평가를 하고 평균값 선택

  

  

![kfoldcrossvalidation](/assets/images/classification/kfoldcrossvalidation.png)



### 불균형 부류 데이터(imbalanced class data) 문제

- 특정 부류에 속하는 학습 데이터의 개수가 다른 부류에 비해 지나치게 많은 경우

- 정확도에 의한 성능 평가는 무의미할 수 있음

  - 예. A 부류의 데이터가 전체의 99%인 경우, 분류기의 출력을 항상 A 부류로 하더라도 정확도는 99%가 됨 -> 정확도가 의미없음



- 고객 거래 정보와 신용도에 관한 데이터 분석을 했을 때, 신용이 안전한 고객(1)과 신용이 위험한 고객(0)의 비율이 9:1 정도 되었음. 이런 경우를 불균형 부류 데이터 문제라고 칭함. 당시에는 리샘플링을 통해서 해결했음.

![imbalancedclassdata](/assets/images/classification/imbalancedclassdata.png)

#### - 대응 방안

  - ●에서 틀리면 1만큼 틀린 것이지만 ★에 대해서 틀리면 5만큼 틀린 것이라고 하는 등 가중치를 고려한 정확도 척도를 사용

  - 많은 학습 데이터를 갖는 부류에서 재표본추출(re-sampling)

  - 적은 학습데이터를 갖는 부류에 대해서 인공적인 데이터 생성


#### - SMOTE(Synthetic Minority Over-sampling Technique) 알고리즘

  - 빈도가 낮은 부류의 학습 데이터를 인공적으로 만들어 내는 기법

  1. 임의의 낮은 빈도 부류의 학습 데이터 x를 선택

  2. x의 k-근접이웃(KNN)인 같은 부류의 데이터 선택

  3. k-근접이웃 중에 무작위로 하나 y를 선택

  4. x와 y를 연결하는 직선 상의 무작위 위치에 새로운 데이터 생성

  

![SMOTE](/assets/images/classification/SMOTE.png)


### 이진 분류기의 성능 평가

#### - 이진 분류기

  - 두 개의 부류만을 갖는 데이터에 대한 분류기

 

![classifier](/assets/images/classification/classifier.png)



  - 민감도(재현율, 진양성율) : 실제 양성 중에 예측으로 얼마만큼을 맞췄는가 = TP / TP + FN



  - 특이도(진음성율) : 실제 음성 중에 예측으로 얼마만큼을 맞췄는가 = TN / FP + TN



  - 정밀도: 양성이라 예측한 것들 중 진짜 양성인 것 = TP / TP + FP



  - 음성 예측도 : 음성이라 예측한 것들 중 진짜 음성인 것 = TN / TN + FN



  - 위양성율 : 가짜를 진짜로 예측 = FP / FP + TN = 1 - 특이도



  - 정확도 : 전체 중에서 제대로 예측한 것 = TP + TN / TP + FP + TN + FN



  - F1 측도 : 정밀도와 재현율을 결합한 측도로, 재현율, 정밀도, F1 측도는 정보검색 분야에서 주로 쓰임   

  $=\ 2\ \cdot \frac{\left(정밀도\right)\cdot \ \left(재현율\right)}{\left(정밀도\right)+\ \left(재현율\right)}$


#### - ROC 곡선

  - 이진 부류 판정을 하는데 임계값에 따라서 (위양성율, 민감도)를 그래프로 나타낸 것





![ROC](/assets/images/classification/ROC.png)

  - 빨간 선이 기준일 때 기준보다 크면 Positive, 작으면 Negative로 판정하는데 이 기준을 계숙 움직일 때 위양성율과 민감도의 반응을 그래프로 그려놓은 것



  - ROC 커브는 이진 분류기에 대해서 그리게 되고, 그 중 특히 임계값에 대해서 성능이 결정되는 모델에 대해서 ROC 커브 곡선을 그림


#### - AUC(Area under the curve)

  - ROC 커브가 여럿 있을 때, 가짜를 진짜로 예측하는 위양성율은 작을수록 좋고 민감도는 클수록 좋음 -> AUC 면적이 클수록 좋음

  

![auc](/assets/images/classification/auc.png)

