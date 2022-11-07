# Databricks notebook source
# MAGIC %md # 테이블 형식의 데이터를 머신러닝 모델로 학습하기: End-to-End실습
# MAGIC 
# MAGIC 이번 실습은 아래의 내용을 진행합니다:
# MAGIC - Seaborn과 matplotlib를 활용하여 시각화
# MAGIC - Parallel 하이퍼파라미터 스윕 (Sweep)을 통해 데이터를 머신러닝 모델로 학습
# MAGIC - MLflow의 하이퍼파라미터 스윕을 통해 학습된 모델에 대한 결과 탐색
# MAGIC - MLflow 내 최적의 성능을 낸 모델 등록
# MAGIC - Low-latency 요청을 수행하기 위해 모델 서빙 셋업
# MAGIC 
# MAGIC 이 실습에서는 물리화학적 요소를 바탕으로 포르투갈 "Vinho Verde"의 품질을 예측하는 모델을 생성합니다.
# MAGIC 
# MAGIC 해당 샘플 데이터셋은 UCI Machine Learning Repository [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009]에서 원본을 확인하실 수 있습니다.
# MAGIC 
# MAGIC ## 요구사항
# MAGIC 이 노트북을 실행하기 위해서는 Databricks Runtime for Machine Learning을 사용해야 합니다. Databricks Runtime 7.5 ML 이상을 사용해주세요.

# COMMAND ----------

# MAGIC %md 이번 실습 과정 중 마지막 영역에서 모델 서빙을 위해 토큰을 사용하게 됩니다. Personal Access Token은 좌측 사이드바에서 Settings에 들어가서 User Settings 메뉴를 클릭한 다음 "Access Token" 탭 아래에서 새로운 토큰을 생성할 수 있습니다.
# MAGIC 
# MAGIC **주의사항:토큰을 발급하는 과정에서 실제 토큰 값은 처음 한번만 확인할 수 있기 때문에 잘 저장해두시고 기간이 만료된 경우 토큰값을 변경해주어야 합니다**

# COMMAND ----------

# 향후 모델을 서빙하는 과정에서 Personal Access Token이 필요합니다.
import os
os.environ["DATABRICKS_TOKEN"] = "<Personal Access Token>"

# COMMAND ----------

import pandas as pd

# 데이터 업로드
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md 두 개의 다른 DataFrame을 하나의 데이터셋으로 통합합니다. 레드와인인 경우 'is_red' 컬럼이 1, 화이트와인인 경우 0으로 지정합니다.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# 컬럼명 변경 (띄어쓰기 제거)
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## 데이터 시각화
# MAGIC 
# MAGIC 모델을 학습하기 전에 Seaborn과 Matplotlib를 통해 데이터셋을 탐색해보겠습니다. 

# COMMAND ----------

# MAGIC %md 품질이라는 종속변수를 히스토그램으로 표현해보겠습니다.

# COMMAND ----------

import seaborn as sns
sns.displot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md 히스토그램을 그려본 결과 품질 스코어는 3과 9 사이 대부분 분포되어 있습니다.
# MAGIC 
# MAGIC 와인이 품질이 좋다는 것을 품질이 7 이상일 경우로 지정합니다.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plot은 feature와 두 가지의 레이블 간 상관관계를 확인하기에 가장 좋은 방법입니다.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue 
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md 위 box plot에서 몇몇 변수는 품질을 예측하기에 좋은 역할을 합니다. 
# MAGIC 
# MAGIC - 알코올 box plot의 경우, 품질이 좋은 와인의 알코올 함유 중간값이 75th quantile의 품질 낮은 와인보다 더 높습니다. 알코올 함유량은 품질과 상관관계가 있습니다.
# MAGIC - 밀도 box plot의 경우, 품질이 낮은 와인일수록 품질이 높은 와인보다 밀도가 높습니다. 밀도는 품질과 역 상관관계가 있습니다.

# COMMAND ----------

# MAGIC %md ## 데이터 전처리
# MAGIC 모델을 학습하기 이전에 누락된 값을 확인하고 학습용, 검증용 데이터로 나눕니다.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md 현재 데이터에는 누락된 값이 없습니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 기본 모델 (Baseline 모델) 학습을 위한 데이터셋 준비
# MAGIC 데이터는 3 가지 세트로 나뉘게 됩니다:
# MAGIC - 학습용 (60%의 데이터셋이 학습에 사용됩니다)
# MAGIC - 검증용 (20%의 데이터셋이 하이퍼파라미터 튜닝 용도로 사용됩니다)
# MAGIC - 테스트용 (20%의 데이터셋이 학습되지 않은 데이터를 통해 성능을 측정하기 위해 사용됩니다)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# 학습 데이터 분산
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# 나머지 데이터 검증용, 테스트용 데이터로 분산
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md ## 기본 모델 생성 (Baseline model)
# MAGIC 아래 task는 예측하고자 하는 output이 이진분류 (binary classification)이고 다양한 변수간의 교류로 인해 Random Forest classifier가 적합합니다. 
# MAGIC 
# MAGIC 아래 코드는 scikit-learn 라이브러리를 사용하여 간단한 분류기를 생성합니다. MLflow 를 사용하여 모델의 정확도와 나중에 다시 사용할 수 있도록 모델을 저장할 수 있습니다.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# Scikit-learn의 RandomForestClassifier는 이진분류 (0 또는 1)에 대한 예측 결과를 제공합니다. 
# 아래 코드는 SklearnModelWrapper라는 wrapper 함수를 제공하고 
# predict_proba라는 방법을 사용해서 각각의 클래스가 얼만큼의 확률을 가지고 있는지 제공합니다.

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run는 모델의 성능을 추적하기 위해 새로운 MLflow run을 생성합니다. 
# 그리고 아래 mlflow.log_param이라는 방법을 통해 사용된 파라미터를 추적할 수 있고
# mlflow.log_metric를 통해 정확도와 같은 메트릭을 기록할 수 있습니다.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba은 [prob_negative, prob_positive]에 대한 값을 제공합니다.
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # ROC 커브 아래의 넓이를 나타내는 AUC 값을 메트릭으로 사용합니다.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # 모델의 input과 output 스키마를 정의한 시그니처로 로깅합니다.
  # 모델이 배포될 경우 해당 시그니처는 input을 검증하기 위해 사용됩니다.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow는 모델을 서빙하기 위한 콘다 환경 (conda environment)를 생성하여 utility를 보관합니다.
  # 필요한 dependency는 conda.yaml 파일에 저장되고 모델과 함께 로깅됩니다.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md 추가로 모델에서 판단한 feature 중요도도 확인할 수 있습니다.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md 이전에 boxplot에서 확인된 내용과 같이 알코올과 밀도 모두 품질을 예측하는데 중요한 역할을 합니다.

# COMMAND ----------

# MAGIC %md MLflow 를 통해 AUC 메트릭을 로깅했습니다. 우측 상단의 **Experiment**를 클릭하여 Experiment Runs 사이드바를 확인하실 수 있습니다.
# MAGIC 
# MAGIC 모델을 통해 게산된 AUC는 0.854입니다.
# MAGIC 
# MAGIC 무작위의 분류기의 AUC가 0.5가 될 수 있었음에도 불구하고 현재 AUC는 이보다 낫습니다. 해당 메트릭에 대해 더 알고 싶으시면 [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) 을 참고해주세요.

# COMMAND ----------

# MAGIC %md #### MLflow 모델 레지스트리에 모델 등록하기
# MAGIC 
# MAGIC 모델 레지스트리에 모델을 등록하게 되면 Databricks 아무곳에서도 손 쉽게 모델을 참조하실 수 있습니다.
# MAGIC 
# MAGIC 아래 코드는 어떻게 모델 레지스트리에 등록을 할 수 있는지 나타내고 있습니다. 또한 모델을 등록하는 부분은 UI상에서도 가능합니다. "UI를 통해 모델 생성 또는 등록하기"를 참조해주세요 ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# 만약 "PERMISSION_DENIED: User does not have any permission level assigned to the registered model"와 같은 에러가 발생한다면 
# 그 이유는 동일한 이름 (wine_quality)으로 모델이 이미 존재하기 때문입니다. 다른 이름을 시도해주세요.
model_name = "wine_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# 모델을 등록하는 부분은 몇 초 가량 소요됩니다. 아래와 같이 약간의 딜레이를 넣겠습니다.
time.sleep(15)

# COMMAND ----------

# MAGIC %md 이제 Models 페이지에서 모델을 확인하실 수 있습니다. Models 페이지를 확인하기 위해서는 좌측 사이드바에 있는 Models 아이콘을 클릭하시기 바랍니다. 
# MAGIC 
# MAGIC 다음으로 이 모델을 production으로 옮겨보고 모델 레지스트리에서 이 노트북으로 옮겨보도록 하겠습니다.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md Models 페이지에 이제 모델의 버전이 "Production"으로 되어 있는 것을 확인하실 수 있습니다.
# MAGIC 
# MAGIC 그리고 "models:/wine_quality/production"의 경로를 활용하여 모델을 참조하실 수 있습니다.

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# 추가확인: 아래 프린트되는 내용은 MLflow를 통해 로깅된 AUC결과와 동일해야 합니다.
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## 새로운 모델로의 실험
# MAGIC 
# MAGIC Random Forest 모델은 하이퍼파라미터 튜닝을 사용하지 않고도 좋은 성능을 냈습니다.
# MAGIC 
# MAGIC 아래 코드는 XGBoost 라이브러리를 화용하여 더 정확한 모델을 만들 수 있습니다. 코드를 통해 Hyperopt와 SparkTrials 방법을 사용하여 여러 모델을 평행(parallel)하게 하이퍼파라미터 스윕 방법으로 수행할 수 있습니다. 이전과 마찬가지로 코드는 MLflow의 파라미터 셋팅에 따라 성능을 추적할 수 있습니다.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, 
}

def train_model(params):
  # MLflow의 autologging 기능을 통해 하이퍼파라미터와 학습된 모델은 자동으로 MLflow에 로깅 됩니다.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)
    # xgb가 평가 메트릭을 계속 추적할 수 있도록 검증용 셋도 제공합니다.
    # XGBoost는 평가 메트릭이 더 좋아지지 않을때까지 학습을 진행하다가 종료 됩니다.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Loss를 -1*auc_score로 지정하여 fmin이 auc_score를 최대화할 수 있도록 합니다.
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Parallelism이 클 수록 빠른 속도록 보장하지만 하이퍼파라미터 스윕을 최적화 하지는 않습니다. 
# 가장 합리적인 parallelism 값은 max_evals 값의 제곱근입니다.
spark_trials = SparkTrials(parallelism=10)

# fmin을 MLflow에서 실행하고 각각의 하이퍼파라미터 세팅이 parent에 종속된 chile run으로 로깅될 수 있도록 합니다.
# run의 이름은 "xgboost_models"로 지정합니다.
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=20,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md #### MLflow를 사용하여 결과 확인하기
# MAGIC 
# MAGIC Experiment Runs 사이드바를 열어 MLflow run들을 확인할 수 있습니다. Date를 클릭하고 'auc'를 선택하면 auc 메트릭을 기준으로 run이 새로 정렬되고 확인할 수 있습니다. 가장 높은 auc 값은 0.9 입니다.
# MAGIC 
# MAGIC MLflow는 각 run에 해당하는 파라미터와 성능 메트릭을 추적할 수 있습니다. Experiment Runs 사이드바 상단의 외부 링크<img src="https://docs.databricks.com/_static/images/icons/external-link.png"/>를 클릭하셔서 MLflow Runs 테이블로 들어가보실 수 있습니다. 

# COMMAND ----------

# MAGIC %md 이제 AUC와 연관된 하이퍼파라미터 선택에 대해 알아보도록 하겠습니다. "+" 아이콘을 클릭하여 parent run을 확장하고 parent를 제외한 나머지 run을 모두 선택한 후 "Compare"를 클릭합니다. Parallel Coordinates Plot을 선택합니다.
# MAGIC 
# MAGIC Parallel Coordinates Plot은 각각의 파라미터가 메트릭에 어떤 영향을 주는지 이해하기 위해 매우 유용합니다. Plot에서 우측 상단의 핑크색 슬라이딩 바를 드래그하여 AUC값과 이에 해당하는 파라미터 값을 하이라이트하실 수 있습니다. 아래의 이미지와 같이 가장 높은 AUC 값에 대해 확인하실 수 있습니다:
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC 
# MAGIC 보시는 바와 같이 가장 성능이 좋은 run은 reg_lambda 값과 learning_rate의 수치가 낮습니다.
# MAGIC 
# MAGIC 또다른 하이퍼파라미터 스윕을 통해 이런 파라미터 값을 더 낮게하여 탐색할 수도 있습니다. 하지만 이번 실습에서는 더 진행하지 않도록 하겠습니다.

# COMMAND ----------

# MAGIC %md 
# MAGIC 지금까지는 MLflow를 사용하여 각 하이퍼파라미터 세팅을 로깅하였습니다. 아래의 코드에서는 성능이 가장 좋았던 run을 찾고 모델 레지스트리 모델을 저장하게 됩니다.

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### MLflow 모델 레지스트리 내 production 용 `wine_quality` 모델 업데이트 하기
# MAGIC 
# MAGIC 위쪽 코드에서는 기본모델 (Baseline 모델)을 모델 레지스트리에 `wine_quality`라는 이름으로 저장하였습니다. 이번에 더 정확한 모델을 만들었으니 동일한 `wine_quality` 이름으로 업데이트 해보겠습니다.

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# 모델을 등록하는 부분은 몇 초 가량 소요됩니다. 아래와 같이 약간의 딜레이를 넣겠습니다.
time.sleep(15)

# COMMAND ----------

# MAGIC %md 좌측 사이드바에서 **Models** 이라는 부분을 클릭해서 `wine_quality` 모델이 이제는 두 가지의 버전을 가진 것을 확인하실 수 있습니다. 
# MAGIC 
# MAGIC 아래 코드는 새로운 버전을 production 환경에서 사용할 수 있도록 합니다.

# COMMAND ----------

# 이전 버전의 모델을 Archive 하기
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# 새로운 버전의 모델을 Production용으로 Promote 하기
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md 이제 load_model을 호출하는 클라이언트는 새로운 모델을 사용하도록 되어 있습니다.

# COMMAND ----------

# 아래 코드는 "Baseline Model 만들기"와 동일합니다. 클라이언트가 새로운 모델을 사용하기 위해 별 다른 변경이 필요하지는 않습니다.
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md 이제 새로운 모델의 테스트 데이터 auc 값은 0.9입니다. 기본 모델의 성능보다 더 나은 성능입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 모델 서빙
# MAGIC 
# MAGIC 실제로 모델을 production 환경에서 낮은 latency 예측으로 사용하고자 할경우 MLflow 모델 서빙을 사용하여 모델을 endpoint로 배포할 수 있습니다.([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/model-serving.html))
# MAGIC 
# MAGIC 아래 코드는 배포된 모델로부터 REST API를 사용하여 예측치를 요청할 수 있도록 할 수 있습니다.

# COMMAND ----------

# MAGIC %md
# MAGIC 좌측 사이드바에서 **Models**를 클릭하고 등록된 와인 모델을 확인합니다. 서빙 탭을 클릭한 후 **Enable Serving**을 클릭합니다.
# MAGIC 
# MAGIC 그리고 **Call The Model** 아래에서 **Python** 버튼을 클릭하여 request를 위한 파이썬 코드륵 확인합니다. 해당 코드를 복사하여 아래 셀에 붙여넣기 합니다. 파이썬 코드는 아래와 같은 형태로 되어 있습니다. 
# MAGIC 
# MAGIC Databricks 노트북이 아니라 외부에서도 토큰값을 사용하여 요청을 수행할 수 있습니다.

# COMMAND ----------

os.environ["DATABRICKS_TOKEN"]

# COMMAND ----------

# 모델 서빙 페이지에서 복사한 코드를 아래에 붙여넣기 하세요
import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://<Databricks-URL>/model/wine_quality/3/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC Endpoint에서 모델이 예측하는 결과는 로컬에서 모델을 평가하는 내용과 동일해야 합니다.

# COMMAND ----------

# 모델 서빙은 작은 사이즈의 배치 데이터에서 low-latency 예측을 위해서 디자인 되었습니다
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])
# 배포된 모델과 학습된 모델의 결과를 비교할 수 있습니다
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions,
})

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

# COMMAND ----------


