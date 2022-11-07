# Databricks notebook source
# MAGIC %md # AutoML 실습
# MAGIC 
# MAGIC [Databricks AutoML](https://docs.databricks.com/applications/machine-learning/automl.html)은 자동으로 머신러닝 모델을 UI를 통해 만들고 코드륵 추출할 수 있습니다. AutoML 기능은 모델 학습을 위한 데이터셋 준비에서부터 HyperOpt와 같은 기능을 활용하여 다양한 모델을 생성, 튜닝, 평가하는 기능을 제공합니다. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 이번 실습에서는 아래와 같은 내용을 학습하게 됩니다:<br>
# MAGIC  - AutoML을 사용하여 자동으로 모델 학습 및 튜닝하기
# MAGIC  - UI환경과 파이썬 환경에서 AutoML 실행하기
# MAGIC  - AutoML run 결과 해석하기
# MAGIC  - 모델레지스트리를 활용하여 production 환경에 모델 배포하기
# MAGIC  - Production 모델 배치 추론하기

# COMMAND ----------

# MAGIC %md 현재 AutoML은 XGBoost와 scikit-learn (싱글노드 모델) 모델을 사용하고 있으며 해당 모델들의 하이퍼파라미터를 최적화하고 있습니다.

# COMMAND ----------

# MAGIC %md ### UI 사용하기
# MAGIC 
# MAGIC 별도로 모델을 프로그래밍 방식으로 만들기 보다는 Databricks에서는 UI상에서 모델을 만들 수 있습니다. 하지만 데이터셋을 테이블로 먼저 등록을 해야 합니다.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

table_name = "wine_quality"

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
data.isna().any()

X = data.drop(["quality"], axis=1)
y = data.quality

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

spark_df = spark.createDataFrame(X_train)
table_path = "dbfs:/hol/delta/wine_data"

dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)
spark_df.write.mode("overwrite").saveAsTable(table_name)
display(spark.sql(f"SELECT * FROM {table_name}"))

# COMMAND ----------

# MAGIC %md 우선 UI를 사용하여 AutoML을 사용하기 위해서는 좌측 사이드바 메뉴에서 `Machine Learning role`로 변경해주어야 합니다. 그리고 사이드바 메뉴에서 `Experiments`로 이동합니다. `Experiments` 화면 좌측 상단에 있는 `Create AutoML Experiment`를 클릭합니다
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_1_2.png" alt="step12" width="750"/>

# COMMAND ----------

# MAGIC %md 클러스터에는 이미 생성한 Databricks ML Runtime으로 설정되어 있는 클러스터를 선택합니다. 그 아래 Problem type에는 `Classification`을 지정해주시고 데이터셋 아래 `Brwose`를 클릭하여 `default` database에 있는 `wine_quality`테이블을 선택합니다.
# MAGIC 
# MAGIC Prediction Target으로는 `is_red` 컬럼을 선택합니다. 여기까지 진행을 하면 자동으로 Experiment name이 생성됩니다. 필요하신 경우 실험명을 변경하셔도 좋습니다.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_UI.png" alt="ui" width="750"/>

# COMMAND ----------

# MAGIC %md Advanced Configuration (Option)을 선택할 경우 비교할 평가지표, 학습할 프레임워크, Stopping 컨디션 등을 지정하실 수 있습니다. 원활한 실습 진행을 위해 적절한 시간 (Timeout)과 시도할 run 횟수를 지정합니다.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoML_Advanced.png" alt="advanced" width="500"/>

# COMMAND ----------

# MAGIC %md 가장 아래에 있는 Start AutoML을 클릭하시면 바로 AutoML을 사용하여 모델을 학습하게 됩니다.
# MAGIC 
# MAGIC 이 부분이 완료가 된 이후 가장 먼저 진행해야 할 부분은 데이터 탐색 노트북(`data exploration notebook`)을 통해 데이터를 분석하는 일입니다.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-1.1.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md 
# MAGIC 그리고 나서 좌측에 있는 최적의 run 노트북 (`notebook for best model`)을 사용하여 향후 어떤 튜닝된 모델이 어떻게 사용되었는지 확인할 수 있습니다.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-1.2.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md ## 최적의 모델 MLflow 리포지토리에 저장하기

# COMMAND ----------

# MAGIC %md 
# MAGIC 다음으로는 최적의 모델을 production으로 배포해보도록 하겠습니다. 우선 AutoML 실험을 통해 성능이 가장 잘 나온 최적의 모델을 선택합니다.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-2.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md 
# MAGIC 클릭한 이후 실험에 대한 상세한 내용을 확인할 수 있습니다. 특히 해당 실험에서 사용되었던 파라미터나 평가지표 결과도 확인할 수 있습니다. 참고로 위 내용처럼 최적을 모델은 run을 실행할 수 있는 노트북이 자동으로 제공됩니다. 이런 경험을 통해 ML 실험에 있어서 생산성과 가시성을 확보할 수 있습니다.
# MAGIC 
# MAGIC 이 단계는 모델을 MLflow 레지스트리에 보내는 역할을 합니다:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-3.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 이 모델을 `automl_lab`라고 명칭하고 만약 해당 명칭이 미미 존재하는 경우 새로운 버전으로 간단히 저장할 수 있습니다.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-4.png" alt="results" width="500"/>
# MAGIC 
# MAGIC 이제 학습한 모델이 MLflow 레지스트리에 저장되었습니다.

# COMMAND ----------

# MAGIC %md ## Production 환경으로 모델 배포하기

# COMMAND ----------

# MAGIC %md
# MAGIC 이제 모델을 Production 환경으로 배포할 차례입니다. 좌측 사이드바 메뉴에서 **Models**을 클릭하고 `automl_lab` 모델을 검색합니다. 그리고 업로드한 최신 버전을 선택합니다. and select the last version you just uploaded.
# MAGIC 
# MAGIC 메뉴에서는 Stage를 변경할 수 있습니다. 아래 이미지와 같이 Stage를 Production으로 바로 transition 하고 커멘트를 작성한 후 저장하세요.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-5.png" alt="results" width="1000"/>
# MAGIC 
# MAGIC 학습된 모델이 이제 production 용으로 준비가 완료되었습니다.

# COMMAND ----------

# MAGIC %md ## 배포한 모델을 사용하여 추론하기

# COMMAND ----------

# MAGIC %md
# MAGIC 이제 모델을 MLflow 레지스트리로부터 로딩하여 전체 테이블을 추론해보도록 하겠습니다.
# MAGIC 
# MAGIC 선택한 Model 내에서 "Use model for inference" 탭을 선택합니다. 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-6.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Databricks는 배치추론과 실시간 추론을 REST API로 제공합니다. 이번 실습에서는 배치모드로 모델을 추론을 진행해보겠습니다 (예를 들어 매일 밤 스코어링을 하는 등).
# MAGIC 
# MAGIC "Production" 모델을 선택하고 자동으로 가장 최신 모델이 뜨게 됩니다. 그리고 Input table로 auto-ml 학습에 사용되었던 데이터셋을 선택합니다.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-7.png" alt="results" width="500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 마무리를 하며
# MAGIC 
# MAGIC 위 단계까지 진행할 경우 최종적으로 배치 추론을 할 수 있는 노트북이 제공이 됩니다. 이제는 이 노트북을 활용하여 매일 밤 스케쥴링을 할 경우 일 단위로 배치 추론이 가능합니다. 또는 DLT 파이프라인으로 연결하여 사용할 수도 있습니다. 여러 실험을 통해 AutoML을 적용해보고 간편하게 노트북을 추출하여 테스트 해보세요. 그리고 만약 실시간 추론이 필요할 경우 Enable Serving을 통해 low-latency를 보장하는 REST API 기반의 endpoint를 생성할 수도 있습니다 (별도의 inferencing 클러스터 생성).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
