# Databricks notebook source
# MAGIC %md 이 노트북의 목적은 각 스토어의 수 많은 제품별 예측을 Databricks의 분산 컴퓨팅 환경을 사용하여 효율적으로 생성하는 것입니다. 이 노트북은 Spark 3.x 버전을 사용하고 있으며 이전 출시된 Spark 2.x 버전을 토대로 개발되었습니다. 노트북 내에서 **UPDATE** 라고 되어 있는 부분은 Spark 3.x 또는 Databricks 플랫폼의 새로운 기능을 강조하기 위해 코드가 일부 변경된 부분입니다.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 이번 실습을 위해 이 노트북에서는 수요예측에 가장 유명한 라이브러리인 [prophet](https://facebook.github.io/prophet/) 을 사용하게 되고 이를 위해 $pip 매직 커맨드를 활용하여 노트북 세션으로 load합니다.

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

dbutils.fs.cp ('/FileStore/tables/train.csv', '/FileStore/demand_forecast/train/train.csv')

# COMMAND ----------

# MAGIC %fs ls /FileStore/

# COMMAND ----------

# DBTITLE 1,필요 라이브러리 설치
pip install prophet

# COMMAND ----------

# MAGIC %md ## Step 1: 데이터 테스트
# MAGIC 
# MAGIC 이번에 학습할 데이터셋으로 10개의 다른 상점의 50개 다른 아이템으로부터 수집된 5년치의 상점-아이템별 개별 판매 데이터를 사용합니다. 이 데이터는 에전  Kaggle competition에서 활용되어 public하게 사용할 수 있는 데이터이자 원본은 [여기](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)에서 다운 받으실 수 있습니다. (Kaggle 계정이 있는 경우 바로 다운로드 가능)
# MAGIC 
# MAGIC 하지만 이번 실습 편의상 해당 데이터를 [여기](https://sbangdemo.blob.core.windows.net/retailhol/train.csv?si=Reader&spr=https&sv=2021-06-08&sr=b&sig=kXGcZU9TqylAuz3SJGlWMFVV9pMTm5EjaUs5ZM5Q%2BTA%3D)에서 다운 받으실 수 있습니다. 해당 데이터는 이번 Workshop을 위해서만 사용되기 때문에 이후 개인적인 목적으로 실습 데이터를 다운로드 받지 말아주세요.
# MAGIC 
# MAGIC 데이터를 다운로드 받은 이후 */FileStore/demand_forecast/train/* 위치에 파일을 import 하도록 하겠습니다. ([관련내용](https://docs.databricks.com/data/databricks-file-system.html#!#user-interface))

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/

# COMMAND ----------

# DBTITLE 1,데이터 접근
from pyspark.sql.types import *

# 학습용 데이터 스키마
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# 학습할 파일 데이터프레임으로 Read
train = spark.read.csv(
  'dbfs:/FileStore/tables/train.csv', 
  header=True, 
  schema=train_schema
  )

# 데이터프레임을 쿼리할 수 잇는 temporary view로 변환
train.createOrReplaceTempView('train')

# 데이터 조회
display(train)

# COMMAND ----------

# MAGIC %md 수요 예측을 진행할 때는 보통 일반적인 트렌드나 시즌별 트렌드에 대한 관심이 많습니다. 우선 유닛별 판매의 연간 트렌드로 탐색해보도록 하겠습니다.

# COMMAND ----------

# DBTITLE 1,연간 트렌드 확인
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md 데이터를 확인할 경우 전체 판매 개수는 우상향하고 있는 것을 볼 수 있습니다. 만약 각 상점별 마켓에 대한 지식이 있다면 어느 영역에서 가장 많은 growth가 일어났는지 알고 싶을 것이고 지속적으로 예측을 하고 싶을 것입니다. 하지만 데이터에 대한 지식이 없고 데이터셋을 대충 봤을 때에는 일간, 월간, 연간 예측만으로 시간이 지남에 따라 선형적으로 증가하는 것만 기대할 수 있습니다. 
# MAGIC 
# MAGIC 그렇다면 이번에는 시즌별 트렌드에 대해 확인해보겠습니다. 이번에는 년별 각 월의 데이터를 합치게 되면 확실히 시즌별 패턴이 있는 것을 확인할 수 있고 이 트렌드가 연간 조금씩 늘어나고 있는 추세를 확인할 수 있습니다.

# COMMAND ----------

# DBTITLE 1,월간 트렌드 확인
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md 데이터를 합쳐서 주간으로 확인했을 경우 일요일 (weekday가 0인 경우) 가장 높은 수치를 보이고 월요일 (weekday가 1인 경우) 급작스럽게 떨어지는 것을 확인할 수 있고 월요일 이후 일요일까지 점처 증가하는 추세를 보인다는 것을 확인할 수 있습니다. 이런 트렌드는 보통 5년동안 안정적으로 유지되는 것을 확인할 수 있습니다.
# MAGIC 
# MAGIC **UPDATE** Spark 3 버전이 나옴에 따라 CAST(DATE_FORMAT(date, 'u')에서 'u' 옵션은 제거되었습니다. ([Proleptic Gregorian calendar](https://databricks.com/blog/2020/07/22/a-comprehensive-look-at-dates-and-timestamps-in-apache-spark-3-0.html)) 이제는 'E' 옵션을 통해 유사한 결과를 확인하실 수 있습니다.

# COMMAND ----------

# DBTITLE 1,주간 트렌드 확인
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md 이제 데이터의 기본적인 패턴에 대해서 확인했기 때문에 어떻게 모델을 만들면 좋을지 탐색해보도록 하겠습니다.

# COMMAND ----------

# MAGIC %md ## Step 2: 하나의 예측 모델 만들기
# MAGIC 
# MAGIC 한꺼번에 여러 상점 및 아이템에 대한 조합의 예측 모델을 시도하기 전에 우선은 하나의 예측 모델부터 잘 만들기 위해 prophet을 적용해보도록 하겠습니다.
# MAGIC 
# MAGIC 첫 단계에서는 모델을 학습할 historical 데이터를 조합해보도록 하겠습니다. 

# COMMAND ----------

# DBTITLE 1,하나의 아이템-상점 조합의 데이터 가지고 오기
# 날짜 기준 병합된 데이터 쿼리하기
sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    sales as y
  FROM train
  WHERE store=1 AND item=1
  ORDER BY ds
  '''

# 데이터셋 Pandas 데이터프레임으로 변환
history_pd = spark.sql(sql_statement).toPandas()

# 누락된 record가 있을 경우 데이터 drop
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md 이번에는 prophet 라이브러리를 import하겠습니다. 하지만 prophet을 사용할 때 메세지가 너무 많을 수 있기 때문에 환경의 logging 세팅을 변경해주도록 하겠습니다.

# COMMAND ----------

# DBTITLE 1,Prophet 라이브러리 Import
from prophet import Prophet
import logging

# prophet에서 제공하는 정보성 메세지 해제
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md Step 1에서 확인했던 내용처럼 전반적인 growth 패턴은 선형적이고 주간 및 연간 시즌 패턴이 존재했습니다. 따라서 모델 학습 시 seasonality mode를 'multiplicative'로 두고 시즌별 패턴을 따르면서 growth도 반영할 수 있도록 합니다.

# COMMAND ----------

# DBTITLE 1,Train Prophet Model
# 모델 파라미터 세팅
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# historical 데이터에 모델 fit
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md 이제 모델을 학습했습니다. 다음으로는 90일을 예측해보도록 하겠습니다.

# COMMAND ----------

# DBTITLE 1,예측 생성
# historical 날짜와 90일 이후의 날짜를 포함한 데이터셋 생성
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# 해당 데이터셋의 예측
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# MAGIC %md 모델이 얼마나 좋은 성능을 냈을까요? 아래에서는 모델에서 생성하고 그래프로 표현한 일반적인 트렌드 및 시즌별 트렌드를 보여주고 있습니다.

# COMMAND ----------

# DBTITLE 1,예측 결과 시각화
trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

# MAGIC %md 그리고 아래와 같이 실제와 예측한 데이터가 잘 맞는지 그리고 향후 예측은 어떻게 될지 확인할 수 있습니다. 아래 예시에서는 확인해보고 쉽게 가장 최근 1년만 표현하겠습니다.

# COMMAND ----------

# DBTITLE 1,Historical 데이터 및 예측 비교
predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')

# 표현하고자 하는 날짜를 작년 + 앞으로의 90일 예측으로 조정
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

# MAGIC %md **NOTE** 이 시각화는 조금 난해합니다. Bartosz Mikulski는 [an excellent breakdown](https://www.mikulskibartosz.name/prophet-plot-explained/)이라는 글에서 해당 시각화에 대한 설명을 잘 해주고 있어 확인 해볼만합니다. 요약해서 검정색 점은 실제 데이터를 의미하고 짙은 파란색 선은 생성한 예측치를 나타내며 옅은 파란색 영역은 95% 불확실성 interval을 포함한 예측치입니다.

# COMMAND ----------

# MAGIC %md 시각화 자료도 매우 중요하지만 예측 모델을 더 나은 방법으로 평가하기 위해서는 Mean Absolute Error (MAE). Mean Squared Error (MSE) 또는 Root Mean Squared Error (RMSE)와 같은 방법을 사용하는게 좋습니다.

# COMMAND ----------

# DBTITLE 1,평가지표 계산
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# 비교를 위한 Historical 데이터 및 예측 값 데이터프레임화
actuals_pd = history_pd[ history_pd['ds'] < date(2018, 1, 1) ]['y']
predicted_pd = forecast_pd[ forecast_pd['ds'] < pd.to_datetime('2018-01-01') ]['yhat']

# 평가지표 계산
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

# 평가지표 display
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md Prophet 모델은 시간이 지남에 따라 예측이 어떻게 되고 있는지 등을 평가하는 기능도 제공합니다 ([additional means](https://facebook.github.io/prophet/docs/diagnostics.html)). 이러한 기능을 사용하는 것이 권장되지만 이번 실습에서는 다음 step의 주제인 여러 모델로의 확장에 집중하기 위해 넘어가도록 하겠습니다.

# COMMAND ----------

# MAGIC %md ## Step 3: 규모있는 예측 생성
# MAGIC 
# MAGIC 기본적인 예측하는 방법에 대해 만들어봤으니 이제 본격적으로 본래 목적인 각 상점, 아이템별 잘 만들어진 예측 모델을 다량 생성해보도록 하겠습니다. 우선 현재 데이터를 상점-아이템-날짜 레벨로 만들도록 하겠습니다.
# MAGIC 
# MAGIC **NOTE**: 이미 해당 데이터셋은 이 정도 수준으로 준비가 되어 있겠지만 다시 한 번 더 처리하여 원하는 형식의 데이터 구조를 가지도록 하겠습니다.

# COMMAND ----------

# DBTITLE 1,상점-아이템별 좋바으로 데이터 추출
sql_statement = '''
  SELECT
    store,
    item,
    CAST(date as date) as ds,
    SUM(sales) as y
  FROM train
  GROUP BY store, item, ds
  ORDER BY store, item, ds
  '''

store_item_history = (
  spark
    .sql( sql_statement )
    .repartition(sc.defaultParallelism, ['store', 'item'])
  ).cache()

# COMMAND ----------

# MAGIC %md 데이터를 상점-아이템-날짜 레벨별로 합치고난 이후 해당 데이터를 어떻게 prophet으로 보낼지 고민해야 합니다. 만약 목표가 각 상점 및 아이템 조합별 모델을 생성하는 것이라면 방금 생성한 상점-아이템 subset의 데이터를 모델 학습을 위해 보내고 예측 값을 받을 수 있어야 합니다. 예측을 한 이후의 데이터셋은 상점과 아이템의 식별자가 있는채로 만들어져야 하며 Prophet모델을 통해 만들어진 필드와 함께 subset 데이터에 표현되어야 합니다.

# COMMAND ----------

# DBTITLE 1,예측  Output을 위한 스키마 정의
from pyspark.sql.types import *

result_schema =StructType([
  StructField('ds',DateType()),
  StructField('store',IntegerType()),
  StructField('item',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

# MAGIC %md 모델 학습과 예측을 생성하기 위해서 아래 코드에서는 Pandas function을 사용합니다. 해당 함수를 사용하여 상점 및 아이템별 데이터를 전달 받고 바로 위 Cell에서 정의한 포맷대로 예측값을 반환합니다.
# MAGIC 
# MAGIC **UPDATE** Spark 3.0 부터는 Pandas UDF의 기능을 Pandas function dl eocpgkqslek. Pandas UDF 문법은 여전히 제공되겠지만 결국 사라지게 될 예정입니다. 새로운 Pandas function API를 확인하기 위해서는 [여기](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html)를 참고해주세요.

# COMMAND ----------

# DBTITLE 1,모델 학습 및 예측 생성 함수 정의
def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # 이전과 같이 모델 학습
  # --------------------------------------
  # 누락값 처리 (날짜-상점-아이템 기준)
  history_pd = history_pd.dropna()
  
  # 모델을 위한 설정
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # 모델 학습
  model.fit( history_pd )
  # --------------------------------------
  
  # 이전과 같이 예측 생성
  # --------------------------------------
  # 예측
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # 결과 데이터넷 취합
  # --------------------------------------
  # 예측에서 필요한 필드 값만 취합
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # historical 데이터에서 필요한 필드 값만 취합
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # historical 데이터와 예측 데이터 join
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # 상점 / 아이템별 정보 조회
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # 결과 데이터셋 반환
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

# MAGIC %md 함수형식으로 보게될 경우 일반적인 코드 보다 복잡해보일 수도 있습니다. 하지만 실제 첫 두 번째 코드 block인 학습과 예측 부분을 보면 Step 2에서 사용한 코드와 다르지 않다는 것을 알 수 있습니다. 새로운 코드에서는 이러한 예측 결과 값을 조합하는 부분만 사용하고 있고 표준 Pandas dataframe 전환하는 부분을 사용하기 때문에 실제로는 어렵지 않습니다.

# COMMAND ----------

# MAGIC %md 이제 Pandas function을 호출하여 예측을 생성해보도록 하겠습니다. 실습에서는 이러한 부분을 상점과 아이템을 기준으로 grouping한 historical 데이터셋을 사용합니다. 그리고 각 그룹에 function을 적용하여 오늘의 날짜를 *training_date*로 지정하여 데이터 관리 목적으로 활용하겠습니다.
# MAGIC 
# MAGIC **UPDATE** 이전 업데이트 내용과 유사하게 이제는 Pandas UDF가 아닌 applyInPandas()라는 Pandas 함수를 사용하여 호출합니다.

# COMMAND ----------

# DBTITLE 1,각 상점-아이템 조합에 예측 함수 적용
from pyspark.sql.functions import current_date

results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema=result_schema)
    .withColumn('training_date', current_date() )
    )

results.createOrReplaceTempView('new_forecasts')

display(results)

# COMMAND ----------

# MAGIC %md 이러한 예측 결과는 리포트를 위해 쿼리해볼 수 있는 테이블 형식의 구조로 변환하겠습니다.

# COMMAND ----------

# DBTITLE 1,명칭 충돌 방지를 위한 사용자 기반 database명 지정
import re
from pathlib import Path
# 사용자별 path 및 database명 정의
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_sql_compatible = re.sub('\W', '_', useremail.split('@')[0])
tmp_data_path = f"/tmp/fine_grain_forecast/data/{useremail}/"
database_name = f"fine_grain_forecast_{username_sql_compatible}"

# 사용자 기반 환경 생성
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name} LOCATION '{tmp_data_path}'")
spark.sql(f"USE {database_name}")
Path(tmp_data_path).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,예측 Output 유지
# MAGIC %sql
# MAGIC -- 예측 테이블 생성
# MAGIC create table if not exists forecasts (
# MAGIC   date date,
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   sales float,
# MAGIC   sales_predicted float,
# MAGIC   sales_predicted_upper float,
# MAGIC   sales_predicted_lower float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC -- 테이블에 데이터 load
# MAGIC insert overwrite forecasts
# MAGIC select 
# MAGIC   ds as date,
# MAGIC   store,
# MAGIC   item,
# MAGIC   y as sales,
# MAGIC   yhat as sales_predicted,
# MAGIC   yhat_upper as sales_predicted_upper,
# MAGIC   yhat_lower as sales_predicted_lower,
# MAGIC   training_date
# MAGIC from new_forecasts;

# COMMAND ----------

# MAGIC %md 하지만 각 예측이 얼마나 잘(또는 안)되었을까요? Pandas function을 활용하여 각 상점-아이템별 팡가 지표도 아래와 같이 생성할 수 있습니다.

# COMMAND ----------

# DBTITLE 1,동일한 방법을 활용하여 예측 평가
# 결과 데이터셋 스키마 정의
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# 지표 계산을 위한 함수 정의
def evaluate_forecast( evaluation_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # 상점 및 아이템 정보 조회
  training_date = evaluation_pd['training_date'].iloc[0]
  store = evaluation_pd['store'].iloc[0]
  item = evaluation_pd['item'].iloc[0]
  
  # 평가 지표 계산
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # 결과 데이터 조합
  results = {'training_date':[training_date], 'store':[store], 'item':[item], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# 지표 계산
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # limit evaluation to periods where we have historical data
    .select('training_date', 'store', 'item', 'y', 'yhat')
    .groupBy('training_date', 'store', 'item')
    .applyInPandas(evaluate_forecast, schema=eval_schema)
    )

results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------

# MAGIC %md 다시 한 번 더, 각 예측에 대한 지표를 유지하기 위해 쿼리할 수 있는 테이블 형식으로 저장하도록 하겠습니다.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from forecast_evals

# COMMAND ----------

# DBTITLE 1,평가지표 유지
# MAGIC %sql
# MAGIC 
# MAGIC create table if not exists forecast_evals (
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   mae float,
# MAGIC   mse float,
# MAGIC   rmse float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC insert overwrite forecast_evals
# MAGIC select
# MAGIC   store,
# MAGIC   item,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse,
# MAGIC   training_date
# MAGIC from new_forecast_evals;

# COMMAND ----------

# MAGIC %md 이제 상점, 아이템 조합별 예측과 기본적인 평가 지표를 만들었습니다. 이러한 예측 데이터를 확인하기 위해서는 아래와 같이 간단하게 쿼리할 수 있습니다. (아래의 경우 제품1에 대해 상점 1에서 3까지의 시각화)

# COMMAND ----------

# DBTITLE 1,예측 시각화
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md 그리고 이러한 개별 예측은 평가지표를 추출하여 각 모델에 대한 신뢰도를 확인할 수 있습니다.

# COMMAND ----------

# DBTITLE 1,평가지표 추출
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM forecast_evals a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store
