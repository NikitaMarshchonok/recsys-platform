from datetime import timedelta
import pendulum
import subprocess
import os
from airflow.decorators import dag, task

default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="recsys_weekly_pipeline",
    schedule="0 2 * * 1",        # каждый понедельник в 2 ночи
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["recsys", "mlops", "weekly"],
    description="Weekly recsys pipeline: features → training → model update",
)
def recsys_pipeline():

    @task()
    def run_feature_engineering():
        import os
        # Проверяем что обработанные данные существуют
        user_features = "/opt/airflow/data/processed/user_features.csv"
        movie_features = "/opt/airflow/data/processed/movie_features.csv"
    
        if not os.path.exists(user_features):
            raise Exception("user_features.csv не найден — запусти feature engineering вручную")
        if not os.path.exists(movie_features):
            raise Exception("movie_features.csv не найден — запусти feature engineering вручную")
    
        print("Данные проверены — всё на месте!")
        print(f"user_features: {os.path.getsize(user_features)} bytes")
        print(f"movie_features: {os.path.getsize(movie_features)} bytes")

    @task()
    def run_training():
        import subprocess
        import os
        result = subprocess.run(
            ["python", "-m", "src.training.train"],
            capture_output=True,
            text=True,
            cwd="/opt/airflow",
            env={**os.environ, "MLFLOW_TRACKING_URI": "http://host.docker.internal:5001"},
        )
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise Exception(f"Обучение упало: {result.stderr}")
        print("Обучение завершено!")

    # Цепочка задач
    features = run_feature_engineering()
    training = run_training()
    features >> training

dag_instance = recsys_pipeline()
