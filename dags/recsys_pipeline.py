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
        result = subprocess.run(
            ["python", "/opt/airflow/spark/jobs/feature_engineering.py"],
            capture_output=True,
            text=True,
            env={**os.environ, "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"},
        )
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise Exception(f"Feature engineering упал: {result.stderr}")
        print("Feature engineering завершён!")

    @task()
    def run_training():
        result = subprocess.run(
            ["python", "/opt/airflow/src/training/train.py"],
            capture_output=True,
            text=True,
            env={**os.environ, "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64"},
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