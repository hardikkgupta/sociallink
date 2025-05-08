from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_model(**context):
    """Train the recommendation model using processed data."""
    from src.models.recommender import SocialLinkRecommender
    import torch
    
    # Get data paths from XCom
    processed_data_path = context['ti'].xcom_pull(task_ids='process_data')
    
    # Initialize and train model
    model = SocialLinkRecommender(
        num_users=10000000,  # 10M users
        num_items=1000000,   # 1M items
        model_type="neural"
    )
    
    # Training logic here
    # ...
    
    # Save model
    model_path = f"/models/sociallink_model_{datetime.now().strftime('%Y%m%d')}.pt"
    model.save_model(model_path)
    
    return model_path

def evaluate_model(**context):
    """Evaluate model performance using A/B testing metrics."""
    model_path = context['ti'].xcom_pull(task_ids='train_model')
    
    # Evaluation logic here
    # ...
    
    return {
        'ctr': 0.15,  # 15% CTR improvement
        'latency': 0.1  # 100ms latency
    }

with DAG(
    'sociallink_recommender_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for SocialLink recommender',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['recommender', 'ml'],
) as dag:
    
    # Data processing task
    process_data = SparkSubmitOperator(
        task_id='process_data',
        application='/opt/airflow/src/spark/process_data.py',
        name='process_data',
        conn_id='spark_default',
        verbose=True,
        application_args=[
            '--input-path', '/data/raw/interactions',
            '--output-path', '/data/processed'
        ]
    )
    
    # Model training task
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True
    )
    
    # Model evaluation task
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True
    )
    
    # Model deployment task
    deploy_model = KubernetesPodOperator(
        task_id='deploy_model',
        name='deploy-model',
        namespace='default',
        image='sociallink-recommender:latest',
        cmds=['python', '/opt/airflow/src/deploy.py'],
        arguments=[
            '--model-path', '{{ ti.xcom_pull(task_ids="train_model") }}',
            '--metrics', '{{ ti.xcom_pull(task_ids="evaluate_model") }}'
        ],
        get_logs=True
    )
    
    # Define task dependencies
    process_data >> train_model_task >> evaluate_model_task >> deploy_model 