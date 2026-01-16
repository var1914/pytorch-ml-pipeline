"""
CV Pipeline Airflow DAG

A config-driven ML pipeline for computer vision tasks.
Supports classification, detection, and segmentation workflows.

Usage:
    1. Create a config file (e.g., configs/templates/medical_imaging.yaml)
    2. Set environment variable: CV_CONFIG_PATH=/path/to/config.yaml
    3. Trigger the DAG

The pipeline executes 4 stages:
    Data Preparation → Model Training → Model Evaluation → Model Deployment
"""

from datetime import datetime, timedelta
import os
import sys
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data.data_loading import DataDownloader
from src.data.preprocessing import DataPreprocessor
from src.models import ResNetClassifier, ClassificationModel
from src.training.training import ModelTrainer
from src.evaluation.evaluation import ModelEvaluator
from src.deployment.deployment import ModelDeployer

# Pipeline Configuration (can be overridden via environment variables)
PIPELINE_CONFIG = {
    # Dataset settings
    'dataset_name': os.getenv('CV_DATASET_NAME', 'cifar10'),
    'data_root': os.getenv('CV_DATA_ROOT', '../data'),
    'num_classes': int(os.getenv('CV_NUM_CLASSES', '10')),

    # Training settings
    'batch_size': int(os.getenv('CV_BATCH_SIZE', '32')),
    'num_epochs': int(os.getenv('CV_NUM_EPOCHS', '10')),
    'learning_rate': float(os.getenv('CV_LEARNING_RATE', '0.001')),
    'early_stopping_patience': int(os.getenv('CV_EARLY_STOPPING', '5')),

    # Model settings
    'model_architecture': os.getenv('CV_MODEL_ARCH', 'resnet50'),
    'pretrained': os.getenv('CV_PRETRAINED', 'true').lower() == 'true',

    # Infrastructure
    'minio_endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    'minio_access_key': os.getenv('MINIO_ACCESS_KEY', 'admin'),
    'minio_secret_key': os.getenv('MINIO_SECRET_KEY', 'admin123'),
    'minio_bucket': os.getenv('MINIO_BUCKET', 'ml-models'),
    'mlflow_tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
}

# MinIO configuration dict
MINIO_CONFIG = {
    'endpoint': PIPELINE_CONFIG['minio_endpoint'],
    'access_key': PIPELINE_CONFIG['minio_access_key'],
    'secret_key': PIPELINE_CONFIG['minio_secret_key'],
    'secure': False,
    'bucket_name': PIPELINE_CONFIG['minio_bucket']
}


def get_dataset_class(dataset_name: str):
    """Get torchvision dataset class by name."""
    dataset_map = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
        'fashionmnist': datasets.FashionMNIST,
        'pcam': datasets.PCAM,
        'svhn': datasets.SVHN,
        'stl10': datasets.STL10,
    }
    name_lower = dataset_name.lower()
    if name_lower not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    return dataset_map[name_lower]


def get_transforms(dataset_name: str, train: bool = True):
    """Get appropriate transforms for a dataset."""
    # Default ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


def data_preparation(**context):
    """
    Data preparation task:
    1. Downloads dataset
    2. Validates data quality
    3. Checks class balance
    4. Prepares data loaders
    """
    try:
        config = PIPELINE_CONFIG
        download_data = DataDownloader()
        preprocessor = DataPreprocessor()

        dataset_class = get_dataset_class(config['dataset_name'])
        data_root = config['data_root']

        logger.info(f"Preparing dataset: {config['dataset_name']}")

        # Load datasets with basic transform for validation
        basic_transform = transforms.Compose([transforms.ToTensor()])

        # Handle different dataset APIs
        try:
            # Try split-based API (PCAM style)
            train_dataset = dataset_class(
                root=data_root,
                split='train',
                download=True,
                transform=basic_transform
            )
            val_dataset = dataset_class(
                root=data_root,
                split='test',
                download=True,
                transform=basic_transform
            )
        except TypeError:
            # Fall back to train/test API (CIFAR style)
            train_dataset = dataset_class(
                root=data_root,
                train=True,
                download=True,
                transform=basic_transform
            )
            val_dataset = dataset_class(
                root=data_root,
                train=False,
                download=True,
                transform=basic_transform
            )

        # Compute statistics
        logger.info("Computing dataset statistics...")
        download_data.dataset = train_dataset
        train_size, train_label_dist = download_data.get_data_stats()
        logger.info(f"Train size: {train_size}, Label distribution: {train_label_dist}")

        # Validate datasets
        logger.info("Validating dataset quality...")
        train_validation = preprocessor.validate_dataset(train_dataset)
        logger.info(f"Train validation: {train_validation}")

        # Check class balance
        train_balance = preprocessor.check_class_balance(train_dataset)
        logger.info(f"Class balance: {train_balance}")

        # Push results to XCom
        context['task_instance'].xcom_push(key='data_root', value=data_root)
        context['task_instance'].xcom_push(key='dataset_name', value=config['dataset_name'])
        context['task_instance'].xcom_push(key='train_size', value=train_size)
        context['task_instance'].xcom_push(key='train_balance', value=train_balance)

        logger.info("Data preparation completed successfully!")

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise


def model_training(**context):
    """Train the model on prepared data."""
    try:
        config = PIPELINE_CONFIG

        # Pull from XCom
        data_root = context['task_instance'].xcom_pull(task_ids='data_preparation', key='data_root')
        dataset_name = context['task_instance'].xcom_pull(task_ids='data_preparation', key='dataset_name')

        logger.info(f"Training model: {config['model_architecture']}")

        dataset_class = get_dataset_class(dataset_name)
        train_transform = get_transforms(dataset_name, train=True)
        val_transform = get_transforms(dataset_name, train=False)

        # Load datasets
        try:
            train_dataset = dataset_class(root=data_root, split='train', download=False, transform=train_transform)
            val_dataset = dataset_class(root=data_root, split='test', download=False, transform=val_transform)
        except TypeError:
            train_dataset = dataset_class(root=data_root, train=True, download=False, transform=train_transform)
            val_dataset = dataset_class(root=data_root, train=False, download=False, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Create model
        model = ResNetClassifier(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
        logger.info(f"Model parameters: {model.get_num_params():,}")

        # Initialize trainer
        experiment_name = f"{dataset_name}_{config['model_architecture']}"
        model_trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=config['mlflow_tracking_uri'],
            mlflow_experiment_name=experiment_name,
            lr=config['learning_rate'],
            checkpoint_dir='./checkpoints'
        )

        # Train
        model_trainer.train(
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['early_stopping_patience'],
            save_best_only=True,
            log_to_mlflow=True
        )

        # Push to XCom
        context['task_instance'].xcom_push(key='best_model_path', value='./checkpoints/best_model.pt')
        context['task_instance'].xcom_push(key='data_root', value=data_root)
        context['task_instance'].xcom_push(key='dataset_name', value=dataset_name)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


def model_evaluation(**context):
    """Evaluate the trained model."""
    try:
        config = PIPELINE_CONFIG

        # Pull from XCom
        best_model_path = context['task_instance'].xcom_pull(task_ids='model_training', key='best_model_path')
        data_root = context['task_instance'].xcom_pull(task_ids='model_training', key='data_root')
        dataset_name = context['task_instance'].xcom_pull(task_ids='model_training', key='dataset_name')

        logger.info("Evaluating model...")

        # Load model
        model = ResNetClassifier(num_classes=config['num_classes'], pretrained=False)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load validation data
        dataset_class = get_dataset_class(dataset_name)
        val_transform = get_transforms(dataset_name, train=False)

        try:
            val_dataset = dataset_class(root=data_root, split='test', download=False, transform=val_transform)
        except TypeError:
            val_dataset = dataset_class(root=data_root, train=False, download=False, transform=val_transform)

        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Evaluate
        experiment_name = f"{dataset_name}_evaluation"
        evaluator = ModelEvaluator(
            model=model,
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=config['mlflow_tracking_uri'],
            mlflow_experiment_name=experiment_name
        )

        eval_metrics = evaluator.evaluate(
            test_loader=val_loader,
            log_to_mlflow=True,
            save_visualizations=True
        )

        classification_rep = evaluator.get_classification_report(val_loader)
        logger.info(f"\nClassification Report:\n{classification_rep}")

        context['task_instance'].xcom_push(key='eval_metrics', value=eval_metrics)

        logger.info("Model evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def model_deployment(**context):
    """Deploy model to production."""
    try:
        config = PIPELINE_CONFIG

        # Pull from XCom
        best_model_path = context['task_instance'].xcom_pull(task_ids='model_training', key='best_model_path')
        eval_metrics = context['task_instance'].xcom_pull(task_ids='model_evaluation', key='eval_metrics')
        dataset_name = context['task_instance'].xcom_pull(task_ids='model_training', key='dataset_name')

        logger.info("Deploying model...")

        # Load model
        model = ResNetClassifier(num_classes=config['num_classes'], pretrained=False)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Deploy
        experiment_name = f"{dataset_name}_deployment"
        deployer = ModelDeployer(
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=config['mlflow_tracking_uri'],
            mlflow_experiment_name=experiment_name
        )

        model_name = f"cv_{config['model_architecture']}_{dataset_name}"
        version = datetime.now().strftime('%Y%m%d_%H%M%S')

        deployment_info = deployer.deploy_model(
            model=model,
            model_name=model_name,
            version=version,
            metadata={
                'eval_metrics': eval_metrics,
                'deployment_date': datetime.now().isoformat(),
                'model_architecture': config['model_architecture'],
                'dataset': dataset_name,
                'num_classes': config['num_classes']
            },
            register_to_mlflow=True
        )

        logger.info(f"Model deployed: {deployment_info}")
        context['task_instance'].xcom_push(key='deployment_info', value=deployment_info)

        logger.info("Model deployment completed successfully!")

    except Exception as e:
        logger.error(f"Error in model deployment: {str(e)}")
        raise


# DAG Definition
default_args = {
    'owner': 'cv_pipeline',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cv_ml_pipeline',
    description='Generic CV Pipeline for Classification, Detection, and Segmentation',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['cv', 'ml', 'pytorch']
)

# Define tasks
task_data_preparation = PythonOperator(
    task_id='data_preparation',
    python_callable=data_preparation,
    dag=dag,
    provide_context=True
)

task_model_training = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
    provide_context=True
)

task_model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
    provide_context=True
)

task_model_deployment = PythonOperator(
    task_id='model_deployment',
    python_callable=model_deployment,
    dag=dag,
    provide_context=True
)

# Task dependencies
task_data_preparation >> task_model_training >> task_model_evaluation >> task_model_deployment