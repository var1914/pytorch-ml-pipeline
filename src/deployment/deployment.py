from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import logging
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime

from ..minio.minio_init import MinIO


class ModelDeployer:
    """
    Model deployment utilities for saving and loading production models.

    Provides:
    - Model artifact saving to MinIO
    - Model versioning
    - MLflow model registry integration
    - Production model tagging
    """

    def __init__(
        self,
        minio_config: Optional[Dict[str, Any]] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "model_deployment"
    ):
        """
        Initialize the ModelDeployer.

        Args:
            minio_config: MinIO configuration for storing artifacts.
            mlflow_tracking_uri: MLflow tracking server URI.
            mlflow_experiment_name: Name of the MLflow experiment.
        """
        self.logger = self._setup_logger()

        # Setup MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_experiment_name = mlflow_experiment_name

        # Setup MinIO
        self.minio = MinIO(minio_config) if minio_config else None
        self.minio_config = minio_config

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the deployer."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def deploy_model(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        register_to_mlflow: bool = True
    ) -> Dict[str, str]:
        """
        Deploy model to production environment.

        Args:
            model: Trained PyTorch model.
            model_name: Name for the deployed model.
            version: Version string (e.g., 'v1.0.0', '2024-12-08').
            metadata: Additional metadata to store with the model.
            register_to_mlflow: Whether to register model in MLflow Model Registry.

        Returns:
            Dictionary containing deployment information.
        """
        self.logger.info(f"Deploying model: {model_name} (version: {version})")

        deployment_info = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        try:
            # Save to MinIO
            if self.minio:
                minio_path = self._save_to_minio(model, model_name, version, metadata)
                deployment_info['minio_path'] = minio_path

            # Register to MLflow
            if register_to_mlflow:
                mlflow_uri = self._register_to_mlflow(model, model_name, version, metadata)
                deployment_info['mlflow_model_uri'] = mlflow_uri

            deployment_info['status'] = 'success'
            self.logger.info(f"Model deployed successfully: {model_name} v{version}")

        except Exception as e:
            deployment_info['status'] = 'failed'
            deployment_info['error'] = str(e)
            self.logger.error(f"Deployment failed: {str(e)}")
            raise

        return deployment_info

    def _save_to_minio(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model to MinIO storage.

        Returns:
            Path to the saved model in MinIO.
        """
        from io import BytesIO
        import json

        bucket_name = self.minio_config['bucket_name']
        model_path = f"production/{model_name}/{version}/model.pt"

        try:
            # Save model
            model_buffer = BytesIO()
            torch.save(model.state_dict(), model_buffer)
            model_buffer.seek(0)

            self.minio.client.put_object(
                bucket_name=bucket_name,
                object_name=model_path,
                data=model_buffer,
                length=len(model_buffer.getvalue()),
                content_type='application/octet-stream'
            )

            # Save metadata if provided
            if metadata:
                metadata_path = f"production/{model_name}/{version}/metadata.json"
                metadata_buffer = BytesIO(
                    json.dumps(metadata, indent=2).encode('utf-8')
                )
                metadata_buffer.seek(0)

                self.minio.client.put_object(
                    bucket_name=bucket_name,
                    object_name=metadata_path,
                    data=metadata_buffer,
                    length=len(metadata_buffer.getvalue()),
                    content_type='application/json'
                )

            self.logger.info(f"Model saved to MinIO: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Failed to save model to MinIO: {str(e)}")
            raise

    def _register_to_mlflow(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register model to MLflow Model Registry.

        Returns:
            MLflow model URI.
        """
        try:
            with mlflow.start_run(run_name=f"{model_name}_{version}"):
                # Log model
                model_info = mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name
                )

                # Log metadata as parameters
                if metadata:
                    mlflow.log_params(metadata)

                # Tag the version
                mlflow.set_tag("version", version)
                mlflow.set_tag("deployment_timestamp", datetime.now().isoformat())

                self.logger.info(f"Model registered to MLflow: {model_name}")
                return model_info.model_uri

        except Exception as e:
            self.logger.error(f"Failed to register model to MLflow: {str(e)}")
            raise

    def load_model_from_minio(
        self,
        model_class: nn.Module,
        model_name: str,
        version: str,
        device: Optional[str] = None
    ) -> nn.Module:
        """
        Load model from MinIO storage.

        Args:
            model_class: Model class/architecture.
            model_name: Name of the deployed model.
            version: Version string.
            device: Device to load model on.

        Returns:
            Loaded PyTorch model.
        """
        from io import BytesIO

        bucket_name = self.minio_config['bucket_name']
        model_path = f"production/{model_name}/{version}/model.pt"

        try:
            # Download model from MinIO
            response = self.minio.client.get_object(bucket_name, model_path)
            model_buffer = BytesIO(response.read())
            model_buffer.seek(0)

            # Load model
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(model_buffer, map_location=device)

            model = model_class
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self.logger.info(f"Model loaded from MinIO: {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model from MinIO: {str(e)}")
            raise
        finally:
            response.close()
            response.release_conn()

    def promote_to_production(
        self,
        model_name: str,
        version: str
    ) -> None:
        """
        Promote a model version to production in MLflow Model Registry.

        Args:
            model_name: Name of the model.
            version: Version to promote.
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # Get the latest version number from registry
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            if latest_versions:
                model_version = latest_versions[0].version

                # Transition to Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage="Production"
                )

                self.logger.info(f"Model {model_name} v{model_version} promoted to Production")
            else:
                self.logger.warning(f"No model versions found for {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to promote model: {str(e)}")
            raise

    def get_production_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about the current production model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary containing production model information.
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # Get production versions
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])

            if prod_versions:
                version = prod_versions[0]
                info = {
                    'name': version.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp,
                    'source': version.source,
                    'run_id': version.run_id
                }

                self.logger.info(f"Production model info: {info}")
                return info
            else:
                self.logger.warning(f"No production model found for {model_name}")
                return {}

        except Exception as e:
            self.logger.error(f"Failed to get production model info: {str(e)}")
            raise
