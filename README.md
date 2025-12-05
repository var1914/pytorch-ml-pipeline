# PCAM Baseline Model Training

## Overview
This is a baseline model for the PCAM (PatchCamelyon) dataset using ResNet50. The goal is to build a simple, working model that we can later integrate into a full ML pipeline with Airflow, MLflow, and Kubernetes.

## Dataset
The PCAM dataset is a binary classification task on 96x96 histopathology image patches. The task is to identify whether a patch contains tumor tissue or not.

- Train samples: 131,072
- Val samples: 32,768
- Test samples: 32,768

For quick iteration, we use a subset of 10,000 training samples.

## Model Architecture
- **Base model**: ResNet50 (pretrained weights not used initially)
- **Input**: 96x96 RGB images
- **Output**: 2 classes (tumor / non-tumor)
- **Training**: Binary cross-entropy loss with Adam optimizer

## Prerequisites

### Infrastructure Setup
Before running the training, ensure the following services are running:

1. **MinIO** (Object Storage)
```bash
   # Using Docker
   docker run -d \
     -p 9090:9000 \
     -p 9091:9091 \
     --name minio \
     -e "MINIO_ROOT_USER=admin" \
     -e "MINIO_ROOT_PASSWORD=admin123" \
     minio/minio server /data --console-address ":9091"
```
   Access MinIO Console at: http://localhost:9091

2. **MLflow** (Experiment Tracking)
```bash
   # Using pip
   pip install mlflow
   
   # Start MLflow server
   mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlflow-artifacts \
     --host 0.0.0.0 \
     --port 5000
```
   Access MLflow UI at: http://localhost:5000

## Project Structure
```
pcam-baseline/
├── src/
│   ├── data/
│   │   └── data_loader.py        # DataDownloader class
│   ├── models/
│   │   └── model.py               # PCamModel class
│   └── training/
│       └── training.py            # ModelTrainer class
├── data/                          # Downloaded datasets (auto-created)
├── checkpoints/                   # Model checkpoints (auto-created)
├── requirements.txt
└── README.md
```

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
matplotlib>=3.7.0
minio>=7.1.0
mlflow>=2.8.0
```

## How to Run

### Step 1: Import Dependencies
```python
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from src.data.data_loader import DataDownloader
from src.models.model import PCamModel
from src.training.training import ModelTrainer
```

### Step 2: Define Data Transforms
```python
# Define transforms with ImageNet normalization
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert to PyTorch Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])
```

### Step 3: Load and Explore Dataset
```python
# Initialize data downloader
download_data = DataDownloader()

# Load training data
train_dataset = download_data.load_data(
    datasets.PCAM, 
    '../data',
    split='train',
    download=True,
    transform=transform
)

# Get dataset statistics
dataset_size, label_distribution = download_data.get_data_stats()
print(f"Dataset size: {dataset_size}")
print(f"Label distribution: {label_distribution}")

# Visualize sample images (6 samples in 2x3 grid)
download_data.visualize_samples(
    num_samples=6, 
    nrows=2, 
    ncols=3,
    figsize=(12, 8)
)

# Load validation data
val_dataset = download_data.load_data(
    datasets.PCAM, 
    '../data',
    split='val',
    download=True,
    transform=transform
)
```

### Step 4: Create Training Subsets
```python
# Create subset for faster iteration (10k training samples)
SUBSET_SIZE = 10000
subset_indices = torch.randperm(len(train_dataset))[:SUBSET_SIZE].tolist()
train_subset = Subset(train_dataset, subset_indices)

# Create proportional validation subset (20% of training subset)
val_subset_size = int(SUBSET_SIZE * 0.2)
val_indices = torch.randperm(len(val_dataset))[:val_subset_size].tolist()
val_subset = Subset(val_dataset, val_indices)

print(f"\nSubset sizes (for training):")
print(f"  Train subset: {len(train_subset)}")
print(f"  Val subset: {len(val_subset)}")
```

### Step 5: Create Data Loaders
```python
# Create data loaders with batching
train_loader = DataLoader(
    train_subset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

val_loader = DataLoader(
    val_subset, 
    batch_size=32, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### Step 6: Initialize Model
```python
# Create ResNet50 model for binary classification
model = PCamModel(num_classes=2, pretrained=False)
print(f"Model created: ResNet50 with {model.num_classes} output classes")

# Check model parameters
print(f"Total parameters: {model.get_num_params():,}")
print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
```

### Step 7: Configure MLOps Infrastructure
```python
# MinIO configuration for model storage
minio_config = {
    'endpoint': 'localhost:9090',
    'access_key': 'admin',
    'secret_key': 'admin123',
    'secure': False,
    'bucket_name': 'ml-models'
}

# Initialize trainer with MLflow tracking
model_trainer = ModelTrainer(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    minio_config=minio_config, 
    mlflow_experiment_name='pcam_resnet50_model',
    lr=0.001,
    checkpoint_dir='./checkpoints'  # Optional local checkpoints
)
```

### Step 8: Train Model
```python
# Train for 5 epochs with early stopping
history = model_trainer.train(
    num_epochs=5,
    early_stopping_patience=3,  # Stop if no improvement for 3 epochs
    save_best_only=True,        # Only save best model
    log_to_mlflow=True          # Enable MLflow tracking
)

# Print training history
print("\nTraining History:")
print(f"Best Train Loss: {min(history['train_loss']):.4f}")
print(f"Best Val Loss: {min(history['val_loss']):.4f}")
print(f"Best Val Accuracy: {max(history['val_acc']):.4f}")
```

### Step 9: Load and Evaluate Best Model (Optional)
```python
# Load best model from MinIO
best_model_path = "models/20241205_120000/model.pt"  # Use actual timestamp
model_trainer.load_from_minio(best_model_path)

# Or load from local checkpoint
checkpoint = model_trainer.load_local_checkpoint('./checkpoints/best_model.pt')
print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
```

## Complete Training Script

You can also run the complete training pipeline with a single script:
```python
# train.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from src.data.data_loader import DataDownloader
from src.models.model import PCamModel
from src.training.training import ModelTrainer

def main():
    # Configuration
    SUBSET_SIZE = 10000
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    print("Loading PCAM dataset...")
    download_data = DataDownloader()
    
    train_dataset = download_data.load_data(
        datasets.PCAM, '../data', split='train',
        download=True, transform=transform
    )
    
    val_dataset = download_data.load_data(
        datasets.PCAM, '../data', split='val',
        download=True, transform=transform
    )
    
    # Create subsets
    train_indices = torch.randperm(len(train_dataset))[:SUBSET_SIZE].tolist()
    val_indices = torch.randperm(len(val_dataset))[:int(SUBSET_SIZE*0.2)].tolist()
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = PCamModel(num_classes=2, pretrained=False)
    
    # Configure MLOps
    minio_config = {
        'endpoint': 'localhost:9090',
        'access_key': 'admin',
        'secret_key': 'admin123',
        'secure': False,
        'bucket_name': 'ml-models'
    }
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        minio_config=minio_config,
        mlflow_experiment_name='pcam_resnet50_model',
        lr=LEARNING_RATE,
        checkpoint_dir='./checkpoints'
    )
    
    # Train
    print("Starting training...")
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=3,
        save_best_only=True
    )
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python train.py
```

## Monitoring and Results

### View Experiments in MLflow
1. Open http://localhost:5000
2. Navigate to the "pcam_resnet50_model" experiment
3. View metrics, parameters, and artifacts for each run

### View Models in MinIO
1. Open http://localhost:9091
2. Login with `admin` / `admin123`
3. Browse the `ml-models` bucket
4. Download models and metadata files

### Expected Results
After 5 epochs on 10k training samples:
- Train Accuracy: ~82%
- Val Accuracy: ~83%
- Test Accuracy: ~83%

## Troubleshooting

### MinIO Connection Error
```bash
# Check if MinIO is running
docker ps | grep minio

# Restart MinIO if needed
docker restart minio
```

### MLflow Connection Error
```bash
# Check if MLflow is running
ps aux | grep mlflow

# Restart MLflow
mlflow server --host 0.0.0.0 --port 5000
```

### CUDA Out of Memory
```python
# Reduce batch size
train_loader = DataLoader(train_subset, batch_size=16, ...)  # Instead of 32
```

### Slow Data Loading
```python
# Increase num_workers
train_loader = DataLoader(train_subset, num_workers=8, ...)  # Instead of 4
```

## What's Next
This baseline is the foundation. Next steps:

1. ✅ **MLflow integration** - Track experiments and metrics
2. ✅ **MinIO integration** - Store model artifacts
3. 🔄 **Airflow DAG** - Orchestrate the ML pipeline
4. 🔄 **Kubeflow Pipeline** - Deploy on Kubernetes
5. 🔄 **Model Serving** - Deploy with KServe/Seldon
6. 🔄 **Full Dataset Training** - Scale to 131k samples
7. 🔄 **Hyperparameter Tuning** - Optimize with Optuna/Ray Tune
8. 🔄 **Model Registry** - Version control with MLflow Model Registry
9. 🔄 **CI/CD Pipeline** - Automate testing and deployment
10. 🔄 **Monitoring Dashboard** - Track model performance in production

## Contributing
This is a personal project to make learn about building end-to-end ML pipelines. Feedback and suggestions are welcome!

## License
MIT License