import json
from matplotlib.path import Path
import torch
from transformers import Trainer
from src.model.sound2sheet_model import Sound2SheetModel
from src.dataset.dataset_generator import DatasetGenerator, DatasetConfig
from src.model import ModelConfig, DataConfig, Trainer, TrainingConfig
from src.model import create_dataloaders

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_synthetic_data(total_samples, complexity_distribution, output_dir):
    config = DatasetConfig(total_samples=total_samples,
                           complexity_distribution=complexity_distribution, 
                           output_dir=output_dir)
    generator = DatasetGenerator(config)
    data = generator.generate()
    return data

def create_model_config(config, device):
    model_config = ModelConfig(
        num_piano_keys=config["model_config"]["num_piano_keys"],
        hidden_size=config["model_config"]["hidden_size"],
        dropout=config["model_config"]["dropout"],
        frame_duration_ms=config["model_config"]["frame_duration_ms"],
        classification_threshold=config["model_config"]["classification_threshold"],
        num_classifier_layers=config["model_config"]["num_classifier_layers"],
        classifier_hidden_dim=config["model_config"]["classifier_hidden_dim"],
        use_temporal_conv=config["model_config"]["use_temporal_conv"],
        temporal_conv_kernel=config["model_config"]["temporal_conv_kernel"],
        device=device
    )

    return model_config


def create_loader(config, model_config):    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        optimizer=config["training"]["optimizer"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        use_mixed_precision=config["training"]["use_mixed_precision"],
        max_grad_norm=config["training"]["max_grad_norm"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        checkpoint_dir=f'{config["experiment_name"]}/checkpoints',
        log_dir=f'{config["experiment_name"]}/logs',
        save_every_n_epochs=config["training"]["save_every_n_epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"]
    )

    # Data configuration
    data_config = DataConfig(
        sample_rate=config["data_config"]["sample_rate"],
        n_mels=config["data_config"]["n_mels"],
        dataset_dir=config["experiment_name"],
    )

    return create_dataloaders(data_config, model_config, training_config), training_config


def create_model(config, device):
    model_config = create_model_config(config, device)
    model = Sound2SheetModel(model_config, freeze_encoder=True)
    return model

def run_train(model, train_loader, val_loader, model_config, training_config):
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        training_config=training_config
    )

    trainer.train()
    return trainer

def run_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {device}")
    
    config = read_config("config.json")

    print("STEP 1: Generating synthetic data...")
    generate_synthetic_data(
        total_samples=config["dataset"]["total_samples"],
        complexity_distribution=config["dataset"]["complexity_distribution"],
        output_dir=config["experiment_name"]
        )
    
    print("STEP 2: Creating dataloaders...")
    model_config = create_model_config(config, device)
    loaders, training_config = create_loader(config, model_config)
    train_loader, val_loader, test_loader = loaders
    
    print("STEP 3: Creating model...")
    model = create_model(config, device)

    print("STEP 4: Starting training loop...")
    run_train(model, train_loader, val_loader, model_config, training_config)
    

if __name__ == "__main__":
    run_pipeline()