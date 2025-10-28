# Project Structure
```
pathology_segmentation/
├── README.md
├── requirements.txt
├── setup.py # optional, if you want to make it installable
├── configs/
│ ├── default.yaml # default hyperparameters and config
│ └── experiments/ # experiment-specific configs
├── data/
│ ├── raw/ # raw data (optional)
│ └── processed/ # processed/augmented tiles
├── src/
│ ├── init.py
│ ├── datasets/
│ │ ├── init.py
│ │ └── cocahis_dataset.py
│ ├── models/
│ │ ├── init.py
│ │ ├── factory.py # model factory
│ │ └── unetpp.py # UNet++ specific code
│ ├── losses/
│ │ ├── init.py
│ │ └── loss_factory.py # modular loss class
│ ├── utils/
│ │ ├── init.py
│ │ ├── logging.py # MLflowLogger, TensorBoard, etc.
│ │ ├── checkpoints.py # save/load checkpoints
│ │ ├── transforms.py # get_train_transform(), get_val_transform()
│ │ └── metrics.py # IoU, F1, TPR, FPR, etc.
│ ├── training/
│ │ ├── init.py
│ │ ├── trainer.py # train_model() refactored
│ │ └── validation.py # validate_one_epoch()
│ └── inference/
│ ├── init.py
│ └── predict.py # test-time evaluation & visualization
├── notebooks/
│ └── exploration.ipynb
├── scripts/
│ └──train.py # entry point for training
│ └──evaluate.py # entry point for testing
│ └── visualize_results.py
```