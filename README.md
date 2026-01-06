Multimodal House Price Prediction

This project predicts residential property prices by combining structured tabular data with satellite image–derived visual features. It compares traditional tabular-only regression models against multimodal models that incorporate neighborhood-level visual context extracted from satellite imagery.

The primary objective is to evaluate whether visual information such as greenery, building density, road layout, and open space improves predictive performance and interpretability over standard tabular approaches.

Project Structure
├── data/
│   ├── raw/
│   │   ├── train(1).xlsx
│   │   ├── test2.xlsx
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   ├── train_eda.csv
│   │   ├── test_eda (2).csv
│   │   └── .gitkeep
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── .gitkeep
│
├── data_fetcher.py
├── 23113029_final.csv
├── README.md

File Descriptions

data_fetcher.py
Downloads satellite images using geographic coordinates (latitude and longitude). Each image corresponds to a property location and is indexed using the property ID. Internet access is required for this step.

preprocessing.ipynb
Performs exploratory data analysis (EDA), tabular data cleaning, and feature engineering.
Also includes:

Satellite image preprocessing

Deep image feature extraction using a pretrained ResNet-based CNN

Dimensionality reduction of image embeddings using PCA

Grad-CAM–based visual explainability to identify important spatial regions influencing predictions

model_training.ipynb
Trains and evaluates multiple regression models, including:

Tabular-only models

Multimodal models combining tabular and image features
The notebook compares models using RMSE (log price) and R² (train and validation), selects the final model, builds a reproducible pipeline, and generates predictions on unseen test data.

23113029_final.csv
Final output file containing predicted house prices for the test dataset.

Environment Setup

To run this project successfully, ensure the following:

Python version 3.9 or higher

Standard scientific Python stack (NumPy, Pandas, scikit-learn)

XGBoost and LightGBM installed

PyTorch and torchvision for CNN-based image feature extraction

GPU support is recommended for faster image embedding extraction but is not mandatory

Internet access is required to download satellite images

All experiments were conducted using fixed random seeds to ensure reproducibility.

How to Run the Project

The project should be executed in the following order:

1. Image Acquisition

Run the satellite image fetching script to download images using latitude and longitude coordinates:

python data_fetcher.py


This step prepares the raw satellite images required for visual feature extraction.

2. Preprocessing & Feature Engineering

Open and execute the preprocessing notebook:

notebooks/preprocessing.ipynb


This notebook:

Performs univariate, bivariate, multivariate, and geospatial EDA

Cleans and transforms tabular features

Extracts deep image embeddings using a pretrained CNN

Applies PCA to reduce image feature dimensionality

Generates Grad-CAM visual explanations

Processed datasets are saved to the data/processed/ directory.

3. Model Training & Evaluation

Run the model training notebook:

notebooks/model_training.ipynb


This notebook:

Trains baseline tabular-only regression models

Trains multimodal models using tabular + image features

Compares models using RMSE (log price) and R²

Analyzes overfitting using train vs validation performance

Selects XGBoost (Tabular + Image) as the final model

Builds a reproducible pipeline

Generates predictions on unseen test data

Outputs & Reproducibility

All intermediate datasets, extracted features, and evaluation metrics are saved during execution.

Results can be reproduced by rerunning the notebooks in the specified order.

Model comparisons use consistent train–validation splits and evaluation metrics.

Grad-CAM visualizations provide interpretability for multimodal predictions.

Author
Ansul
23113029
Ansul Lakhlan
Mult
