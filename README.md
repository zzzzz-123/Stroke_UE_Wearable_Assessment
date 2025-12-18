# Stroke_UE_Wearable_Assessment

This repository provides the complete data processing, feature engineering, scoring refinement, and regression modeling pipeline for quantitative assessment of upper-limb motor function based on wearable sensor data.
Raw data are not shared due to ethical and privacy constraints; this repository focuses on methodology and implementation.

📁 Project Structure and Script Description

1: `d01_data_preprocessing.py` — Data Preprocessing

This script performs raw data preprocessing, including:
* Signal smoothing using moving average filtering
* Motion cycle segmentation based on therapist-marked time points
* Linear interpolation to standardize each movement cycle to a fixed length
* Coordinate transformation and signal alignment
Note: Raw sensor data are not included or shared in this repository.

2: `d02_feature_extraction.py` — Feature Extraction

This script extracts multi-domain features from preprocessed signals for each movement cycle, including:
* **Amplitude features** (peak-to-peak range)
* **Smoothness features** (RMS of first-order derivative)
* **Dynamic Time Warping (DTW)** distances to standard movement templates
* **Approximate Entropy (ApEn)** to quantify signal complexity
Features are extracted independently for each sensor and movement and saved as structured feature matrices.

3: `d03_feature_selection_model_establishment_feature_importance.py`

Feature Selection, Model Establishment, and Feature Importance Analysis**

This script:
* Performs **Sequential Feature Selection (SFS)**
* Trains **Random Forest (RF)** regression models
* Establishes **single-action scoring models**
* Computes feature importance scores
A total of 8 independent RF models are built, corresponding to 8 different rehabilitation actions.

4: `d04_hyper_para_sfs.py` — Feature Number Sensitivity Analysis

This script investigates the influence of feature number on model performance:
* Fixes RF hyperparameters
* Varies the number of selected features in SFS
* Evaluates model performance across feature set sizes
* Determines the **optimal number of features**
Outputs include:
* Optimal feature subsets
* Corresponding RF model performance
* Final feature importance weights

5: `d05_feature_stretch.py` — Feature Normalization for Score Refinement

This script prepares features for score refinement by:
* Normalizing each feature to the range [0, 2]
* Performing label-aware segmented stretching
This step lays the foundation for fine-grained scoring beyond discrete ordinal labels.


6: `d06_score_refine_with_feature_importance.py` — Score Refinement Model

This script implements the **score refinement strategy** by:
* Combining normalized feature values
* Weighting features using RF-derived importance scores
* Producing **continuous refined scores** for each individual action
This enables finer differentiation within the same original clinical score level.


7: `d07_regress_for_UE.py` — Upper-Limb Function Prediction (Refined Scores)

This script predicts overall upper-limb motor function scores by:
* Using the **8 refined action scores** as model inputs
* Training regression models to estimate global upper-limb function (e.g., FMA-UE)
This represents the **proposed full pipeline**.


8: `d08_regress_for_UE.py` — Upper-Limb Function Prediction (Original Scores)

This script serves as a baseline comparison by:
* Using the **8 original discrete action scores**
* Predicting overall upper-limb motor function without score refinement
Performance comparison between `d07` and `d08` highlights the benefit of refined scoring.

🔁 Overall Pipeline Summary

1. **Raw data preprocessing**
2. **Feature extraction**
3. **Feature selection & RF model training (per action)**
4. **Feature number optimization**
5. **Feature normalization and stretching**
6. **Single-action score refinement**
7. **Upper-limb function regression using refined scores**
8. **Baseline regression using original scores**

⚠️ Data Availability Statement

Due to patient privacy and ethical regulations, **raw sensor data are not publicly available**.
This repository is intended to support **methodological transparency and reproducibility**.


