# PAI-Project
For our PAI course project, we are building several disease prediction systems, including heart disease, diabetes, Parkinson's, and breast cancer classification. Using machine learning algorithms, we aim to analyze patient data and improve the accuracy of early diagnosis, providing valuable insights to healthcare professionals.
# Diabetes Prediction System using Pima Indian Dataset

This project is a machine learning application to predict diabetes using the **Pima Indian Diabetes Dataset**. It utilizes data preprocessing, feature engineering, and machine learning models to achieve accurate predictions. Additionally, a **Tkinter-based GUI** provides an intuitive interface for individual and batch predictions.

---

## Features
- **Machine Learning Models:**
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Ensemble Voting Classifier (Soft Voting)
- **Exploratory Data Analysis (EDA):**
  - Visualizations (Histograms, Boxplots, Pairplots, Correlation Heatmap)
- **GUI:**
  - Predict individual diabetes outcomes
  - Batch prediction from CSV files
  - Light/Dark themes
  - Validation of inputs
  - Save prediction results
- **Model Persistence:** Save and load trained models with `joblib`.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Machine Learning Workflow](#machine-learning-workflow)
5. [Tkinter GUI Application](#tkinter-gui-application)
6. [Results](#results)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Installation
1. Clone the repository:
    ```bash
    (https://github.com/iamrayyan1/PAI-Project/blob/main/diabetes.csv)
    ```
2. Navigate to the project directory:
    ```bash
    (https://github.com/iamrayyan1/PAI-Project/tree/main)
    ```
3. Install the required libraries:
    ```bash
    pip install pandas
    pip install matplotlib
    pip install seaborn
    pip install numpy
    pip install scikit-learn
    pip install imbalanced-learn
    pip install joblib
    ```

---

## Dataset
- **Name:** Pima Indian Diabetes Dataset
- **Description:** This dataset contains medical diagnostic measurements for predicting the onset of diabetes based on specific factors. It has 768 entries with 8 features and a target variable (`Outcome`).

---

## Machine Learning Workflow
### Steps:
1. **Data Preprocessing:**
   - Handle missing values using median imputation.
   - Address outliers using the **IQR method**.
   - Scale features using `StandardScaler`.

2. **Class Imbalance Handling:**
   - Applied **SMOTETomek** for oversampling minority class and undersampling majority class.

3. **Model Training:**
   - Models Used:
     - Random Forest Classifier
     - K-Nearest Neighbors (KNN)
     - Ensemble Voting Classifier
   - Train-Test Split: 80%-20% with stratification.

4. **Model Evaluation:**
   - Metrics:
     - Confusion Matrix
     - Accuracy
     - Classification Report
     - ROC-AUC Score

### Metrics:
- **Accuracy:** 0.86
- **ROC-AUC Score:** 0.92

### Feature Importance:
![Feature Importance](./images/feature_importance.png)

---

## Tkinter GUI Application
The **Tkinter-based GUI** allows:
- **Individual Predictions:** Enter patient data manually.
- **Batch Predictions:** Load and predict multiple records from a CSV file.
- **Themes:** Switch between Light/Dark themes.
- **Help Section:** Tooltips for field inputs and detailed help instructions.

### GUI Screenshot:
![GUI Example](./images/gui_example.png)

---

## Results
### Model Performance:
| Metric               | Score  |
|----------------------|--------|
| **Accuracy**         | 86%    |
| **ROC-AUC Score**    | 92%    |

---

## Usage
### Command-Line Prediction:
1. Train and save the model:
    ```bash
    python train_model.py
    ```
2. Load and use the saved model:
    ```bash
    python predict.py
    ```

### GUI Usage:
1. Run the GUI:
    ```bash
    python gui.py
    ```

---

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## License
This project is licensed under the Creative Commons Legal Code License. See the `LICENSE` file for details.
