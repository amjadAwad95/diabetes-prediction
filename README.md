# Diabetes Prediction Using Machine Learning

## Overview

This project aims to predict the likelihood of diabetes in individuals based on various health, lifestyle, and socioeconomic factors. The dataset used for this project contains 70,692 samples with 22 features, including BMI, age, blood pressure, cholesterol levels, smoking habits, and more. The goal is to build a machine learning model that can accurately predict whether an individual has diabetes or not.

The project leverages several machine learning algorithms, including Logistic Regression, Linear SVM, Decision Tree, Random Forest, and AdaBoost. The best-performing model, AdaBoost, achieved an accuracy of 75.2% and an F1 score of 75.7%.

## Dataset

The dataset used in this project is sourced from Kaggle and contains the following features:

- **BMI**: Body Mass Index, a measure of body fat based on weight and height.
- **PhysHlth**: Number of days (in the past 30 days) when physical health was not good.
- **Age**: Age category of the individual.
- **HighBP**: Whether the individual has been diagnosed with high blood pressure (binary).
- **HighChol**: Whether the individual has been diagnosed with high cholesterol (binary).
- **CholCheck**: Whether the individual has had their cholesterol checked in the past five years (binary).
- **Smoker**: Whether the individual is currently a smoker (binary).
- **Stroke**: Whether the individual has had a stroke (binary).
- **HeartDiseaseorAttack**: Whether the individual has had heart disease or a heart attack (binary).
- **PhysActivity**: Whether the individual engages in physical activity (binary).
- **Fruits**: Whether the individual consumes fruit at least once per day (binary).
- **Veggies**: Whether the individual consumes vegetables at least once per day (binary).
- **HvyAlcoholConsump**: Whether the individual consumes heavy amounts of alcohol (binary).
- **AnyHealthcare**: Whether the individual has any kind of healthcare coverage (binary).
- **NoDocbcCost**: Whether the individual could not see a doctor in the past year due to cost (binary).
- **GenHlth**: Self-reported general health status (ordinal, ranging from "Excellent" to "Poor").
- **MentHlth**: Number of days (in the past 30 days) when mental health was not good.
- **DiffWalk**: Whether the individual has difficulty walking or climbing stairs (binary).
- **Sex**: The sex of the individual (binary).
- **Education**: Level of education (ordinal, typically ranging from "No formal education" to "College graduate").
- **Income**: Income level category.
- **Diabetes_binary**: Whether the individual has been diagnosed with diabetes (1 for diabetes, 0 for no diabetes).

## Preprocessing

The dataset underwent several preprocessing steps:

- **Feature Selection**: Relevant features were selected based on their correlation with the label (`Diabetes_binary`). Features with low correlation, such as `AnyHealthcare`, `NoDocbcCost`, and `Sex`, were removed.
- **Transformation**: Numerical features were scaled using Min-Max scaling to normalize the data. The `BMI` feature was scaled using a user-defined function.
- **Feature Extraction**: New features were created to improve the model's performance:
  - `health_score`: A new feature representing the health score from a scale of 0 to 3 by combining `BMI`, `HighBP`, and `HighChol`.
  - `MultiRisk_factors`: A new feature flagging individuals with multiple risk factors using `BMI`, `HighBP`, and `HighChol`.
  - `EatingHealthyFood`: A new feature flagging users who eat either veggies or fruits at least once per day.
  - `SocioeconomicStatus`: A new feature representing the socioeconomic level of the individual using `Income` and `Education`.

## Methodology

### Approach

The project uses supervised learning, where the model learns from labeled data. The following algorithms were applied:

1. **Logistic Regression**: Used for binary classification, producing a probability value between 0 and 1.
2. **Linear SVM**: Uses a linear decision boundary to separate data points of different classes.
3. **Decision Tree**: A flowchart-like structure used to make decisions or predictions.
4. **Random Forest**: A powerful tree learning technique that creates multiple decision trees during training.
5. **AdaBoost**: Iteratively trains weak classifiers, giving more weight to misclassified data points.

### Tools and Frameworks

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **scikit-learn**: For machine learning algorithms and tools.
- **matplotlib**: For data visualization.
- **DVC**: For data version control.

### Evaluation Criteria

The models were evaluated using the following metrics:

- **Accuracy**: The primary metric, selected because the dataset is balanced.
- **F1 Score**: A measure of the model's balance between precision and recall.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **ROC AUC**: The area under the receiver operating characteristic curve.

### Experimental Setup

- **Data Splitting**: The dataset was split into 80% training data and 20% testing data using random splitting.
- **Cross-Validation**: A 5-fold cross-validation approach was used to ensure the models generalize well to unseen data.
- **Hyperparameter Tuning**: GridSearch and RandomSearch were used to optimize the hyperparameters for each model.

## Results

The best-performing model was **AdaBoost**, achieving an accuracy of 75.2% and an F1 score of 75.7%. The second-best model was **Logistic Regression**, with an accuracy of 74.9% and an F1 score of 75.4%.

### Key Findings

- **AdaBoost** outperformed other models, demonstrating its capability to handle the complexities of the dataset.
- **Decision Tree** and **Random Forest** models showed signs of overfitting.
- Normalization using Min-Max scaling provided slightly better results across all models.
- The F1 score was slightly higher than the accuracy in all models, indicating a good balance between recall and precision.

## Conclusion

This project successfully addresses the classification problem of predicting diabetes using machine learning techniques. The AdaBoost model achieved the best performance, with an accuracy of 75.2%. The project provided valuable insights into building a machine learning pipeline, including data preprocessing, feature selection, model training, and evaluation.

## How to Run the Project

To run this project, follow the steps below. Note that you will need to download the dataset before running the main script or the EDA notebook.

---

### 1. **Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone https://github.com/amjadAwad95/diabetes-prediction.git
cd diabetes-prediction
```

---

### 2. **Download the Dataset**
The dataset used in this project is sourced from Kaggle. You need to download it and place it in the `data/` folder.

#### Steps to Download the Dataset:
1. Go to the Kaggle dataset page: [Diabetes Dataset on Kaggle]([https://www.kaggle.com/datasets/your-dataset-link](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv)).
2. Download the dataset (usually a CSV file).
3. Place the downloaded file (e.g., `diabetes.csv`) in the `data/` folder of the project.

The folder structure should look like this:
```
diabetes-prediction/
├── data/
│   └── diabetes.csv  # Place the dataset here
```

---

### 3. **Set Up the Virtual Environment**
Create and activate a virtual environment to manage dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

---

### 4. **Install Dependencies**
Install the required Python packages, copy the command line in `packages.txt`:

```bash
pip install pandas numpy scikit-learn matplotlib dvc dvc_s3 flask flask_cors
```

---

### 5. **Run the Main Script**
Once the dataset is in place and dependencies are installed, you can run the main script to train and evaluate the models:

```bash
python src/main.py
```

---

### 6. **Explore the EDA Notebook**
To explore the exploratory data analysis (EDA), open the `EDA.ipynb` notebook in Jupyter Notebook:

```bash
jupyter notebook src/EDA.ipynb
```

Make sure the dataset is in the `data/` folder before running the notebook.

---

## Notes:
- Ensure the dataset is correctly placed in the `data/` folder before running any scripts or notebooks.
- If you encounter any issues, check the dataset path in the code and ensure it matches the location of your dataset file.

---

## Acknowledgments

- **Kaggle** for providing the dataset.
- **scikit-learn** for the machine learning tools and algorithms.
- **DVC** for data version control.
