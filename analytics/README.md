# Titanic Data Analysis and Modeling Project

## About This Project

This project is about analyzing the Titanic dataset and building some machine learning models.

I have divided the work into two Python files:

- `01_eda.py` - Exploratory Data Analysis
- `02_modeling.py` - Machine Learning Modeling
- `README.md` - Project explanation

The main purpose of this project is to understand the data first and then use the cleaned data to build prediction models.

---

## Dataset

I am using the Titanic dataset available from Seaborn.

In `01_eda.py`, I load the dataset using:

```python
sns.load_dataset("titanic")
```

After loading the data, I save it as:

```text
titanic.csv
```

This is done so that the second Python file can use the saved CSV file instead of loading the dataset again.

---

## Project Files

### 1. `01_eda.py`

This file contains my Exploratory Data Analysis.

I have done the following things:

- Loaded the Titanic dataset
- Checked the shape of the dataset
- Used `df.info()`
- Used `df.describe()`
- Checked missing values
- Calculated missing-value percentages
- Handled missing values
- Analyzed age
- Analyzed fare
- Checked outliers using IQR
- Calculated fare mean, median and mode
- Checked fare skewness
- Checked survival rate by sex
- Checked survival rate by passenger class
- Checked survival rate using sex and passenger class together
- Used boolean filtering with `&` and `|`
- Created a correlation matrix
- Found the two strongest correlations
- Created charts to understand the data
- Standardized age and fare as an additional EDA exercise

Some charts are also saved as PNG files.

---

### 2. `02_modeling.py`

This file is for the machine learning part.

The target variable is:

```text
survived
```

The main features I use are:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

I first split the data into training and testing data using a stratified split.

Then I used preprocessing for numerical and categorical columns.

For numerical columns I used:

- Median imputation
- StandardScaler

For categorical columns I used:

- Most frequent imputation
- One-hot encoding

I kept the preprocessing inside pipelines so that the test data is not used while fitting the preprocessing.

---

## Machine Learning Models

I trained three classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

I compared the models using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Confusion Matrix

I also created ROC curves to compare the models.

---

## Class Imbalance

I also checked whether the target classes were balanced.

For Random Forest, I compared:

- Normal/baseline model
- `class_weight="balanced"`
- SMOTE

The results are compared using precision, recall and F1 score.

SMOTE is used only on the training data through a pipeline.

---

## Random Forest Tuning

After the initial models, I used `GridSearchCV` to find better Random Forest parameters.

I searched different values for:

- Number of trees
- Maximum depth
- Maximum features

I also used the Random Forest OOB score.

The best parameters and scores are printed when the script is executed.

---

## Regression

As an additional exercise, I used Linear Regression to predict:

```text
fare
```

using other available Titanic features.

For the regression model I calculated:

- MAE
- RMSE
- R²
- Adjusted R²

I also created a residual plot to check the regression errors.

---

## Model Saving

After tuning the Random Forest, I save the complete model pipeline as:

```text
titanic_best_pipeline.joblib
```

The saved pipeline includes the preprocessing and the Random Forest model.

I also reload the saved pipeline and test it with a sample input to make sure it works.

---

## How to Run the Project

First install the required Python packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
```

Then run the EDA file:

```bash
python 01_eda.py
```

After that, run the modeling file:

```bash
python 02_modeling.py
```

It is important to run `01_eda.py` first because it creates the `titanic.csv` file used by `02_modeling.py`.

---

## Files Created After Running

After running the scripts, some additional files will be created, such as:

```text
titanic.csv
titanic_best_pipeline.joblib
classification_results.csv
imbalance_results.csv
regression_results.csv
```

There will also be several PNG files containing the charts and model visualizations.

---

## What I Learned From This Project

Through this project I practiced:

- Loading data using Python
- Pandas DataFrame operations
- Checking and handling missing values
- Exploratory Data Analysis
- Data visualization
- Correlation analysis
- Outlier detection
- Data preprocessing
- Train/test splitting
- Classification models
- Regression
- Model evaluation
- Handling class imbalance
- SMOTE
- Hyperparameter tuning
- Pipelines
- Saving and loading machine learning models

---

## Conclusion

This project helped me understand the complete process from data analysis to machine learning.

First, I explored and cleaned the Titanic data. Then I built different classification models and compared their performance. I also tried different methods for class imbalance and tuned the Random Forest model.

Finally, I created a regression model for fare prediction and saved the best classification pipeline so it can be reused later.
