import numpy as np              # NumPy for numerical operations
import pandas as pd             # Pandas for data manipulation
import matplotlib.pyplot as plt # Matplotlib for basic plotting
import seaborn as sns           # Seaborn for statistical data visualization
from tqdm import tqdm          # tqdm for progress bars
import datetime                # Datetime for date and time operations


from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedKFold, cross_val_score  # Cross-validation techniques
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss  # Evaluation metrics
from sklearn.model_selection import cross_validate  # Cross-validation scoring
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, precision_score, average_precision_score, mean_squared_error # Metrics and displays
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression  # Regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Random Forest and Gradient Boosting
from sklearn.pipeline import Pipeline, make_pipeline  # Pipeline for building a sequence of data transformations
import xgboost as xgb  # XGBoost for gradient boosting
import lightgbm as lgb  # LightGBM for gradient boosting
from catboost import Pool  # CatBoost for gradient boosting


# get file path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.set_option('display.max_columns', None)  # Show all columns


# Suppressing warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
