from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer,make_column_selector

# 欠損値代入->カテゴリカル変数のEncoding->標準化->モデル学習

# 処理する対象が違うので，カテゴリカルカラムと数値カラムのリストを取得する
cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()
num_cols = X.select_dtypes(include=np.number).columns.to_list()

# 欠損値代入
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')
ct = ColumnTransformer([('cat_imputer', cat_imputer, cmake_column_selector(dtype_exclude=np.number),
                   ('num_imputer', num_imputer, make_column_selector(dtype_include=np.number)])
ct.set_output(transform='pandas')

# target encoding
pipeline_te = Pipeline([('ct', ct),
          ('encoder', TargetEncoder()),
          ('scaler', StandardScaler()),
          ('model', LogisticRegression())])
