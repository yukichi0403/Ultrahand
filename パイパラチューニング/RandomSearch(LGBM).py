from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint
import lightgbm as lgbm
from lightgbm import LGBMClassifier

params = {
    'objective': 'binary',
    'n_estimators': 300,
    'eval_metric': 'accuracy',
    'learning_rate': 0.03,
    'boosting': 'gbdt'
}

# LightGBMのモデルのトレーニングなどを行うコードを書く
X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# モデルの準備
model = LGBMClassifier(**params, random_state=0)

param_dist = {'num_leaves': randint(10, 60),
              'max_depth': randint(5, 15),
              'reg_alpha': uniform(0, 0.03)}

cv = KFold(n_splits=3, random_state=0, shuffle=True)

eval_set = [(X_val, y_val)]
callbacks = [lgbm.early_stopping(stopping_rounds=10),lgbm.log_evaluation(100)]
fit_params = {'callbacks':callbacks,'eval_set':eval_set}

rs = RandomizedSearchCV(model, cv=cv, param_distributions=param_dist, n_iter=36)
rs.fit(X_train_train, y_train_train, **fit_params)
