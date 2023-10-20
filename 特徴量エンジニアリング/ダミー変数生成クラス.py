#ダミー変数生成クラスを自作
from sklearn.base import BaseEstimator,TransformerMixin
class Get_Dummies(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.columns = None
    
    def fit(self,X,y=None):
        self.columns = pd.get_dummies(X).columns
        return self
    
    def transform(self,X):
        X_new = pd.get_dummies(X)
        X_new.reindex(self.columns,fill_value=0)
        return X_new
