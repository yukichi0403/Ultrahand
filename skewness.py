from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

numeric_cols = df_train.select_dtypes(include = np.number).columns.to_list()
skew_features = features[numeric_cols].apply(lambda x: skew(x)).sort_values(ascending=False)

# NOTE: `skewness`が`0.5`を超える列だけに絞り込む
high_skew = skew_features[skew_features > 0.75]
skew_index = high_skew.index

# log1p関数で正規分布に近い形に変換
df_train[skew_index] = np.log1p(df_train[skew_index])
