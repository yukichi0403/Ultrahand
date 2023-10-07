import pandas as pd

#基本
df = pd.get_dummies(df, drop_first=True)

#列を指定してダミー変数か
df = pd.get_dummies(df, drop_first=True, columns=['sex', 'rank']))

#Nanもダミー変数化
df = pd.get_dummies(df, drop_first=True, dummy_na=True)

#ダミー変数の列名を指定
df = pd.get_dummies(df, drop_first=True, prefix='', prefix_sep='')


#異なるdfでダミー変数カラムがことならないようにする方法
categories = set(df_A['state'].unique().to_list() + df_B['state'].unique().to_list())
df_A['state'] = pd.Categorical(df_A['state'], categories=categories)
df_B['state'] = pd.Categorical(df_B['state'], categories=categories)]
df_A = pd.get_dummies(df_A)
df_B = pd.get_dummies(df_B)


#番外編：各カテゴリー（水準）に対して文字列で分類された各カテゴリーを任意の数値に置換
df['state'] = df['state'].map({'CA': 0, 'NY': 1, 'TX': 2})
