missing_num = df_train.isnull().sum().sort_values(ascending = False)
missing_num = pd.DataFrame(missing_num,columns = ['Total'])

df_train = df_train.drop(columns = missing_num[missing_num['Total'] > 1].index)
