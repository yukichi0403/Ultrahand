all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na.sort_values(ascending = False).head(10)
