
# 最頻値で埋める
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])

# GroupByして特定のカテゴリーごとの中央値で埋める
all_data["LotFrontage"] = 
all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
