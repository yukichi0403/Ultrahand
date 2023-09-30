#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = 
all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
