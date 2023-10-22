ct = ColumnTransformer([('num_transform', SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),
                        ('cat_transform', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_exclude=np.number))])
ct.set_output(transform='pandas')
