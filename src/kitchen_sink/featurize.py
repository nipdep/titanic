import os
import sys
import yaml
import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.impute import KNNImputer

params = yaml.safe_load(open('params.yaml'))['ft']

np.set_printoptions(suppress=True)

if len(sys.argv) != 3:
    sys.stderr.write('Argument error. Usage:\n')
    sys.stderr.write('\tpython featurization.py data-dit-path feature-dir-path\n')
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'train.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')
train_output = os.path.join(sys.argv[2], 'ks_train.csv')
test_output = os.path.join(sys.argv[2], 'ks_test.csv')

n_neigh = params['n_neigh']

train_df = pd.read_csv(train_input)
test_df = pd.read_csv(test_input)

col_mask = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
ms_train_df = train_df.loc[:, col_mask]
ms_train_df['target'] = train_df['Survived']
ms_test_df = test_df.loc[:, col_mask]

# Label Encoding 
def cat_to_int(df, columns, enc={}):
    df = df.copy()
    if enc == {}:
        maps = {}
        for col in columns:
            mapping = {k: i for i,k in enumerate(df.loc[:,col].unique())}
            df[col] = df[col].map(mapping)
            maps[col] = mapping
        return df, maps
    else:
        maps = enc
        for col in columns:
            df[col] = df[col].map(maps[col])
        return df

enc_train_df, enc_map = cat_to_int(ms_train_df, ['Sex', 'Embarked'])
enc_test_df = cat_to_int(ms_test_df, ['Sex', 'Embarked'], enc_map)

# Data Imputation

# drop EMbarked missing value records
fl_train_df = enc_train_df.dropna(subset=['Embarked'])
fl_test_df = enc_test_df.dropna(subset=['Embarked'])

# knn imputation for age
imputer = KNNImputer(n_neighbors=n_neigh, weights="uniform")
imp_train_df = pd.DataFrame(imputer.fit_transform(fl_train_df.iloc[:,:-1]), columns=col_mask)
imp_test_df = pd.DataFrame(imputer.transform(fl_test_df), columns=col_mask)
imp_train_df['target'] = fl_train_df.iloc[:,-1]


imp_train_df.to_csv(train_output, index=False)
imp_test_df.to_csv(test_output, index=False)
