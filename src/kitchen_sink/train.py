import os
import sys
import yaml
import json
import pickle
import lightgbm as lgb
import dagshub
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

params = yaml.safe_load(open('params.yaml'))['train']

np.set_printoptions(suppress=True)

if len(sys.argv) != 5:
    sys.stderr.write('Argument error. Usage:\n')
    sys.stderr.write('\tpython featurization.py data-dit-path model-dir-path\n')
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'ks_train.csv')
test_input = os.path.join(sys.argv[1], 'ks_test.csv')
model_output = os.path.join(sys.argv[2], 'ks_model.pkl')
test_output = os.path.join(sys.argv[2], 'ks_test.csv')
score_path = os.path.join(sys.argv[3])
plots_file = os.path.join(sys.argv[4])

split = params['split']

df = pd.read_csv(train_input)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1],
    test_size=split,
    random_state=43
)

# model training
clf = lgb.LGBMClassifier()
clf.fit(X_train,y_train)

with dagshub.dagshub_logger() as logger:
    logger.log_hyperparams(model_class=type(clf).__name__)
    logger.log_hyperparams({'model': clf.get_params()})  

# model evaluation
y_tr_pred = clf.predict(X_train)
y_ts_pred = clf.predict(X_test)

#print(f"For train dataset \n \taccuracy : {accuracy_score(y_train, y_tr_pred)} \n \tprecision : {precision_score(y_train, y_tr_pred)} \n \trecall : {recall_score(y_train, y_tr_pred)} \n \t f1 score : {f1_score(y_train, y_tr_pred)}")
#print(f"For test dataset \n \taccuracy : {accuracy_score(y_test, y_ts_pred)} \n \tprecision : {precision_score(y_test, y_ts_pred)} \n \trecall : {recall_score(y_test, y_ts_pred)} \n \t f1 score : {f1_score(y_test, y_ts_pred)}")

with open(score_path,'w') as pf:
    json.dump({'train' : {'accuracy' : accuracy_score(y_train, y_tr_pred), 'precision' : precision_score(y_train, y_tr_pred), 'recall' : recall_score(y_train, y_tr_pred), 'f1-score' : f1_score(y_train, y_tr_pred)},
    'test' : {'accuracy' : accuracy_score(y_test, y_ts_pred), 'precision' : precision_score(y_test, y_ts_pred), 'recall' : recall_score(y_test, y_ts_pred), 'f1-score' : f1_score(y_test, y_ts_pred)}
    }, pf)

precision, recall, thresholds = precision_recall_curve(y_test, y_ts_pred)

with open(plots_file, 'w') as fd:
    json.dump({'prc': [{
            'precision': float(p),
            'recall': float(r),
            'threshold': float(t)
        } for p, r, t in zip(precision, recall, thresholds)
    ]}, fd)

# make prediction for test set
test_df = pd.read_csv(test_input)
preds = clf.predict(test_df)
test_df['prediction'] = preds

test_df.to_csv(test_output, index=False)

def save_model(path, model):
    
    msg = 'The output model : {} \n'
    sys.stderr.write(msg.format(model))

    with open(path, 'wb') as fd:
        pickle.dump(model, fd, pickle.HIGHEST_PROTOCOL)

save_model(model_output, clf)