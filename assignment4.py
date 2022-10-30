from sklearn import datasets, svm, metrics, tree
import pdb
import pandas as pd
import numpy as np

from utils import (
    pre_processing_digits,
    train_dev_test_split,
    data_viz,
    combination,
    save_tune,
    f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.004, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.3, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = combination(svm_params)

max_depth_list = [3, 10, 15, 30, 50]

d_params = {}
d_params["max_depth"] = max_depth_list
d_h_param_comb =combination(d_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": d_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = pre_processing_digits(digits)
# housekeeping
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, f1]
h_metric = metrics.accuracy_score
predicted_lable_svm  = list()
predicted_labels_DecisionTree = list()
n_cv = 5
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }
    for clf_name in models_of_choice:
        clf = models_of_choice[clf_name]
        print("[{}] Running hyper param tuning for {}".format(n,clf_name))
        actual_model_path =save_tune(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if clf_name =='svm':
    	       predicted_lable_svm = predicted
        if clf_name =='decision_tree':
    	       predicted_labels_DecisionTree = predicted
 
        print(predicted)
        print(len(predicted))
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print(results)

count=0
for i in range(len(predicted_lable_svm)):
 	if predicted_lable_svm[i] != predicted_labels_DecisionTree [i]:
         count=count+1
         print("index for which prediction is not matching",i)
print("Number of time predicted lable were not the same with both the classifier",':',count)
mean_svm=[]
for i in range(5):
    mean_svm.append(results['svm'][i]['accuracy_score'])
df1=pd.DataFrame(mean_svm)
print("svm mean accuracy")
print(float(np.round(df1.mean(),4)))
print("svm std")
print(float(np.round(df1.std(),4)))

mean_dt=[]
for i in range(5):
    mean_dt.append(results['decision_tree'][i]['accuracy_score'])
df2=pd.DataFrame(mean_dt)
print("svm mean accuracy")
print(float(np.round(df2.mean(),4)))
print("svm std")
print(float(np.round(df2.std(),4)))