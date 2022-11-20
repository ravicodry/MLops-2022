from sklearn import datasets, svm, metrics, tree
import pdb
import pandas as pd
import numpy as np
import argparse
import joblib

from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    f1_scoring
)
from joblib import dump, load

def f1(clf_args,random_state_args):
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    assert train_frac + dev_frac + test_frac == 1.0

    # 1. set the ranges of hyper parameters
    gamma_list = [0.01, 0.005, 0.001, 0.0004, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    svm_params = {}
    svm_params["gamma"] = gamma_list
    svm_params["C"] = c_list
    svm_h_param_comb = get_all_h_param_comb(svm_params)

    max_depth_list = [5, 10, 25, 50, 100]

    dec_params = {}
    dec_params["max_depth"] = max_depth_list
    dec_h_param_comb = get_all_h_param_comb(dec_params)

    h_param_comb = {"svm": svm_h_param_comb, "dtree": dec_h_param_comb}

    # PART: load dataset -- data from csv, tsv, jsonl, pickle
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    del digits

    # define the evaluation metric
    metric_list = [metrics.accuracy_score, f1_scoring]
    h_metric = metrics.accuracy_score
    predicted_lable_svm  = list()
    predicted_labels_DecisionTree = list()
    n_cv = 1
    results = {}
    for n in range(n_cv):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac
        )
        if clf_args=="svm":
            clf=svm.SVC()
        elif clf_args=="dtree":
            clf = tree.DecisionTreeClassifier()
        
       
        print("[{}] Running hyper parameter tuning for {} ====>> ".format(n,clf_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
        )
        best_model = load(actual_model_path)
        # filename = 'finalized_model.sav'
        joblib.dump(best_model , './model/model_jlib')

        predicted = best_model.predict(x_test)
        if clf_name =='svm':
            predicted_lable_svm = predicted
        if clf_name =='dtree':
            predicted_labels_DecisionTree = predicted

        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})

        print(
            f"Classifier's Matrix ==== {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

    #print(results)
    df = pd.DataFrame(results)
    print(df)
    if clf_args=="svm":
        mean_svm=[]
        for i in range(1):
            mean_svm.append(results['svm'][i]['accuracy_score'])
        df1=pd.DataFrame(mean_svm)
        print("SVM Accuracy Mean")
        print(float(np.round(df1.mean(),4)))
        print("SVM Standard Deviation")
        print(float(np.round(df1.std(),4)))

    if clf_name =='dtree':
        mean_dt=[]
        for i in range(1):
            mean_dt.append(results['dtree'][i]['accuracy_score'])
        df2=pd.DataFrame(mean_dt)
        print("DECISION TREE Accuracy Mean")
        print(float(np.round(df2.mean(),4)))
        print("DECISION TREE Standard Deviation")
        print(float(np.round(df2.std(),4)))

    path = "./results/{0}_{1}".format(clf_name,random_state)
    
    f = open(path, "a")
    f.write("test accuracy "+ str(float(np.round(df2.mean(),4))))
    f.write("test macro-f1 "+str(float(np.round(df2.mean(),4))))
    f.write("model saved at svm_gamma=0.001_C=0.2.joblib")
    f.close()
    
    
    


        # count=0
        # for i in range(len(predicted_lable_svm)):
        #     if predicted_lable_svm[i] != predicted_labels_DecisionTree [i]:
        #         count=count+1
        # print("Gap count in prediction label betweeen SVM and DECISION TREE",':',count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='INp/Op path')

    parser.add_argument(
        "--clf_name",required = True,
        help="clf_name not present")
    parser.add_argument(
        "--random_state",required = True,
        help="random_state not present")
   
    args = parser.parse_args()
    clf_name = args.clf_name
    random_state = args.random_state
    f1(clf_name,random_state)