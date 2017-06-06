# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function
import optparse
import csv
import json
import os
import time
import sys
import pickle
import base64
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


def one_hot_dataframe_train(data, cols, replace=True):
    """ Takes a dataframe and a list of columns that need to be encoded.
    Returns the data and the fitted vectorizor.
    """
    vec = DictVectorizer(dtype=np.int8)
    vecData = vec.fit_transform(data[cols].to_dict(orient='records')).toarray()
    feature_names = vec.get_feature_names()

    return vecData, vec, np.array(feature_names)

    
    
def preprocess(dataframe, thresh):

    #start = time.time()
    #filling columns with all missing values with 0
    for col in dataframe.columns[pd.isnull(dataframe).all()]:
        dataframe[col] = dataframe[col].astype(object).fillna(0)
    

    #filling missing numeric values with mean and non-numeric with "unknown
    dataframe[["device_advertising_app_impressions", "bid_value"]] = dataframe[["device_advertising_app_impressions", "bid_value"]].fillna(0)
    dataframe[["publisher_app", "advertiser_campaign", "advertiser_app", "cat_action", 
    "cat_adventure", "cat_card", "cat_casino", "cat_educational", "cat_family", "cat_music", 
    "cat_non_game", "cat_puzzle", "cat_racing", "cat_role_playing", "cat_simulation", "cat_sports", 
    "cat_strategy", "cat_trivia", "cat_other", "country", "region", "reachability", "ad_type", 
    "asset_size", "mapped_device_name", "has_install"]] = dataframe[["publisher_app", "advertiser_campaign", "advertiser_app", "cat_action", 
    "cat_adventure", "cat_card", "cat_casino", "cat_educational", "cat_family", "cat_music", 
    "cat_non_game", "cat_puzzle", "cat_racing", "cat_role_playing", "cat_simulation", "cat_sports", 
    "cat_strategy", "cat_trivia", "cat_other", "country", "region", "reachability", "ad_type", 
    "asset_size", "mapped_device_name", "has_install"]].fillna("unknown")
    
    #copy dataframe
    xtrain = dataframe.copy()

    #target value
    ytrain = pd.to_numeric(xtrain.pop('has_install').values)

    #add sample weights to balance data
    #sample_weights = np.array([float((len(ytrain[ytrain==1])+1))/(len(ytrain[ytrain==0]+1)) if i == 0 else 1 for i in ytrain])
   
    #convert to lower case
    xtrain = xtrain.apply(lambda x: x.astype(str).str.lower())
    
    #convert to numerical
    xtrain[["device_advertising_app_impressions", "bid_value"]] = xtrain[["device_advertising_app_impressions", "bid_value"]].apply(pd.to_numeric)

    #convert categorical variables to binary variables
    xtrain_Arr, vec, feature_names = one_hot_dataframe_train(xtrain, ["publisher_app", "advertiser_campaign", "advertiser_app", "cat_action", 
    "cat_adventure", "cat_card", "cat_casino", "cat_educational", "cat_family", "cat_music", 
    "cat_non_game", "cat_puzzle", "cat_racing", "cat_role_playing", "cat_simulation", "cat_sports", 
    "cat_strategy", "cat_trivia", "cat_other", "country", "region", "reachability", "ad_type", 
    "asset_size", "mapped_device_name"])    
    
    #print "finished one hot"

    xtrain_new = np.empty((len(xtrain_Arr),0))
    selected_features = []

    for i in range(0, len(feature_names), 1000):
        try:
            sel = VarianceThreshold(threshold=(thresh))
            xtrain_batch = sel.fit_transform(xtrain_Arr[:,i:i+1000])
            features_batch = feature_names[i:i+1000][sel.get_support()]
            xtrain_new = np.append(xtrain_new, xtrain_batch, axis = 1)
            selected_features = np.append(selected_features, features_batch)
        except:
            continue

    xtrain_new = np.append(xtrain_new, np.array(xtrain[['device_advertising_app_impressions','bid_value']],dtype='int32'),axis=1)
    selected_features = np.append(selected_features,['device_advertising_app_impressions','bid_value'])

    #print "done sel_names : %s" %len(sel_names)
    sel_dict = {}
    
    for s in selected_features:
        sel_dict[s] = len(sel_dict)        

    #min max scaling
    #min_max_scaler = MinMaxScaler()
    #xtrain_minmax = min_max_scaler.fit_transform(xtrain_new)
    #scale = min_max_scaler.scale_
    #minim = min_max_scaler.min_
    
    return xtrain_Arr, ytrain, sel_dict

    

def GridSearchModel(Xtrain, ytrain, model_name):
    
    # Split the dataset train, test
    X_train, X_test, y_train, y_test = train_test_split(
        Xtrain, ytrain, test_size=0.3, random_state=0)
    
    scores = ['roc_auc', 'average_precision']
    
    if model_name == "RF":
        param_grid = {'n_estimators' : [50, 200, 500],
                      'max_features': ['sqrt', 'log2'],
                      'min_samples_split': [100, 500]
                     }

        param_grid = {'n_estimators' : [10],
                      'max_features': [None],
                      'min_samples_split': [100]
                     }
                     
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5,
                               scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()    
        
    elif model_name == "GB":
        print("place holder")
        
    elif model_name == "RFLR":
        print("place holder")
        
    elif model_name == "GBLR":
        print("place holder")
        
    elif model_name == "LR":
        print("place holder")
        
    else:
        sys.stderr.write("unexpected model name!")
        exit(1)   
    
    return



def gbc_classify(X, Y, sample_weights):
#gradient boosting classifier training
    gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3)
    gbc.fit(X, Y, sample_weight=sample_weights)
    return gbc


def rf_classify(X, Y):
#random forest classifier
    rf = ensemble.RandomForestClassifier(n_estimators=10)
    rf.fit(X, Y)
    return rf


def SG_classify(X, Y, class_0_weight, class_1_weight, sgc=None):
#stochastic gradient descent classifier
    if sgc:
        if np.bincount(Y)[0]>0 and len(np.bincount(Y))>1:
            sgc.partial_fit(X, Y)
    else:
        param_SGC = {'loss': 'hinge', 'penalty': 'elasticnet', 'n_iter': 1, 'shuffle': True, 'class_weight': {0: class_0_weight, 1:class_1_weight}, 'warm_start':True, 'alpha':0.001}
        sgc = SGDClassifier(**param_SGC)
        if np.bincount(Y)[0]>0 and len(np.bincount(Y))>1:
            sgc.partial_fit(X, Y, np.unique(Y))
            coef = sgc.coef_
            intercept = sgc.intercept_
        else:
            sgc=None
            coef = None
            intercept = None 
        
    return sgc, coef, intercept

    
    
def main():

    variance_threshold = 0.01
    
    parser = optparse.OptionParser()
    parser.add_option("-i", "--input_file", type="string", dest="input_file",
                      help="path to the training input file, eg, data.csv",
                      default="../data/top10_pubs_impressions.csv")

    
    (opts, args) = parser.parse_args()

    input_dir = opts.input_file.split(".csv")[0]

    columns = ["publisher_app", "advertiser_campaign", "advertiser_app", "cat_action", 
    "cat_adventure", "cat_card", "cat_casino", "cat_educational", "cat_family", "cat_music", 
    "cat_non_game", "cat_puzzle", "cat_racing", "cat_role_playing", "cat_simulation", "cat_sports", 
    "cat_strategy", "cat_trivia", "cat_other", "country", "region", "reachability", "device_advertising_app_impressions", 
    "bid_value", "ad_type", "asset_size", "mapped_device_name", "has_install"]



    #load data
    training_all=pd.read_csv("%s" %opts.input_file, header=0, sep='\t', dtype={'publisher_app':np.str, 'advertiser_campaign':np.str, 'cat_action':np.bool, 
                                'cat_adventure':np.str, 'cat_card':np.bool, 'cat_casino':np.bool,'cat_educational':np.str, 'cat_family':np.bool, 
                                'cat_music':np.bool, 'cat_non_game':np.bool, 'cat_puzzle':np.bool, 'cat_racing':np.bool, 'cat_role_playing':np.bool,
                                'cat_simulation':np.bool, 'cat_sports':np.bool, 'cat_strategy':np.str, 'cat_trivia':np.bool,
                                'cat_other':np.bool, 'country':np.str, 'region':np.str, 'reachability':np.str, 'device_advertising_app_impressions': np.str,
                                'bid_value':np.float64, 'ad_type':np.str, 'asset_size':np.str, 'mapped_device_name':np.str, 'has_install':np.bool}, names=columns)


    
    #preprocess the data and save the selected features in a pickle file
    X, y, sel_feature_dict = preprocess(training_all, variance_threshold)
    
    MyFeatures = open('%s_selected_features.pickle' %input_dir, 'wb') 
    pickle.dump(sel_feature_dict, MyFeatures)
    MyFeatures.close()
            
    
    #split data into training and testing
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

    #class_0_weight = len(ytrain)/(2.*np.bincount(ytrain)[0]+1)
    #class_1_weight = 1.3 * len(ytrain)/(2.*np.bincount(ytrain)[1]+1)
     
    #ranfom forest classifier                       
    rf = rf_classify(Xtrain, ytrain)

    #pickle the trained random forest classifier
    MyRFModel = open('%s_rf_model.pickle' %input_dir, 'wb') 
    pickle.dump(rf, MyRFModel)
    MyRFModel.close() 
    
    #just do the ranking of the test set
    ypred_prob = rf.predict_proba(Xtest)


    return


if __name__=="__main__":
    main()

    
    
    

        
        