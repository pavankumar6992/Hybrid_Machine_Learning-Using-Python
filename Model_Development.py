#*******************************# Model Development #*******************************

import numpy as np
import pandas as pd
#%matplotlib inline 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import preprocessing

X = data.iloc[:,1:11]
cat1 = X[['Dew_PointC','Humidity','Sea_Level_PressurehPa','TemperatureC','VisibilityKm','Wind_Speed_in_KMperHour','WindDirDegrees']]
cat2 = X[['Conditions','Wind_Direction','Hour']]

x = cat1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
cat1 = pd.DataFrame(x_scaled)
cat1.columns = ['Dew_PointC','Humidity','Sea_Level_PressurehPa','TemperatureC','VisibilityKm','Wind_Speed_in_KMperHour','WindDirDegrees']
# Merge cat1 cat2
X = pd.concat([cat1,cat2], axis = 1)
y = pd.DataFrame(data.iloc[:, 0])

y['Energy_consumption'] = y['Energy_consumption'].astype(str)
y.Energy_consumption    = y.Energy_consumption.str.slice(0,3)
y['Energy_consumption'] = y['Energy_consumption'].astype(float)

y.loc[y['Energy_consumption'] > 2, 'Energy_consumption'] = 'High'
y['Energy_consumption'] = y['Energy_consumption'].astype(str)
y['Energy_consumption'].replace(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.4','1.5'],['Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low','Low'], inplace = True)
y['Energy_consumption'].replace(['1.6','1.7','1.8','1.9','2.0'],['Medium','Medium','Medium','Medium','Medium'], inplace = True)

bin_cols = y.nunique()[y.nunique() <= 3].keys().tolist()
le = LabelEncoder()
for i in bin_cols :
    y[i] = le.fit_transform(y[i])

plt.figure(figsize=(8, 6))
sns.countplot('Energy_consumption', data=y)
plt.title('Balanced Classes')
plt.show()

from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2       
#15000 features with highest chi-squared statistics are selected 
chi2_features = SelectKBest(chi2, k = 8)
X = chi2_features.fit_transform(X, y)   

# Training & Test Holdouts
# Used : Humidity, TemperatureC,Wind_Speed_in_KMperHour,VisibilityKm
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10, 
                                                                          random_state=50,
                                                                          shuffle=True)

#******** XG Boost************

import numpy as np
from xgboost.sklearn import XGBClassifier
#from sklearn.grid_search import  GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
seed = 1
np.random.seed(seed) 
#learning_rates = np.arange(0.01, 0.37, 0.01)
#learning_rates = [0.05, 0.1, 0.25, 0.5, 0.6, 0.75, 1]
#for learning_rate in learning_rates:  Use this line while using a list of parameters


def evaluation(y_act, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_curve,auc
    print("Confusion Matrix = ", confusion_matrix(y_act, y_pred))
    print("Accuracy = ", accuracy_score(y_act, y_pred))
    print("Precision = " ,precision_score(y_act, y_pred, average='micro'))
    print("Recall = " ,recall_score(y_act, y_pred, average='micro'))
    print("F1 Score = " ,f1_score(y_act, y_pred, average='micro'))
#   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_act, y_pred)
#   print("AUC Score =", auc(false_positive_rate, true_positive_rate))
    print("Kappa score = ",cohen_kappa_score(y_act,y_pred))
    print("Error rate = " ,1 - accuracy_score(y_act, y_pred))
    #print("AUC-ROC Curve: ")
    #plt.plot([0, 1], [0, 1], linestyle='--')
    #plt.plot(false_positive_rate, true_positive_rate,marker='.')
    #plt.show()
    pass


# Model 1: XGBoost Start: ************

xgb = XGBClassifier(learning_rate =0.43,
booster = 'gbtree', # 'dart 69%' 
n_estimators=100,
max_depth=26,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
verbosity =1,
seed=27)
xgb.fit(x_training_set, y_training_set.values.ravel())

# Predicting the Test set results
y_pred = xgb.predict(x_test_set)
#accuracy_score(y_pred,y_test_set)
accuracy = cross_val_score(estimator = xgb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
accuracy.mean()
evaluation(y_test_set,y_pred)

from xgboost import plot_importance
plot_importance(xgb)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_set, y_pred)
cm
#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test_set)


# Model 2 SVC Start ***************

from sklearn.metrics import classification_report
from sklearn.svm import SVC
svc = SVC(C = 100, kernel = 'rbf',gamma = 0.001, random_state = 0)
svc.fit(x_training_set, y_training_set.values.ravel())
y_pred = svc.predict(x_test_set)
#accuracy_score(y_pred,y_test_set)
print("Accuracy score (training): {0:.3f}".format(svc.score(x_training_set, y_training_set)))
print("Accuracy score (validation): {0:.3f}".format(svc.score(x_test_set, y_test_set)))
print() 
evaluation(y_test_set, y_pred)

# Model 3 Gaussian NB Start  ****************

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_training_set, y_training_set.values.ravel())
# Predicting the Test set results
y_pred = gnb.predict(x_test_set)
print("Accuracy score (training): {0:.3f}".format(gnb.score(x_training_set, y_training_set)))
print("Accuracy score (validation): {0:.3f}".format(gnb.score(x_test_set, y_test_set)))
print() 
evaluation(y_test_set, y_pred)


# Model 4  GBC-Start  ****************
    
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
gb = GradientBoostingClassifier(n_estimators=26, learning_rate = 0.28 , max_features=2, max_depth = 19, random_state = 0)
gb.fit(x_training_set, y_training_set.values.ravel())
y_pred = gb.predict(x_test_set)
print("Accuracy score (training): {0:.3f}".format(gb.score(x_training_set, y_training_set)))
print("Accuracy score (validation): {0:.3f}".format(gb.score(x_test_set, y_test_set)))
print()
accuracy = cross_val_score(estimator = gb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
accuracy.mean()
evaluation(y_test_set,y_pred)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=51),
    n_estimators= 200
)
adb.fit(x_training_set, y_training_set.values.ravel())
y_pred = adb.predict(x_test_set)
#print('Final prediction score: [%.8f]' % accuracy_score(y_test_set, y_pred))
print("Accuracy score (validation): {0:.3f}".format(adb.score(x_test_set, y_test_set)))

#Hyper paramater tuning starts for XGBClassifier

depth = np.arange(0,1,1)
estimators = np.arange(0,10,1)
rate = np.arange(0.1,1,0.1)

params_grid = {
        'max_depth' : depth,
        'n_estimators' : estimators,
        'learning_rate' : rate
        }
params_fixed = {
        'objective' : 'binary:logistic',
        'silent' :1
        }
bst_grid = GridSearchCV(
        estimator = XGBClassifier(**params_fixed,seed = seed),
        param_grid = params_grid,
        scoring ='accuracy'
        )
bst_grid.fit(x_training_set,y_training_set)
bst_grid.cv_results_
print("Best accuracy Obtained: {0}".format(bst_grid.best_score_))
print("Parameters")
for key, value in bst_grid.best_params_.items():
    print("\t{}:{}".format(key, value))
    
# Hyper parameter tuning for XGB Ends. 

# Hyper parameter tuning for Ada Boost using for loop . 

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#
#learning_rates  = np.arange(10,200,10)
#
#for learning_rate in learning_rates:
#    classifier = AdaBoostClassifier(
#        DecisionTreeClassifier(max_depth=51),
#        n_estimators= 200
#    )
#    classifier.fit(x_training_set, y_training_set.values.ravel())
#    
#    y_pred = classifier.predict(x_test_set)
#    print(learning_rate)
#    print('Final prediction score: [%.8f]' % accuracy_score(y_test_set, y_pred))

# Hyper parameter tuning for AdaBoost Ends. 


# pip install vecstack (run in anaconda console to download vecstack)

from vecstack import stacking
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

models = [
    SVC(C = 100, kernel = 'rbf',gamma = 0.001, random_state = 0),
    GradientBoostingClassifier(n_estimators=26, learning_rate = 0.28 , max_features=2, max_depth = 19, random_state = 0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=51),n_estimators= 200)   
]
S_train, S_test = stacking(models,                   
                           x_training_set, y_training_set, x_test_set,   
                           regression=False,      
                           mode='oof_pred_bag',        
                           needs_proba=False,         
                           save_dir=None,             
                           metric=accuracy_score,     
                           n_folds=10,                  
                           stratified=True,            
                           shuffle=True,              
                           random_state=0,             
                           verbose=2)

model =   XGBClassifier(learning_rate = 0.43,booster = 'gbtree',n_estimators=100,max_depth=26,min_child_weight=1,
                  gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,
                  scale_pos_weight=1,seed=27)
model = model.fit(S_train, y_training_set.values.ravel())

y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test_set, y_pred))
from xgboost import plot_importance
plot_importance(model)



# AUC curve codes
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Learn to predict each class against the other

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10,random_state=50)

classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=26, learning_rate = 0.28 , max_features=2, max_depth = 19, random_state = 0))

y_score = classifier.fit(X_train, y_train).decision_function(X_test)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average AUC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average AUC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='AUC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class using Micro, macro averaging')
plt.legend(loc="lower right")
plt.show()

 # AUC code ends


test1 = cross_val_score(estimator = model, X = S_train, y = y_training_set.values.ravel(), cv =10)
test2 = cross_val_score(estimator = xgb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
test3 = cross_val_score(estimator = gb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
test4 = cross_val_score(estimator = svc, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
test5 = cross_val_score(estimator = adb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)
test6 = cross_val_score(estimator = gnb, X = x_training_set, y = y_training_set.values.ravel(), cv =10)


output = pd.DataFrame(test1)
output[xgb] = pd.DataFrame(test2)
output[gb] = pd.DataFrame(test3)
output[svc] = pd.DataFrame(test4)
output[adb] = pd.DataFrame(test5)
output[gnb] = pd.DataFrame(test6)


from scipy import stats
print(stats.shapiro(output))
output.columns = ['Stack','xgb','gb','svc','adb','gnb']

# P-Value of 0.002 indicates that the data is not normally distributed, 
#hence using Mann-Whitney U-test to chech for statistical significance

stats.mannwhitneyu(output['Stack'],output['gnb'],use_continuity=True, alternative=None)
stats.mannwhitneyu(output['Stack'],output['svc'],use_continuity=True, alternative=None)
stats.mannwhitneyu(output['Stack'],output['adb'],use_continuity=True, alternative=None)
stats.mannwhitneyu(output['Stack'],output['xgb'],use_continuity=True, alternative=None)
stats.mannwhitneyu(output['Stack'],output['gb'],use_continuity=True, alternative=None)



results = [0.0000903,0.0000729,0.00693772344142142,0.09244142720655457,0.454742198871193]
results = pd.DataFrame(results)
results['Algorithm'] = ['SVC','naivebayes','Adaboost','XGBoost','Gradient']
results['Significance'] = ['Yes','Yes','Yes','No','No']








