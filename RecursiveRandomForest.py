# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import warnings

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import r2_score#, regression_report
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)

data = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "A:Y")

##Mortality
y_mort = data.loc[:,'Mortality']
# Process
x_pro = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "P:W")
# Process + socioeconomic 
x_proso = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "A:W")
# Process + readmissions
x_pror = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "P:X")
# Process + socioeconomic + readmissions
x_prosor = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "A:X")

##Readmissions
y_read = data.loc[:, 'Readmissions']
# Process
#x_process
# Process + socioeconomic
#x_prosoc
# Process + mortality
x_prom = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "P:W,Y")
# Process + socioeconomic + mortality
x_prosom = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "A:W,Y")

##Rating
y_star = data.loc[:,'StarRating']
# Process(minus rating) + outcome 
x_pro_s = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "Q:W")
# Process(minus rating) + outcome + socioeconomic
x_proso_s = pd.read_excel('Data\ImputedDataUpdate.xlsx', parse_cols = "A:O,Q:W")

random_state = 100
######################## Mortality
print("_______________________________________")
print("*********__ MORTALITY __*********")
#Process vs Mortality: y_mort ~ x_pro
print("### Mortality ~ Process ###")
x_train, x_test, y_train, y_test = train_test_split(x_pro, y_mort, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_pro.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Ordered features for Mortality ~ Process : ", orderedFeatures)
print("Selected features for Mortality ~ Process : ", selected_feat)
print("Mortality ~ Process Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Mortality ~ Process Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Socioeconomic vs Mortality: y_mort ~ x_proso
print("### Mortality ~ Process + Socio ###")
x_train, x_test, y_train, y_test = train_test_split(x_proso, y_mort, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_proso.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Mortality ~ Process + Socio : ", selected_feat)
print("Ordered features for Mortality ~ Process + Socio : ", orderedFeatures)
print("Mortality ~ Process + Socio Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Mortality ~ Process + Socio Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Readmissions vs Mortality: y_mort ~ x_pror
print("### Mortality ~ Process + Read + Read ###")
x_train, x_test, y_train, y_test = train_test_split(x_pror, y_mort, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_pror.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Mortality ~ Process + Read : ", selected_feat)
print("Ordered features for Mortality ~ Process + Read : ", orderedFeatures)
print("Mortality ~ Process + Read Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Mortality ~ Process + Read Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Socioeconomic + Readmissions vs Mortality: y_mort ~ x_prosor
print("### Mortality ~ Process + Socio + Read ###")
x_train, x_test, y_train, y_test = train_test_split(x_prosor, y_mort, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_prosor.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Mortality ~ Process + Socio + Read: ", selected_feat)
print("Ordered features for Mortality ~ Process + Socio + Read : ", orderedFeatures)
print("Mortality ~ Process + Socio + Read Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Mortality ~ Process + Socio + Read Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))

######################## Readmissions
print("_______________________________________")
print("********* __READMISSIONS__ *********")
#Process vs Readmissions: y_read ~ x_pro
print("### Readmissions ~ Process ###")
x_train, x_test, y_train, y_test = train_test_split(x_pro_s, y_read, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_pro_s.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Readmissions ~ Process: ", selected_feat)
print("Ordered features for Readmissions ~ Process : ", orderedFeatures)
print("Readmissions ~ Process Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Readmissions ~ Process Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Socioeconomic vs Readmissions: y_read ~ x_proso
print("### Readmissions ~ Process + Socio ###")
x_train, x_test, y_train, y_test = train_test_split(x_proso_s, y_read, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_proso_s.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Readmissions ~ Process + Socio: ", selected_feat)
print("Ordered features for Readmissions ~ Process + Socio : ", orderedFeatures)
print("Readmissions ~ Process + Socio Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Readmissions ~ Process + Socio Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Mortality vs Readmissions: y_read ~ x_prom
'''print("### Readmissions ~ Process + Mort ###")
x_train, x_test, y_train, y_test = train_test_split(x_prom, y_read, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_prom.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Readmissions ~ Process + Mort: ", selected_feat)
print("Ordered features for Readmissions ~ Process + Mort : ", orderedFeatures)
print("Readmissions ~ Process + Mort Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Readmissions ~ Process + Mort Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Socioeconomic + Mortality vs Readmissions: y_read ~ x_prosom
print("### Readmissions ~ Process + Socio + Mort ###")
x_train, x_test, y_train, y_test = train_test_split(x_prosom, y_read, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_prosor.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Readmissions ~ Process + Socio + Read : ", selected_feat)
print("Ordered features for Readmissions ~ Process + Socio + Read : ", orderedFeatures)
print("Readmissions ~ Process + Socio + Read Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Readmissions ~ Process + Socio + Read Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
'''
######################## Rating
print("_______________________________________")
print("********* __RATING__ *********")
#Process + Outcome vs Rating: y_star ~ x_proo
print("### Rating ~ Process ###")
x_train, x_test, y_train, y_test = train_test_split(x_pro, y_star, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_pro.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Rating ~ Process : ", selected_feat)
print("Ordered features for Rating ~ Process : ", orderedFeatures)
print("Rating ~ Process Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Rating ~ Process Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))
#Process + Socioeconomic vs Rating: y_star ~ x_proso
print("### Rating ~ Process + Outcome + Socio ###")
x_train, x_test, y_train, y_test = train_test_split(x_proso, y_star, train_size = 0.7, random_state = random_state, shuffle = True)
randForD = RandomForestRegressor() #default n_estimators = 10
randForD = RandomForestRegressor() #default n_estimators = 10
randModel = randForD.fit(x_train, y_train)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF_D = GridSearchCV(randForD, param_grid = param_grid_RF, cv = 10)
grid_search_RF_D.fit(x_train, y_train)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds = randModelGrid.predict(x_train)
gridTestPreds = randModelGrid.predict(x_test)
gridTrainAccuracy = r2_score(y_train, gridTrainPreds)
gridTestAccuracy = r2_score(y_test, gridTestPreds)
sortedFeatureIndices = -np.argsort(randModelGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_proso.columns.values[i])
sel = SelectFromModel(RandomForestRegressor(n_estimators = grid_search_RF_D.best_params_['n_estimators'], max_depth = grid_search_RF_D.best_params_['max_depth']))
sel.fit(x_train,y_train)
selected_feat = x_train.columns[(sel.get_support())]
print("Selected features for Rating ~ Process + Socio : ", selected_feat)
print("Ordered features for Rating ~ Process + Socio : ", orderedFeatures)
print("Rating ~ Process + Socio Test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("Rating ~ Process + Socio Test R2 with hyperparameter tuning: ",round(gridTestAccuracy,2))


'''
############Random Forest with SocioEconomic data
print("Donabdeian and Socioeconomic Predictors")
print("_______________________________________")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.7, random_state = random_state, shuffle = True)
randFor = RandomForestRegressor() #default n_estimators = 10

randModel = randFor.fit(x_train, y_train)

print("Time to fit Random Forest model: ", rf_time)
rfPreds_train = randModel.predict(x_train)
rfTrainAccuracy = r2_score(y_train, rfPreds_train)
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = r2_score(y_test, rfPreds_test)
print("Random Forest full data train R2 without hyperparameter tuning: ",round(rfTrainAccuracy,2))
print("Random Forest full data test R2 without hyperparameter tuning: ",round(rfTestAccuracy,2))
sortedFeatureIndices = -np.argsort(randModel.feature_importances_)
#print("Sorted indices of important features: ", sortedFeatureIndices)
orderedFeatures = list()
for i in sortedFeatureIndices: 
    orderedFeatures.append(x_data.columns.values[i])
print(orderedFeatures)
param_grid_RF ={'n_estimators':[30,32,34,36,40], 
            'max_depth':[15,20,25,30,35]}
grid_search_RF = GridSearchCV(randFor, param_grid = param_grid_RF, cv = 10)
grid_search_RF.fit(x_train, y_train)
print("n_estimators tested: ", param_grid_RF['n_estimators'])
print("max_depths tested: ", param_grid_RF['max_depth'], "\nbest parameters and score: ")
print(grid_search_RF.best_params_)#n_estimators = 36, max_depth = 25
print(grid_search_RF.best_score_)
randForGrid = RandomForestRegressor(n_estimators = grid_search_RF.best_params_['n_estimators'], max_depth = grid_search_RF.best_params_['max_depth'])
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds_RF = randModelGrid.predict(x_train)
gridTestPreds_RF = randModelGrid.predict(x_test)
gridTrainAccuracy_RF = r2_score(y_train, gridTrainPreds_RF)
gridTestAccuracy_RF = r2_score(y_test, gridTestPreds_RF)
print("RandomForest test accuracy without hyperparameter tuning: ",round(rfTestAccuracy,2))
print("RandomForest test accuracy with hyperparameter tuning: ",round(gridTestAccuracy_RF,2))
print('Best n_estimators: ', grid_search_RF.best_params_['n_estimators'], '\n','Best max_depth: ', grid_search_RF.best_params_['max_depth'], "\n\n_______________")
'''

