import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

sns.set(style='darkgrid', context='notebook')
df = pd.read_csv('Data_Train.csv')
dff = pd.read_csv('Data_Test.csv')



#print(df.head())

print(df.dtypes)

print(df.shape)

#print(df.describe())   #about numeric fields


#to check for null values
print(df.isnull().sum())
print(dff.isnull().sum())


#extract int part and convert type to int
df['New_Price'] = df['New_Price'].str.extract('(\d+)').astype(float)
dff['New_Price'] = dff['New_Price'].str.extract('(\d+)').astype(float)








df.dropna(subset = ['Mileage'],inplace = True)
dff.dropna(subset = ['Mileage'],inplace = True)
'''two rows have been dropped'''


print(df[df['Mileage'].str.contains('km/kg')])
print(dff[dff['Mileage'].str.contains('km/kg')])

#print(df.isnull().sum())
#print(df.info())

#print(df[df.duplicated()])
'''its an empty dataframe so no duplicates'''


#to extract float from string and convert type to float
df['Mileage'] = df['Mileage'].str.extract('(\d+\.\d+)').astype(float)
dff['Mileage'] = dff['Mileage'].str.extract('(\d+\.\d+)').astype(float)


df['Power'] = df['Power'].str.extract('(\d+\.\d+)').astype(float)
dff['Power'] = dff['Power'].str.extract('(\d+\.\d+)').astype(float)



#extract int part and convert type to int
df['Engine'] = df['Engine'].str.extract('(\d+)').astype(float)
dff['Engine'] = dff['Engine'].str.extract('(\d+)').astype(float)





#print(df.dtypes)




print(df.loc[df.Price > 100])
'''it is the outlier as seen from visualisation'''
df.drop(4079,inplace = True)








numeric = ['Year','Kilometers_Driven','Mileage','Engine','Power','Seats','Price','New_Price']

# Correlation matrix between numerical values
l = sns.heatmap(df[numeric].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

'''km driven has no effect on price so ill drop that from the heat map
and also seats too dont affect the price and year too'''
df.drop(['Kilometers_Driven','Seats','Year'], axis = 'columns',inplace = True)
dff.drop(['Kilometers_Driven','Seats','Year'], axis = 'columns',inplace = True)




#as new price column has many missing values i.e more than 50% so i'm dropping that
df.drop(['New_Price'], axis = 'columns',inplace = True)
dff.drop(['New_Price'], axis = 'columns',inplace = True)



#print(df.loc[df.Engine.isnull(),['Power','Price','Mileage']])
#print(df.loc[df.Power.isnull(),['Engine','Price','Mileage']])

#to fill engine values with mean
values = {'Engine': df.Engine.mean()}
valuesdff = {'Engine': dff.Engine.mean()}



df.fillna(value = values,inplace = True)
dff.fillna(value = valuesdff,inplace = True)




df.Price = np.log1p(df.Price)





##########################




'''as 2324 null values in power so replacing with mean not a precise measure as only 6k points are there in total
and as power is highly related with price we cant just drop that'''
'''so what we will do is we will predict the missing power values from the feature highly related with power'''

from sklearn import datasets, linear_model

regr = linear_model.LinearRegression()



d_nan = df[['Mileage','Engine','Price','Power']]
d_without_nan = d_nan.dropna()
#print(d_without_nan)

#train = d_without_nan.values
#OR
trainx = d_without_nan.iloc[:,:3] #training vaues
trainy = d_without_nan.iloc[:,3]  #training labels

#training the data
regr.fit(trainx,trainy)

test = d_nan.iloc[:,:3]
#print(test)

pred = pd.Series(regr.predict(test))   #pred as series as fillna needs to have series as argument not a list

df.Power.fillna(pred,inplace = True)





########################################

###########################################
from sklearn import datasets, linear_model

regr = linear_model.LinearRegression()


d_an = dff[['Mileage','Engine','Power']]
d_without_an = d_an.dropna()
#print(d_without_nan)

#train = d_without_nan.values
#OR
traix = d_without_an.iloc[:,:2] #training vaues
traiy = d_without_an.iloc[:,2]  #training labels

#training the data
regr.fit(traix,traiy)

tes = d_an.iloc[:,:2]
#print(test)

pre = pd.Series(regr.predict(tes))   #pred as series as fillna needs to have series as argument not a list

dff.Power.fillna(pre,inplace = True)





########################################
df.Power.fillna(df.Power.mean(),inplace = True)
###########################################

#categorical value to numeric codes to check for the coorelation
df.Fuel_Type = df.Fuel_Type.astype('category').cat.codes
df.Transmission = df.Transmission.astype('category').cat.codes
df.Owner_Type = df.Owner_Type.astype('category').cat.codes
df.Name = df.Name.astype('category').cat.codes
df.Location = df.Location.astype('category').cat.codes
print(df.dtypes)


dff.Transmission = dff.Transmission.astype('category').cat.codes
dff.Location= dff.Location.astype('category').cat.codes
dff.Fuel_Type= dff.Fuel_Type.astype('category').cat.codes
dff.Name = dff.Name.astype('category').cat.codes


print(df.shape)


numeri = ['Mileage','Engine','Power','Price','Transmission']

# Correlation matrix between numerical values
o = sns.heatmap(df[numeri].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.title('after some processing')
plt.show()




'''fuel type and owner type doesnt have any strong relation so dropping them '''
'''also name col'''

df.drop(['Owner_Type'], axis = 'columns',inplace = True)
dff.drop(['Owner_Type'], axis = 'columns',inplace = True)

df.drop(['Fuel_Type'], axis = 'columns',inplace = True)
dff.drop(['Fuel_Type'], axis = 'columns',inplace = True)

df.drop(['Name'], axis = 'columns',inplace = True)
dff.drop(['Name'], axis = 'columns',inplace = True)












'''
df["Fuel_Type"] = df["Fuel_Type"].map({"Diesel":1, "Petrol":2,'CNG':3,'LPG':4})
'''

#print(df.Name.value_counts())





'''
#########################



plt.hist(df['Mileage'])
plt.title('mileage wise  count of cars')
plt.xlabel('mileage in kmpl')
plt.show()

plt.hist(df['Engine'])
plt.title('engine performance wise count of cars')
plt.xlabel('engine performance in cc')
plt.show()

plt.scatter(df['Engine'], df['Price'], edgecolors='b')
plt.xlabel('engine performance in cc')
plt.ylabel('price')
plt.show()

plt.scatter(df['Transmission'], df['Price'], edgecolors='b')
plt.title('transmission vs price')
plt.show()

plt.scatter(df['Power'], df['Price'], edgecolors='b')
plt.title('Power vs price')
plt.show()

plt.scatter(df['Mileage'], df['Price'], edgecolors='b')
plt.title('Mileage vs price')
plt.show()

plt.hist(dff['Mileage'])
plt.show()




'''
plt.hist(df['Price'])
plt.show()
#plt.hist(np.log1p(df['Price'])) # to make a normal curve
#plt.show()



df.drop(['Location'], axis = 'columns',inplace = True)
dff.drop(['Location'], axis = 'columns',inplace = True)

# most influencing features for the price
print(df.corr().loc[:,'Price'].abs().sort_values(ascending=False))


print(df.info())
#to recheck for null values


print(df.isnull().sum())

#print(df.Power.isnull())
print(dff.isnull().sum())

#print(df.shape)
#print(dff.shape)
print(df.describe())
print(dff.describe())
df.to_csv('train_new.csv',index = False)






#############MODELLING####################
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split, KFold
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn import ensemble
import xgboost



m = df.values  #trainig set
x = m[:,0:4]
#print(x)
y = m[:,4]  #training labels
#print(y)

mm = dff.values #test set
xx = mm[:,0:4]


validation = 0.30
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(x,y,test_size=validation
                                                                ,random_state=7)




def linear_regression(train_ip,train_op,valid_ip,valid_op,test_ip):
    regr = linear_model.LinearRegression()
    regr = regr.fit(train_ip,train_op)
    predtest = regr.predict(test_ip)
    predtrain = regr.predict(valid_ip)
    print(r2_score(valid_op, predtrain))
    #np.sqrt(mean_squared_log_error( valid_op, predtrain ))





def randon_forest(train_ip,train_op,valid_ip,valid_op,test_ip):
    clf=RandomForestRegressor(n_estimators=300,max_features=3,max_depth=14,min_samples_split=2)
    clf = clf.fit(train_ip,train_op)
    predtest = clf.predict(test_ip)
    predtrain = clf.predict(valid_ip)
    return(r2_score(valid_op, predtrain))
    #dff['Price'] = pd.Series(predictionsrf)
    #dff.to_csv('test_new.csv',index= False)






def Gb(train_ip,train_op,valid_ip,valid_op,test_ip):
    #param_test1 = {'n_estimators': range(100, 800, 100)}  #400 decided
    #param_test2 = {'learning_rate':[.01,.001,.005,.05]} #.05 done
    #param_test3 = {'max_depth': range(5, 12, 2)} # 9
    #param_test4 = {'min_samples_leaf': range(5, 40, 10)} # 5 done
    #param_test5 = {'min_samples_split': range(10, 60, 10)} # 20 done
    #param_test6 = {'max_features':range(2,4,1)} # 2 done
    #param_test7 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]} # .8 done


    grab = ensemble.GradientBoostingRegressor(n_estimators=400,max_depth= 9,min_samples_leaf =5,min_samples_split=20,
                                              learning_rate = .05,random_state=10,max_features =2,subsample =.8)

    #model = GridSearchCV(
     #   estimator=grab, param_grid=param_test8, cv=5)

    grab.fit(train_ip,train_op)
    #print(model.best_params_, model.best_score_)
    predtrain = grab.predict(valid_ip)
    predtest = grab.predict(test_ip)

    #plt.bar(dff.columns,model.best_estimator_.feature_importances_)
    plt.bar(dff.columns,grab.feature_importances_)
    plt.title('Feature Importances')
    plt.show()
    dff['Price'] = pd.Series(predtest)
    rmsle = np.sqrt(mean_squared_log_error(valid_op, predtrain))
    dff.to_csv('test_new.csv',index= False)
    #r2_score(valid_op, predtrain)
    return(1-rmsle,r2_score(valid_op, predtrain))






def xgb(train_ip,train_op,valid_ip,valid_op,test_ip):
    #param_test1 = {'max_depth': range(3, 10, 2),min_child_weight': range(1, 6, 2)} # 5 and 1 done
    param_test2 = {'gamma': [i / 10.0 for i in range(0, 5)]} # 0 done default
    #param_test3 = { 'subsample': [i / 10.0 for i in range(6, 10)],
                    #'colsample_bytree': [i / 10.0 for i in range(5, 10)]}
    #param_test4 = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]} # 0 done


    xg = xgboost.XGBRegressor(learning_rate=0.1,min_child_weight =1,max_depth = 9,colsample_bytree = .5,subsample = .7,
                              gamma = 0.0, n_estimators=1000, reg_alpha=0.75, reg_lambda=0.45,seed=42,silent=1,
                              ref_alpha = 0)

    #model = GridSearchCV(estimator=xg, param_grid=param_test4, cv=5)

    xg = xg.fit(train_ip,train_op)
    #print(model.best_params_, model.best_score_)

    predtrain = xg.predict(valid_ip)
    predtest = xg.predict(test_ip)

    #plt.bar(dff.columns, model.best_estimator_.feature_importances_)
    plt.bar(dff.columns,xg.feature_importances_)
    plt.title('Feature Importances')
    plt.show()
    rmsle = np.sqrt(mean_squared_log_error(valid_op, predtrain))
    #print('xgb score: ', xg.score(valid_op, predtrain))
    return (1 - rmsle,r2_score(valid_op, predtrain))
    #r2_score(valid_op, predtrain)





#rf = randon_forest(X_train, Y_train, X_validation, Y_validation, xx)
#gb = Gb(X_train, Y_train, X_validation, Y_validation, xx)
#XGb = xgb(X_train,Y_train,X_validation,Y_validation,xx)  #we dont need to impute nan values in XGB
#lr = linear_regression(X_train,Y_train,X_validation,Y_validation,xx)



#print('score of linear regression: ',str(lr))
#print('score of random forest: ',str(rf))
#print('score(1-rmsle) of gradient boosting: ',str(gb))
#print('score(1-rmsle) and r^2 score of xgboost: ',str(XGb))







#####ENSEMBLE by weighted average####


def ensem(train_ip,train_op,valid_ip,valid_op,test_ip):
    regr = linear_model.LinearRegression()

    clf = RandomForestRegressor(n_estimators=300, max_features=3, max_depth=14, min_samples_split=2)

    grab = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=9, min_samples_leaf=5,
                                              min_samples_split=20,learning_rate=.05, random_state=10,
                                              max_features=2, subsample=.8)

    xg = xgboost.XGBRegressor(learning_rate=0.1, min_child_weight=1, max_depth=5, colsample_bytree=.5,
                              subsample=.7,gamma=0.0, n_estimators=1000, reg_alpha=0.75, reg_lambda=0.45, seed=42, silent=1,
                              ref_alpha=0)


    regr.fit(train_ip, train_op)
    clf.fit(train_ip, train_op)
    grab.fit(train_ip, train_op)
    xg.fit(train_ip, train_op)

    #validation prediction
    m1= regr.predict(valid_ip)
    m2= clf.predict(valid_ip)
    m3 = grab.predict(valid_ip)
    m4 = xg.predict(valid_ip)


    #test prediction
    mt1= regr.predict(test_ip)
    mt2= clf.predict(test_ip)
    mt3 = grab.predict(test_ip)
    mt4 = xg.predict(test_ip)

    final_pred_validation = (.2*m2 + .3*m3 + .5*m4)
    final_pred_test = (.2*mt2 + .3*mt3 + .5*mt4)


    rmsle = np.sqrt(mean_squared_log_error(valid_op, final_pred_validation))
    dff['Price'] = pd.Series(final_pred_test)
    dff.to_csv('test_new.csv', index=False)

    return (1 - rmsle,r2_score(valid_op,final_pred_validation))


s = ensem(X_train, Y_train, X_validation, Y_validation, xx)
print(s)
