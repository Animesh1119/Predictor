import joblib
import pandas as pan   
import matplotlib.pyplot as plt 
import yfinance as fin 
import numpy as num 
import seaborn as sb 

stocks = input("Enter the code of the stock:- ") 
data = fin.download(stocks, "2021-04-01", "2022-03-31", auto_adjust=True) 




def linear(nm,sd="2021-04-01",ed="2022-03-31"):
    data = fin.download(nm, sd, ed, auto_adjust=True) 
    X = data.drop("Close", axis=1) 
    y = data["Close"] 
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression() 
    lr.fit(X_train, y_train) 
    # rr = lr.predict(X_test)
    import joblib  
    joblib.dump(lr, 'model.pkl') 
    ridge_from_joblib = model = joblib.load("model.pkl")

    def preprocess(Open,High,Low,Volume):
        test_data=num.array([[Open,High,Low,Volume]])
        trained_model=joblib.load("model.pkl")
        prediction=trained_model.predict(test_data)
        return prediction
    t=preprocess(97.31,100.32,97.31,27556600)
    print(t)
    # return t

#lass and ridge
def lasso():
    X = data.drop("Close", axis=1) 
    y = data["Close"] 
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 
    from sklearn.linear_model import Lasso, Ridge 
    la = Lasso().fit(X_train, y_train ) 
    joblib.dump(la, 'model.pkl') 
    ridge_from_joblib = model = joblib.load("model.pkl")

    def preprocess(Open,High,Low,Volume):
        test_data=num.array([[Open,High,Low,Volume]])
        trained_model=joblib.load("model.pkl")
        prediction=trained_model.predict(test_data)
        return prediction
    t=preprocess(97.31,100.32,97.31,27556600)
    return t


def ridge():
    X = data.drop("Close", axis=1) 
    y = data["Close"] 
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 
    from sklearn.linear_model import Lasso, Ridge 
    rr = Ridge(alpha=0.01)
    rr.fit(X_train, y_train) 
    joblib.dump(rr, 'model.pkl') 
    ridge_from_joblib = model = joblib.load("model.pkl")

    def preprocess(Open,High,Low,Volume):
        test_data=num.array([[Open,High,Low,Volume]])
        trained_model=joblib.load("model.pkl")
        prediction=trained_model.predict(test_data)
        return prediction
    t=preprocess(97.31,100.32,97.31,27556600)
    return t   

# svm
def svm():
    X = data.drop("Close", axis=1) 
    y = data["Close"] 
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 
    from sklearn.svm import SVR 
    from sklearn.model_selection import GridSearchCV 
    svr = SVR() 
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf']}   
    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)    
    grid.fit(X_train, y_train)
    svr = SVR(C=10, gamma=0.01, kernel='rbf') 
    svr.fit(X_train, y_train) 
    # svr_pred = svr.predict(X_test) 
    joblib.dump(svr, 'model.pkl') 
    ridge_from_joblib = model = joblib.load("model.pkl")

    def preprocess(Open,High,Low,Volume):
        test_data=num.array([[Open,High,Low,Volume]])
        trained_model=joblib.load("model.pkl")
        prediction=trained_model.predict(test_data)
        return prediction
    t=preprocess(97.31,100.32,97.31,27556600)
    return t





#refer


# X = data.drop("Close", axis=1) 
# y = data["Close"] 
# from sklearn.model_selection import train_test_split 
# X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 

# def algo(rr):
#     def modelSave(rr):
#     #save model
#         import joblib  
#         joblib.dump(rr, 'model.pkl') 
#         ridge_from_joblib = model = joblib.load("model.pkl")

#         def preprocess(Open,High,Low,Volume):
#             test_data=num.array([[Open,High,Low,Volume]])
#             trained_model=joblib.load("model.pkl")
#             prediction=trained_model.predict(test_data)
#             return prediction
#         t=preprocess(97.31,100.32,97.31,27556600)
#         print("Predicted Value for the given stock is: ",t)