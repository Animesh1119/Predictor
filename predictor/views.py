from django.shortcuts import render
from django.http import HttpResponse




import joblib
import pandas as pan   
import matplotlib.pyplot as plt 
import yfinance as fin 
import numpy as num 
import seaborn as sb 

def linear(name,startDate,endDate,Open,High,Low,Volume):
    data = fin.download(name, startDate, endDate, auto_adjust=True) 
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
    t=preprocess(Open,High,Low,Volume)
    return t
    # t=preprocess(97.31,100.32,97.31,27556600)





# Create your views here.
def index(request):
    try:
        nm = request.GET['stockName']
        sdate = request.GET['startDate']
        ldate = request.GET['endDate']
        algo = request.GET['algorithm']
        vol = int(request.GET['volume'])
        open = int(request.GET['open'])
        high = int(request.GET['high'])
        low = int(request.GET['low'])
        if(algo=="LinearRegression"):
            t = linear(nm,sdate,ldate,open,high,low,vol)
            return HttpResponse(f"predicted value is: {t}")

            
    except:
        pass
    return render(request,'./index.html')
