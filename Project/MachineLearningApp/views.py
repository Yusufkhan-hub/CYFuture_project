from django.shortcuts import render
import pickle
import datetime
import numpy
# Create your views here.

def ML_Home(request):
    return render(request,'ML_model/ml_home.html')

def get_prediction_value(passenger_count,month,day,year,weekdays,hour,trip_distance):
    
    model = pickle.load(open('ML_Cab_Fare_price_pred_model.sav','rb'))

    prediction =model.predict([[passenger_count,month,day,year,weekdays,hour,trip_distance]])

    return prediction


def predicted_result(request):

    passenger_count = float(request.GET['passenger_count'])
    date = request.GET['date']
    weekdays = int(request.GET['weekdays'])
    hour = int(request.GET['hour'])
    trip_distance = float(request.GET['trip_distance'])

#splitted date into year, day, month
    print(date)
    print(date.split("-")[1])
    month = int(date.split("-")[1])
    day = int(date.split("-")[2])
    year = int(date.split("-")[0])
    result = get_prediction_value(passenger_count,month,day,year,weekdays,hour,trip_distance)
    result = numpy.around(result,decimals=2)
    return render(request,'ML_model/ml_result.html',{'result':result})