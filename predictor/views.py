from django.shortcuts import render
import joblib





# Create your views here.
def home(request):
    return render(request,'home.html')
     

def result(request):
    
    clf = joblib.load('finalized_model.sav')
    lis= []
   
    lis.append(request.GET['age'])
    lis.append(request.GET['sex'])
    lis.append(request.GET['cp'])
    lis.append(request.GET['threstbps'])
    lis.append(request.GET['chol'])
    lis.append(request.GET['fbs'])
    lis.append(request.GET['restecg'])
    lis.append(request.GET['thalach'])
    lis.append(request.GET['exang'])
    lis.append(request.GET['oldpeak'])
    lis.append(request.GET['slope'])
    lis.append(request.GET['ca'])
    lis.append(request.GET['thal'])

    
    print(lis)
    ans = clf.predict_proba([lis])

    return render(request,'result.html',{'ans':ans})



    
 


