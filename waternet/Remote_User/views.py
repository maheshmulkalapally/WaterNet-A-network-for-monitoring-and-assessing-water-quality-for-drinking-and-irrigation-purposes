from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
from django.contrib import messages  # For validation messages


import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,assessing_water_quality,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Prediction_Water_Quality_Detection(request):
    if request.method == "POST":

        RID = request.POST.get('RID')
        State = request.POST.get('State')
        District_Name = request.POST.get('District_Name')
        Place_Name = request.POST.get('Place_Name')
        ph = request.POST.get('ph')
        Hardness = request.POST.get('Hardness')
        Solids = request.POST.get('Solids')
        Chloramines = request.POST.get('Chloramines')
        Sulfate = request.POST.get('Sulfate')
        Conductivity = request.POST.get('Conductivity')
        Organic_carbon = request.POST.get('Organic_carbon')
        Trihalomethanes = request.POST.get('Trihalomethanes')
        Turbidity = request.POST.get('Turbidity')

        # **Validation Check: Ensure all fields are filled**
        if not all([RID, State, District_Name, Place_Name, ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]):
            messages.error(request, "All fields are required. Please fill in missing values.")
            return render(request, 'RUser/Prediction_Water_Quality_Detection.html')

        data = pd.read_csv("water_datasets.csv")

        def apply_results(results):
            return 0 if results == 0 else 1

        data['Label'] = data['Potability'].apply(apply_results)

        x = data['Place_Name']
        y = data['Label']

        # **Ensure Place_Name is not empty**
        if not Place_Name.strip():
            messages.error(request, "Place Name is required for prediction.")
            return render(request, 'RUser/Prediction_Water_Quality_Detection.html')

        cv = CountVectorizer()
        x = cv.fit_transform(x)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        models = [('naive_bayes', NB)]

        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        models.append(('svm', lin_clf))

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        models.append(('logistic', reg))

        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        models.append(('DecisionTreeClassifier', dtc))

        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        models.append(('KNeighborsClassifier', kn))

        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
        sgd_clf.fit(X_train, y_train)
        models.append(('SGDClassifier', sgd_clf))

        from sklearn.ensemble import VotingClassifier
        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)

        # **Ensure Place_Name is transformed before prediction**
        try:
            Place_Name1 = [Place_Name]
            vector1 = cv.transform(Place_Name1).toarray()
            predict_text = classifier.predict(vector1)
        except Exception as e:
            messages.error(request, f"Error processing input: {str(e)}")
            return render(request, 'RUser/Prediction_Water_Quality_Detection.html')

        prediction = int(predict_text[0])
        val = 'IRRIGATION WATER' if prediction == 0 else 'DRINKING WATER'

        assessing_water_quality.objects.create(
            RID=RID, State=State, District_Name=District_Name, Place_Name=Place_Name,
            ph=ph, Hardness=Hardness, Solids=Solids, Chloramines=Chloramines,
            Sulfate=Sulfate, Conductivity=Conductivity, Organic_carbon=Organic_carbon,
            Trihalomethanes=Trihalomethanes, Turbidity=Turbidity, Prediction=val
        )

        return render(request, 'RUser/Prediction_Water_Quality_Detection.html', {'objs': val})

    return render(request, 'RUser/Prediction_Water_Quality_Detection.html')
