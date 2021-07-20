from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views.generic import CreateView
from django.core.paginator import Paginator,PageNotAnInteger,EmptyPage
from catalog.models import laptop, accessory, cart, useraction
import numpy as np
import pandas as pd
import json
import csv
import nltk
import time
import tflearn
import tensorflow
from tensorflow.python import pywrap_tensorflow
import random
import math

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("intents.json") as file:
    tfdata = json.load(file)
words = []
labels = []
docs_x = []
docs_y = []


for intent in tfdata["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

dataset3 = []

for i in range(0, len(output)):
    dataset3.append([training[i], output[i]])

random.shuffle(dataset3)

training = []
output = []
for x, y in dataset3:
    training.append(x)
    output.append(y)
training = np.array(training)
output = np.array(output)

tensorflow.reset_default_graph()
'''
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=10, batch_size=1, show_metric=True)
model.save("model.tflearn")

'''
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('./model.tflearn', weights_only=True)
type(model)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat(inp):
    results = model.predict([bag_of_words(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]
    return tag,results

def home(request):
    return render(request, 'home.html')

def recommendation(request):
    if request.method == "POST":
        try:
            feedBack = request.POST['feedBack']
        except:
            feedBack = ""
        try:
            rating = request.POST['rating']
        except:
            rating = -1
        if feedBack != "" and rating != -1:
            response,valfrom_neural=chat(feedBack)
            #rate=(math.ceil(float(valfrom_neural[0][1])*10))/2
            rate = float(rating)
            if (rate>=3.0) and (response=="positive"):
                myData = [[feedBack,str(rate),response]]
                myFile = open('reviews.csv', 'a',newline='')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
            elif (rate<3) and (response=="negative"):
                myData = [[feedBack,str(rate),response]]
                myFile = open('reviews.csv', 'a',newline='')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(myData)
            '''
            data = laptop.objects.all()
            for dt in data:
                if dt.product.find("Inspiron 5579") != -1 or dt.product.find("Inspiron 7579") != -1 or dt.product.find("Inspiron 7378") != -1 or dt.product.find("Inspiron 5579") != -1 or dt.product.find("Inspiron 7579") != -1 or dt.product.find("Inspiron 7378") != -1 or dt.product.find("Inspiron 7779") != -1 or dt.product.find("Inspiron 5767") != -1:
                #if dt.product == "Inspiron 5370" :
                #if dt.product.find("Vostro") != -1:
                    dt.image = "Inspiron.jpg"
                    #dt.image = dt.product+".jpg"
                    dt.save()'''
    data = laptop.objects.all()
    for dt in data:
        print("###########################################################")
        dt.recommended = 0
        dt.selected = 0
        dt.checkout = 0
        dt.save()        
    return redirect('Home')


def store(request):
    var = useraction.objects.first()
    if var.visitedCount is None:
        var.visitedCount = 1
    else:
        var.visitedCount = var.visitedCount + 1
    var.save()
    if request.method == "POST":
        flipkartData = request.POST['x']
        amazonData = request.POST['y']
        googleData = request.POST['z']
        if flipkartData == []:
            result = []
            bestLaps = pd.read_csv('noData.csv')
            for bt in bestLaps.index:
                result.append(laptop.objects.get(id=int(bestLaps.at[bt,"Id"])))
            return render(request, 'base.html', {'list': result})
        else:
            am = json.loads(amazonData)
            go = json.loads(googleData)
            maximum_amazon = int(am[0]['maximum'])
            minimum_amazon = int(am[0]['minimum'])
            #print("amazon ",minimum_amazon,maximum_amazon)
            maximum_google = int(go[0]['maximum'])
            minimum_google = int(go[0]['minimum'])
            #print("google ",minimum_google,maximum_google)
            final_max = maximum_google
            final_min = minimum_google
            if minimum_google == -1 or (minimum_google > minimum_amazon):
                final_min=minimum_amazon
            if maximum_google == -1 or maximum_google<maximum_amazon:
                final_max=maximum_amazon
            loaded = pd.read_json(flipkartData)
            prolookup = pd.read_csv('processorsLookup.csv')
            loaded.columns = ["Price", "Cpu", "Ram"]
            for j in loaded.index:
                st = loaded.at[j, "Cpu"]
                tt = prolookup.loc[prolookup['ProcessorName'].str.find(st) != -1]
                for i in tt.index:
                    loaded.at[j, "Cpu"] = tt.at[i, "ProcessorName"]
            flip_min=500000
            flip_max=0
            for j in loaded.index:
                if loaded.at[j,"Price"]<flip_min:
                    flip_min=loaded.at[j,"Price"]
                
                if loaded.at[j,"Price"]>flip_max:
                    flip_max=loaded.at[j,"Price"]
            
            if final_min==-1 or flip_min<final_min:
                final_min=flip_min
            if final_max==-1:
                final_max=flip_max
            #final_max,final_min = 40000, 20000
            #final_max,final_min = 60000, 35000
            #final_max,final_min = 100000, 80000
            minmax_record=pd.read_csv("minmaxrecord.csv")
            min_old=minmax_record.at[0,"min"]
            max_old=minmax_record.at[0,"max"]
            minmax_record=open("minmaxrecord.csv","w")
            minmax_record.write(f'min,max\n{final_min},{final_max}')
            minmax_record.close()
            ll = []
            for k in loaded.index:
                if (loaded.at[k, 'Price'] < final_min) or (loaded.at[k, 'Price'] > final_max):
                    ll.append(k)
            loaded = loaded.drop(ll)
            loaded.to_csv("userBehaviour.csv")

            datasettemp = pd.read_csv('userBehaviour.csv')
            behaviourfile=""
            if len(datasettemp) <= 15:
                behaviourfile="user_midrange.csv"
            else:
                behaviourfile="userBehaviour.csv"
            dataset=pd.read_csv(behaviourfile)
            dataset1 = pd.read_csv('data3.csv')
            ramlookup = pd.read_csv('ramLookup.csv')
            prolookup = pd.read_csv('processorsLookup.csv')
            dlen = len(dataset)
            dellLen = len(dataset1)
            priority = [0]*dlen
            speedCol = [0]*dlen
            perfCol = [0]*dlen
            dataset["Priority"] = priority
            dataset["Performance"] = perfCol
            dataset["Speed"] = speedCol
            speedCol = [0]*dellLen
            perfCol = [0]*dellLen
            dataset1["Performance"] = perfCol
            dataset1["Speed"] = speedCol
            rcount = {}
            ramDict = {}
            proDict = {}
            for i in range(len(ramlookup)):
                ramDict[ramlookup.at[i, "ram"]] = ramlookup.at[i, "speed"]
            for i in range(len(prolookup)):
                proDict[prolookup.at[i, "ProcessorName"]] = prolookup.at[i, "Performance"]
            for i in range(dlen):
                p = dataset.at[i, "Price"]
                k = int(p//10000)
                if k in rcount:
                    rcount[k] = rcount[k]+1
                else:
                    rcount[k] = 1
            key_max = max(rcount.keys(), key=(lambda kk: rcount[kk]))
            valmax = len(prolookup) + len(ramlookup) + rcount[key_max]+1
            valmax *= 2
            for i in range(dlen):
                p = dataset.at[i, "Price"]
                r = dataset.at[i, "Ram"]
                cp = dataset.at[i, "Cpu"]
                k = int(p//10000)
                priori = valmax-float(rcount[k])-math.ceil(proDict[cp]/4)-ramDict[r]
                dataset.at[i, "Performance"] = proDict[cp]
                dataset.at[i, "Speed"] = ramDict[r]
                dataset.at[i, "Priority"] = round(priori, 1)
            for i in range(dellLen):
                r = dataset1.at[i, "Ram"]
                cp = dataset1.at[i, "Cpu"]
                dataset1.at[i, "Performance"] = proDict[cp]
                dataset1.at[i, "Speed"] = ramDict[r]
            #X_train = dataset.iloc[:, [3, 4, 6, 7]].values
            X_train = dataset.iloc[:, [3, 1, 5, 6]].values
            #y_train = dataset.iloc[:, 6].values
            y_train = dataset.iloc[:, 4].values
            X_test = dataset1.iloc[:, [3, 4, 12, 13]].values
            X_test1 = dataset1.iloc[:, [3, 4, 12, 13]].values
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(
                n_neighbors=5, metric='minkowski', p=2)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            dataset1["Priority"] = y_pred
            typeDict = {"2 in 1 Convertible": 2,
                        "Gaming": 15,
                        "Netbook": 5,
                        "Notebook": 5,
                        "Ultrabook": 7,
                        "Workstation": 10}
            for i in range(dellLen):
                p = dataset1.at[i, "Price"]
                k = int(p//10000)
                if not k in rcount:
                    dataset1.at[i, "Priority"] += 100
                dataset1.at[i, "Priority"] -= typeDict[dataset1.at[i, "TypeName"]]
            dataset1 = dataset1.sort_values(by=["Priority"])
            dataset1.to_csv("predicted.csv",index=False)
            delltops = pd.read_csv("predicted.csv")
            acc = pd.read_csv("accessories.csv")
            delltops_len = len(delltops)
            mouseCol = ["na"] * delltops_len
            keyboardCol = ["na"] * delltops_len
            headsetCol = ["na"] * delltops_len
            delltops["Mouse"] = mouseCol
            delltops["Keyboard"] = keyboardCol
            delltops["Headset"] = headsetCol
            accCounts = {
                1: {"mouse": 0, "keyboard": 0, "headset": 0},
                2: {"mouse": 0, "keyboard": 0, "headset": 0},
                3: {"mouse": 0, "keyboard": 0, "headset": 0},
                4: {"mouse": 0, "keyboard": 0, "headset": 0},
            }
            mouses = []
            keyboards = []
            headsets = []
            for i in range(len(acc)):
                if acc.at[i, "Type"] == "mouse":
                    mouses.append(acc.at[i, "Product"])
                if acc.at[i, "Type"] == "keyboard":
                    keyboards.append(acc.at[i, "Product"])
                if acc.at[i, "Type"] == "headset":
                    headsets.append(acc.at[i, "Product"])
                if acc.at[i, "Range"] == 1:
                    accCounts[1][acc.at[i, "Type"]] += 1
                if acc.at[i, "Range"] == 2:
                    accCounts[2][acc.at[i, "Type"]] += 1
                if acc.at[i, "Range"] == 3:
                    accCounts[3][acc.at[i, "Type"]] += 1
                if acc.at[i, "Range"] == 4:
                    accCounts[4][acc.at[i, "Type"]] += 1
            for i in range(delltops_len):
                price = delltops.at[i, "Price"]
                posm = 1
                posk = 1
                posh = 1
                if price >= 20000 and price < 40000:
                    posm = round((1 - ((40000 - price) / 20000)) * accCounts[1]["mouse"])
                    posk = round((1 - ((40000 - price) / 20000)) * accCounts[1]["keyboard"])
                    posh = round((1 - ((40000 - price) / 20000)) * accCounts[1]["headset"])
                if price >= 40000 and price < 60000:
                    posm = round((1 - ((60000 - price) / 20000)) * accCounts[2]["mouse"])
                    posk = round((1 - ((60000 - price) / 20000)) * accCounts[2]["keyboard"])
                    posh = round((1 - ((60000 - price) / 20000)) * accCounts[2]["headset"])
                if price >= 60000 and price < 80000:
                    posm = round((1 - ((80000 - price) / 20000)) * accCounts[3]["mouse"])
                    posk = round((1 - ((80000 - price) / 20000)) * accCounts[3]["keyboard"])
                    posh = round((1 - ((80000 - price) / 20000)) * accCounts[3]["headset"])
                if price >= 80000 and price < 100000:
                    posm = round((1 - ((100000 - price) / 20000)) * accCounts[4]["mouse"])
                    posk = round((1 - ((100000 - price) / 20000)) * accCounts[4]["keyboard"])
                    posh = round((1 - ((100000 - price) / 20000)) * accCounts[4]["headset"])
                if price >= 100000:
                    posm = round((1 - ((500000 - price) / 500000)) * accCounts[4]["mouse"])
                    posk = round((1 - ((500000 - price) / 500000)) * accCounts[4]["keyboard"])
                    posh = round((1 - ((500000 - price) / 500000)) * accCounts[4]["headset"])
                delltops.at[i, "Mouse"] = mouses[int(posm)]
                delltops.at[i, "Headset"] = headsets[int(posh)]
                delltops.at[i, "Keyboard"] = keyboards[int(posk)]
                delltops.to_csv("predicted.csv",index=False)
            if final_min == min_old and final_max == max_old:
                predicted=pd.read_csv("predicted.csv")
                rate=pd.read_csv("reviews.csv")
                rtl=len(rate)
                for x in reversed(rate.index):
                    if x>(rtl-5):
                        pastRating=rate.at[x,"Rating"]
                        print(pastRating)
                        times=int((5-float(pastRating))*2)
                        predindexes=[]
                        c=1
                        for i,j in predicted.iterrows():
                            predicted.at[i,"Priority"]=j.Priority+50
                            c+=1
                            if c==times:
                                break
                        predicted=predicted.sort_values(by=["Priority"])
                predicted.to_csv("predicted.csv",index=False)     
            else:
                rate=pd.read_csv("reviews.csv")
                rate.drop(rate.index,inplace=True)
                rate.to_csv("reviews.csv",index=False)
            
            
            final = pd.read_csv("predicted.csv")
            pid = []
            result = []
            myRecommendation = []
            data = laptop.objects.all()
            for i in range(12):
                pid.append(final.at[i, "Id"])
            for p in pid:
                dt = laptop.objects.get(id=int(p))
                if dt.recommended !=0:
                    myRecommendation.append([dt.id,(dt.checkout/dt.recommended)])
                else:
                    myRecommendation.append([dt.id,0])
            
            for x in range(len(myRecommendation)):
                for y in range(len(myRecommendation)-1):
                    if myRecommendation[y][1]<myRecommendation[y+1][1]:
                        temp = myRecommendation[y]
                        myRecommendation[y] = myRecommendation[y+1]
                        myRecommendation[y+1] = temp

            for x in range(len(myRecommendation)):
                dt = laptop.objects.get(id=myRecommendation[x][0])
                if dt.recommended is None:
                    dt.recommended = 1
                else:
                    dt.recommended = dt.recommended + 1
                dt.save()
                result.append(dt)
            return render(request, 'base.html', {'list': result})
    else:
        return redirect('Home')

def alllaptop(request):
    list = laptop.objects.all()
    page = request.GET.get('page')

    paginator = Paginator(list, 12)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'alltops.html',{'list': users})
    

def allaccessory(request):
    lists = accessory.objects.all()
    for lt in lists:
        lt.image = lt.product+".jpg"
        lt.save()
    return render(request, 'allaccs.html',{'list':lists})

def cartMethod(request):
    lists = []
    if request.method == 'POST':
        if request.POST["clicked_items"] != "" :
            data = laptop.objects.all()
            try:
                lastcart = cart.objects.first()
                lastcartstr = lastcart.lastAdded
                strn = set((request.POST["clicked_items"]+lastcartstr).split(" "))
                strng = list(strn)
                strng.pop(0)
                for st in strng:
                    lists.append(laptop.objects.get(id=int(st)))
                    lp = laptop.objects.get(id=int(st))
                    if lp.selected is None:
                        lp.selected = 1
                    else:
                        lp.selected = lp.selected + 1
                    lp.save()
                    lastcart.lastAdded = (request.POST["clicked_items"]+lastcartstr)
                    lastcart.save()
            except:
                strng = (request.POST["clicked_items"]).split(" ")
                strng.pop(0)
                for st in strng:
                    lists.append(laptop.objects.get(id=int(st)))
                    lp = laptop.objects.get(id=int(st))
                    if lp.selected is None:
                        lp.selected = 1
                    else:
                        lp.selected = lp.selected + 1
                    lp.save()
                    c = cart(lastAdded = (request.POST["clicked_items"]))
                    c.save()
        else:
            try:
                c = cart.objects.first()
                strng = c.lastAdded.split(" ")
                for st in strng:
                    if st != "":
                        lists.append(laptop.objects.get(id=int(st)))
            except:
                lists = []
    else:
        try:
            c = cart.objects.first()
            strng = c.lastAdded.split(" ")
            for st in strng:
                if st != "":
                    lists.append(laptop.objects.get(id=int(st)))
        except:
            lists = []

    return render(request, 'cart.html',{'list':lists})

def rmcartMethod(request):
    if request.method == "POST":
        strng = request.POST["rmvtxt"]
        c = cart.objects.first()
        s = c.lastAdded.split(" ")
        s.remove(strng)
        nw = ""
        for ss in s:
            nw = nw + " " + ss
        c.lastAdded = nw
        c.save()
    return redirect('Cart')

def analytics_view(request):
    var = laptop.objects.all()
    
    gm1,gm2,gm3 = 0,0,0
    nb1,nb2,nb3 = 0,0,0
    wk1,wk2,wk3 = 0,0,0,
    tino1,tino2,tino3 = 0,0,0,
    netb1,netb2,netb3 = 0,0,0
    ub1,ub2,ub3 = 0,0,0
    for x in var:
        if x.typename == "Gaming" :
            if x.recommended is not None and x.selected is not None:
                gm1 = gm1 + x.recommended
                gm2 = gm2 + x.selected
                gm3 = gm3 + x.checkout
        elif x.typename == "Notebook" :
            if x.recommended is not None and x.selected is not None:
                nb1 = nb1 + x.recommended
                nb2 = nb2 + x.selected
                nb3 = nb3 + x.checkout
        elif x.typename == "Workstation" :
            if x.recommended is not None and x.selected is not None:
                wk1 = wk1 + x.recommended
                wk2 = wk2 + x.selected
                wk3 = wk3 + x.checkout
        elif x.typename == "2 in 1 Convertible" :
            if x.recommended is not None and x.selected is not None:
                tino1 = tino1 + x.recommended
                tino2 = tino2 + x.selected
                tino3 = tino3 + x.checkout
        elif x.typename == "Netbook" :
            if x.recommended is not None and x.selected is not None:
                netb1 = netb1 + x.recommended
                netb2 = netb2 + x.selected
                netb3 = netb3 + x.checkout
        elif x.typename == "Ultrabook" :
            if x.recommended is not None and x.selected is not None:
                ub1 = ub1 + x.recommended
                ub2 = ub2 + x.selected
                ub3 = ub3 + x.checkout

    xaxisthings=["Gaming","Notebook","Workstation","2 in 1 Convertible","Netbook","Ultrabook"]
    yaxisthingsR=[gm1,nb1,wk1,tino1,netb1,ub1]
    yaxisthingsS=[gm2,nb2,wk2,tino2,netb2,ub2]
    yaxisthingsC=[gm3,nb3,wk3,tino3,netb3,ub3]

    ydata1 = {
        'name': 'Recommendation Count',
        'data': yaxisthingsR,
        'color': 'blue'
    }
    ydata2 = {
        'name': 'Added to Cart Count',
        'data': yaxisthingsS,
        'color': 'orange'
    }
    ydata3 = {
        'name': 'Checkout Count',
        'data': yaxisthingsC,
        'color': 'greeen'
    }

    chart = {
        'chart': {'type': 'column'},
        'title': {'text': 'Category vs Counts'},
        'xAxis': {'categories': xaxisthings},
        'series': [ydata1,ydata2,ydata3]
    }

    dump = json.dumps(chart)

    ins = 0
    xps = 0
    lat = 0
    al = 0
    cb = 0
    vostro = 0
    ps = 0
    for x in var:
        if x.product.find("Inspiron") != -1 :
            if x.recommended is not None:
                ins = ins + x.recommended
        elif x.product.find("XPS") != -1 :
            if x.recommended is not None:
                xps = xps + x.recommended
        elif x.product.find("Latitude") != -1 :
            if x.recommended is not None:
                lat = lat + x.recommended
        elif x.product.find("Alienware") != -1 :
            if x.recommended is not None:
                al = al + x.recommended
        elif x.product.find("Chromebook") != -1 :
            if x.recommended is not None:
                cb = cb + x.recommended
        elif x.product.find("Vostro") != -1 :
            if x.recommended is not None:
                vostro = vostro + x.recommended
        elif x.product.find("Precision") != -1 :
            if x.recommended is not None:
                ps = ps + x.recommended

    xaxisthings1=["Inspiron Series","XPS","Latitude","Alienware","Chromebook","Vostro","Precision"]
    yaxisthings1=[ins,xps,lat,al,cb,vostro,ps]
    
    ydata1 = {
        'name': 'Recommendation Count',
        'data': yaxisthings1,
        'color': 'green'
    }

    chart1 = {
        'chart': {'type': 'column'},
        'title': {'text': 'Laptop Series vs Recommendation'},
        'xAxis': {'categories': xaxisthings1},
        'series': [ydata1]
    }
    dump1 = json.dumps(chart1)

    xaxisthings3=[]
    yaxisthings3=[]
    data = laptop.objects.all()
    for dt in data:
        if dt.selected is not None:
            if dt.selected !=0 and dt.recommended !=0:
                yaxisthings3.append(dt.selected/dt.recommended)
                xaxisthings3.append(dt.product)
    
    ydata2 = {
        'name': 'Conversion Rate',
        'data': yaxisthings3,
        'color': 'orange'
    }

    chart2 = {
        'chart': {'type': 'column'},
        'title': {'text': 'Laptops vs Conversion Rate'},
        'xAxis': {'categories': xaxisthings3},
        'series': [ydata2]
    }
    dump3 = json.dumps(chart2)

    successrate = useraction.objects.first()
    if successrate.checkedoutCount !=0 and successrate.visitedCount !=0:
        value = (successrate.checkedoutCount/(successrate.visitedCount/2))*100
    else:
        value = 0

    alllaps = laptop.objects.all()
    tr = 0
    for dt in alllaps:
        tr = tr + (dt.checkout * dt.price)

    return render(request, 'analytics.html', {'chart':dump,'chart1':dump1,'chart2':dump3,'value':value,'tr':tr})

def checkout(request):
    ua = useraction.objects.first()
    ua.checkedoutCount = ua.checkedoutCount+1
    ua.save()
    var1 = ""
    var2 = ""
    c = cart.objects.first()
    if c is not None:
        strng = c.lastAdded.split(" ")
        if strng[0] == "":
            strng.pop(0)
        for st in strng:
            if st != "":
                dt = laptop.objects.get(id=int(st))
                dt.checkout = dt.checkout + 1
                dt.save()
        var1 = "Sucessfully Checked Out !!"
        cart.objects.all().delete()
    else:
        var2 = "No items in cart !!"

    return render(request, 'checkout.html',{'text1': var1,'text2':var2})
