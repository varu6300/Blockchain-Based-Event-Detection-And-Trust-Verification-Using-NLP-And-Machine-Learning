from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from datetime import date
import json
from web3 import Web3, HTTPProvider
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xg
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
import io
import base64

global details, username
details=''


#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

global accuracy, precision, recall, fscore

accuracy = []
precision = []
recall = []
fscore = []

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

dataset = pd.read_csv("Dataset/DisasterTweets.csv")
dataset = dataset.dropna()
print(dataset.shape)
dataset = dataset.values
'''
X = []
Y = []

for i in range(len(dataset)):
    tweet = dataset[i,3]
    tweet = tweet.strip("\n").strip().lower()
    label = dataset[i,4]
    tweet = cleanText(tweet)#clean description
    X.append(tweet)
    Y.append(label)
    print(str(i)+" "+str(len(tweet))+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X", X)
np.save("model/Y", Y)
'''

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
print(X[12])
print(Y)

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
X = tfidf_vectorizer.fit_transform(X).toarray()


sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

def calculateMetrics(algorithm, predict, y_test):
    label = ['Normal Event', 'Disaster Event']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

nb_cls = GaussianNB()
nb_cls.fit(X_train, y_train)
predict = nb_cls.predict(X_test)
calculateMetrics("Naive Bayes", predict, y_test)    

knn_cls = KNeighborsClassifier(n_neighbors=2) #create KNN object
knn_cls.fit(X_train, y_train)
predict = knn_cls.predict(X_test)
calculateMetrics("KNN", predict, y_test)   

svm_cls = svm.SVC() #create SVM object
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)
    
lr_cls = LogisticRegression(max_iter=300) #create Logistic Regression object
lr_cls.fit(X_train, y_train)
predict = lr_cls.predict(X_test)
calculateMetrics("Logistic Regression", predict, y_test)

dt_cls = DecisionTreeClassifier() #create Decision Tree object
dt_cls.fit(X_train, y_train)
predict = dt_cls.predict(X_test)
calculateMetrics("Decision Tree", predict, y_test)

    
rf_cls = RandomForestClassifier() #create Random Forest object
rf_cls.fit(X_train, y_train)
predict = rf_cls.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

xg_cls = xg.XGBClassifier() #create XGBOost object
xg_cls.fit(X_train, y_train)
predict = xg_cls.predict(X_test)
calculateMetrics("XGBoost", predict, y_test)


X_train1 = np.reshape(X_train, (X_train.shape[0], 10, 10, 3))
X_test1 = np.reshape(X_test, (X_test.shape[0], 10, 10, 3))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

dl_model = Sequential()
dl_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
dl_model.add(MaxPooling2D(pool_size = (2, 2)))
dl_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
dl_model.add(MaxPooling2D(pool_size = (2, 2)))
dl_model.add(Flatten())
dl_model.add(Dense(units = 256, activation = 'relu'))
dl_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
if os.path.exists("model/dl_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/dl_weights.hdf5', verbose = 1, save_best_only = True)
    hist = dl_model.fit(X_train1, y_train1, batch_size = 16, epochs = 150, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/dl_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    dl_model.load_weights("model/dl_weights.hdf5")

predict = dl_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
calculateMetrics("DL", predict, y_test1)


def readDetails(contract_type):
    global details
    details = ""
    print(contract_type+"======================")
    blockchain_address = 'http://127.0.0.1:9545' #Blokchain connection IP
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'Event.json' #event contract code
    deployed_contract_address = '0x0c72340F11c4b725869d070052ba40A0003f61D5' #hash address to access agriculture contract
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi) #now calling contract to access data
    if contract_type == 'signup':
        details = contract.functions.getUser().call()
    if contract_type == 'event':
        details = contract.functions.getPost().call()
    print(details)    

def saveDataBlockChain(currentData, contract_type):
    global details
    global contract
    details = ""
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'Event.json' #event contract file
    deployed_contract_address = '0x0c72340F11c4b725869d070052ba40A0003f61D5' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    readDetails(contract_type)
    if contract_type == 'signup':
        details+=currentData
        msg = contract.functions.addUser(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'event':
        details+=currentData
        msg = contract.functions.setPost(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)   

def Graph(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        df = pd.DataFrame([['Naive Bayes','Precision',precision[0]],['Naive Bayes','Recall',recall[0]],['Naive Bayes','F1 Score',fscore[0]],['Naive Bayes','Accuracy',accuracy[0]],
                           ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                           ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                           ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','F1 Score',fscore[3]],['Logistic Regression','Accuracy',accuracy[3]],
                           ['Decision Tree','Precision',precision[4]],['Decision Tree','Recall',recall[4]],['Decision Tree','F1 Score',fscore[4]],['Decision Tree','Accuracy',accuracy[4]],
                           ['Random Forest','Precision',precision[5]],['Random Forest','Recall',recall[5]],['Random Forest','F1 Score',fscore[5]],['Random Forest','Accuracy',accuracy[5]],
                           ['XGBoost','Precision',precision[6]],['XGBoost','Recall',recall[6]],['XGBoost','F1 Score',fscore[6]],['XGBoost','Accuracy',accuracy[6]],
                           ['Deep Learning (DL)','Precision',precision[7]],['Deep Learning (DL)','Recall',recall[7]],['Deep Learning (DL)','F1 Score',fscore[7]],['Deep Learning (DL)','Accuracy',accuracy[7]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        #plt.show()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data': img_b64}
        return render(request, 'ViewGraph.html', context)   

def TrainML(request):
    if request.method == 'GET':
        global precision, recall, fscore, accuracy
        algorithms = ['Naive Bayes', 'KNN', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Deep Learning (DL)']
        output = ""
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(accuracy[i])+'</td>'
            output+='<td><font size="" color="black">'+str(precision[i])+'</td>'
            output+='<td><font size="" color="black">'+str(recall[i])+'</td>'
            output+='<td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)    


def ViewTweets(request):
    if request.method == 'GET':
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Username</font></th>'
        output+='<th><font size=3 color=black>Tweet Post</font></th>'
        output+='<th><font size=3 color=black>Posted Date</font></th>'
        output+='<th><font size=3 color=black>Event Classification</font></th></tr>'
        readDetails("event")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("$")
            output+='<tr><td><font size=3 color=black>'+arr[0]+'</font></td>'
            output+='<td><font size=3 color=black>'+arr[1]+'</font></td>'
            output+='<td><font size=3 color=black>'+str(arr[2])+'</font></td>'
            output+='<td><font size=3 color=black>'+str(arr[3])+'</font></td></tr>'                    
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def PredictAction(request):
    if request.method == 'POST':
        global username
        tweet = request.POST.get('t1', False)
        data = tweet.strip().lower()
        data = cleanText(data)
        temp = []
        temp.append(data)
        temp = tfidf_vectorizer.transform(temp).toarray()
        dl_model = load_model("model/dl_weights.hdf5")
        temp = sc.transform(temp)
        temp = np.reshape(temp, (temp.shape[0], 10, 10, 3))
        predict = dl_model.predict(temp)
        predict = np.argmax(predict)
        print(predict)
        output = "Normal Event Detected"
        if predict == 1:
            output = "Disaster Event Detected"
        today = date.today()
        data = username+"$"+tweet+"$"+str(today)+"$"+output+"\n"
        saveDataBlockChain(data,"event")
        context= {'data': 'Your Post Classified as : '+output}
        return render(request, 'Predict.html', context)         

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        record = 'none'
        readDetails("signup")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == "signup":
                if arr[1] == username:
                    record = "exists"
                    break
        if record == 'none':
            data = "signup#"+username+"#"+password+"#"+contact+"#"+email+"#"+address+"\n"
            saveDataBlockChain(data,"signup")
            context= {'data':'Signup process completd and record saved in Blockchain'}
            return render(request, 'Register.html', context)
        else:
            context= {'data':username+'Username already exists'}
            return render(request, 'Register.html', context)    

def UserLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "UserLogin.html"
        context= {'data':'Invalid login details'}
        readDetails("signup")
        rows = details.split("\n")
        for i in range(len(rows)-1):
            arr = rows[i].split("#")
            if arr[0] == "signup":
                if arr[1] == username and arr[2] == password:
                    context = {'data':"Welcome "+username}
                    status = 'UserScreen.html'
                    break
        return render(request, status, context)              


