import numpy as np
import pandas as pd
import nltk
import csv
import re
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem.lancaster import LancasterStemmer
from pandas import DataFrame
import nltk.stem
from nltk.probability import FreqDist
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import time
#read the file
s = nltk.stem.SnowballStemmer('english')
with open('Exercise2_data\\twitter-train-cleansed-A.tsv','r') as f:
    raw = f.readlines()
with open('Exercise2_data\\twitter-dev-gold-A.tsv','r') as f:
    test = f.readlines()
with open ('Exercise2_data\\twitter-test-A.tsv','r') as f:
    test1 = f.readlines()
arr = {}
result_1 = open('Result_task_svm_1.csv','w')
result_2 = open('Result_task_svm_2.csv','w')

id_1=[]
id_1_1=[]
id_2=[]
id_2_2=[]
id_3=[]
id_3_3=[]
#text processing
def pre_processing(raw):
    #划分六个区域（根据数据）
    nu1=[]
    nu2 = []
    data =[]
    label =[]
    unk ="unknown"
    for line in raw:
        # Divdide four parts in orgianal data
        lon_num, short_num, num1, num2,result, content = line.split("\t")

        cont_split = content.split(" ")
        content = content.lower()
        content = re.sub(r"@[^\s]+","",content,flags = re.I)#clean @
        content = re.sub(r"[+\.\/_,$%^*()+?\"!:]+|[+——！，。?、~#￥%……&*]+",' ',content,flags=re.I)#clean punctiation
        content = re.sub(r"^http","URL",content,flags = re.I)#clean URL
        content = re.sub(r"'",'',content,flags = re.I)
        content = content.strip()
        number =3
        #print(data,content)
        if result == 'positive':
            rem=0
            tem = nltk.word_tokenize(content)
            phrase=tem[int(num1):int(num2) + 1]
            '''
            if(num1-number>=0):
                num1=num1-number
            else:
                x=number-num1
                num1=0
                for i in range(x-1):
                    phrase.append()
         '''
            data.append(str(phrase))
            label.append(result)

        elif result == 'negative':
            tem = nltk.word_tokenize(content)
            phrase = tem[int(num1):int(num2) + 1]
            #print(str(phrase))
            data.append(str(phrase))
            label.append(result)
        elif result == "neutral":
            tem = nltk.word_tokenize(content)
            phrase = tem[int(num1):int(num2) + 1]
            data.append(str(phrase))
            label.append(result)
        else:
            tem = nltk.word_tokenize(content)
            phrase = tem[int(num1):int(num2) + 1]
            data.append(str(phrase))
            label.append("unknown")

        nu1.append(lon_num)
        nu2.append(short_num)
    return data,label,nu1,nu2
train_data,train_labels,id_1,id_1_1=pre_processing(raw)


test_data,test_labels,id_3,id_3_3=pre_processing(test)
test_data1,test_labels1,id_2,id_2_2=pre_processing(test1)
#Feature Exarction and transfer into the ti-idf vector
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_data+test_data)
test_vectors = vectorizer.transform(test_data)
test1_vectors = vectorizer.transform(test_data1)
train_labels=train_labels+test_labels

 # Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test1_vectors)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test1_vectors)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1


print("Results for SVC(kernel=linear)")
# Print results in a nice table for test set with marked labels
#print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
#print(classification_report(train_labels, prediction_linear))

for i in range(len(prediction_linear)):
    result_1.write('%s,%s,%s\n'%(id_2[i],id_2_2[i],prediction_linear[i]))


print("Results for LinearSVC()")
# Print results in a nice table for test set with marked labels
#print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
#print(classification_report(train_labels, prediction_liblinear))
for i in range(len(prediction_liblinear)):
    result_2.write('%s,%s,%s\n'%(id_2[i],id_2_2[i],prediction_liblinear[i]))
result_1.close()
result_2.close()