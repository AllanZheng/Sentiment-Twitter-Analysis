import numpy as np
import pandas as pd
import nltk
import csv
import re
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem.lancaster import LancasterStemmer
from pandas import DataFrame
import codecs
from nltk.probability import FreqDist
import pickle
from decimal import *
# read the file


# Unigram funciton
def unigrams(input):
    uni = {}

    for word in input:
        if word in uni:
             uni[word] = uni[word] + 1 / len(input)
        else:
             uni[word] = 1 / len(input)

    count = 0

    pos_gram = {}
    #low = 1/len(input)
    c =1001
    for k in sorted(uni,key=lambda x: uni[x],reverse=True):
        if count <=1000:

         pos_gram[k]=uni[k]
         #print(k,pos_gram[k])
         count += 1

        else:
          break;

    return pos_gram#,low

with open(
        'Exercise2_data\\twitter-train-cleansed-A.tsv',
        'r') as f:
    raw = f.readlines()

with open(
        'Exercise2_data\\twitter-dev-gold-A.tsv',
        'r') as f:
    raw1 = f.readlines()
raw=raw+raw1

arr = {}
con_sum = []
c =1001
positive = []
negative = []
neutural = []
pos_sum = 0
neg_sum = 0
neu_sum = 0
# text processing
def pre_process(content):
    #print(content)

    content = content.lower()
    content = re.sub(r"@[^\s]+", "MM", content, flags=re.I)
    #content = re.sub(r"[+\.\/_,$%^*()+?\"!:]+|[+——！，。？、~#￥%……&*]+", ' ', content, flags=re.I)
    content = re.sub(r"http\S+", "URL", content, flags=re.I)
    content = re.sub(r"'", '', content, flags=re.I)
    #re.sub(r"\b[a-z]{1}\b", "", 目标变量名)

    #return cont_split
    return content
pdata=[]
ndata=[]
n1data=[]
id=[]
id_2=[]
for line in raw:
    # Divdide four parts in orgianal data
    lon_num, short_num,num1, num2, result, content = line.split("\n")[0].split("\t")
    arr = {result: content}
    cont_split = content.split(" ")

    num1=int(num1)
    num2=int(num2)

    if result == 'positive':
        pos_sum += 1

        for j in range(num1,num2+1):
            #print(cont_split[j])
            cont_split[j] = pre_process(cont_split[j])
            pdata.append(cont_split[j])

        #words = pre_process(words)
        #print(words)
        #positive.write(('%s' %(words)))
    elif result == "negative":
        neg_sum += 1

            #print(cont_split)
            #print(len(cont_split), i, num2-1)
        for j in range(num1,num2+1):
            #print(cont_split[j])
            cont_split[j] = pre_process(cont_split[j])
            ndata.append(cont_split[j])

        #negative.write(('%s' %(words)))
    elif result == "neutral":
        neu_sum += 1

        for j in range(num1,num2+1):
            #print(cont_split[j])
            cont_split[j] = pre_process(cont_split[j])
            n1data.append(cont_split[j])


        #neu.write(('%s' % (words)))
        # print(arr)
print(pos_sum,neg_sum,neu_sum)


con_sum2 = []
con_sum3 = []
con_sum4 = []
pos_word = 0
neg_word = 0
neu_word = 0

result_1 = open('Result_bayes_A.csv','w')

pos_uni = {}
#pos_uni,pos_low = unigrams(pdata)
pos_uni = unigrams(pdata)
neg_uni = {}
#neg_uni,neg_low = unigrams(ndata)
neg_uni = unigrams(ndata)
neu_uni = {}
neu_uni = unigrams(n1data)

Sum =pos_sum+neg_sum+neu_sum
#S = "@MsSheLahY I didnt want to just pop up... but yep we have chapel hill next wednesday you should come.. and shes great ill tell her you asked"

num1=int(num1)
num2=int(num2)
mini=3/(len(pdata)+len(ndata)+len(n1data))
#p = open('S:\\CS918\\Assignment2\\Exercise2_data\\twitter-dev-gold-A.tsv', 'r')
p = open('Exercise2_data\\twitter-test-A.tsv', 'r')
acc=0
count=0
print(pos_sum,neg_sum,neu_sum)
pp = 0
result1=[]
for line in p:

    lon_num, short_num,num1, num2, result, content = line.split("\t")
    arr = {result: content}
    cont_split = content.split(" ")
    #con_sum = con_sum + cont_split
    #cont_split=pre_process(content)
    num1=int(num1)
    num2=int(num2)
    pos = pos_sum
    neg = neg_sum
    neu = neu_sum
    words=[""]
    for j in range(num1, num2):
        #print(cont_split[j])
        cont_split[j] = pre_process(cont_split[j])
        words.append(cont_split[j])

        #Sentence = line.split()
    #print(line)
    #print(pos, neg, neu)
    for i in words:
       judge = 0

       if i in pos_uni:


            pos=Decimal(pos)*Decimal(pos_uni[i])

       else:
            pos*=Decimal(mini)

       if i in neg_uni:

            neg=Decimal(neg)*Decimal(neg_uni[i])

       else:
            neg *=Decimal(mini)

       if i in neu_uni:

            neu=Decimal(neu) * Decimal( neu_uni[i])

       else:
            neu *=Decimal(mini)
    #
    maximum = max(pos, neg, neu)
    #print(pos, neg, neu,maximum)
    if maximum == pos:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "positive"))
        #if result =="positive":

            #acc+=1
    elif maximum == neg:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "negative"))
        #if result == "negative":
            #acc += 1
    elif maximum == neu:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "neutral"))
        #if result == "neutral":
            #acc += 1
    count+=1
    #print(acc)


#print("accuracy",acc,count,acc/count,pp)

p.close()