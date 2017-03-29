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


#unigram functon
def unigrams(input):
    uni = {}

    for word in input:
        if word in uni:
             uni[word] = uni[word] + 1 / len(input)
        else:
             uni[word] = 1 / len(input)

    count = 0

    pos_gram = {}

    c =1001
    for k in sorted(uni,key=lambda x: uni[x],reverse=True):
        #if count < 200:

         pos_gram[k]=uni[k]
         #print(k,pos_gram[k])
         count += 1

        #else:
          #break
    return pos_gram

with open(
        'Exercise2_data\\twitter-train-cleansed-B.tsv',
        'r') as f:
    raw = f.readlines()

with open(
        'Exercise2_data\\twitter-dev-gold-B.tsv',
        'r') as f:
    raw1 = f.readlines()
raw=raw+raw1

arr = {}
con_sum = []
c =1001
positive = open('Exercise2_data\\positive.txt', 'r+')
negative = open('Exercise2_data\\negative.txt', 'r+')
neutural = open('Exercise2_data\\neu.txt', 'r+')
pos_sum = 0
neg_sum = 0
neu_sum = 0


#textprocessing
for line in raw:
    # Divdide six parts in orgianal data
    lon_num, short_num, result, content = line.split("\t")
    arr = {result: content}
    cont_split = content.split()
    con_sum = con_sum + cont_split
    content = content.lower()
    content = re.sub(r"@[^\s]+","",content,flags = re.I)
    content = re.sub(r"[+\.\/_,$%^*()+?\"!:]+|[+——！，。？、~#￥%……&*]+",' ',content,flags=re.I)
    content = re.sub(r"^http","URL",content,flags = re.I)
    content = re.sub(r"'",'',content,flags = re.I)
    content = content.strip()
    if result == 'positive':
        pos_sum += 1
        positive.write(('%s' % (content)))
    elif result == "negative":
        neg_sum += 1
        negative.write(('%s' % (content)))
    elif result == "neutral":
        neu_sum += 1
        neutural.write(('%s' % (content)))
        # print(arr)
print(pos_sum,neg_sum,neu_sum)

con_sum2 = []
con_sum3 = []
con_sum4 = []
pos_word = 0
neg_word = 0
neu_word = 0
for line in positive:
    positive_split = line.split()
    con_sum2 = con_sum2 + positive_split
    pos_word+=1
dF = DataFrame(con_sum2)
dF.columns = ['Words']
# print(con_sum2)
# tokenizer = nltk.RegexpTokenizer(r'w+')
for line in negative:
    negative_split = line.split()
    con_sum3 = con_sum3 + negative_split
    neg_word += 1
for line in neutural:
    neu_split = line.split()
    con_sum4 = con_sum4 + neu_split

result_1 = open('Result_bayes_B.csv','w')
#training the unigram model
pos_uni = {}
pos_uni = unigrams(con_sum2)
neg_uni = {}
neg_uni = unigrams(con_sum3)
neu_uni = {}
neu_uni = unigrams(con_sum4)

Sum =pos_sum+neg_sum+neu_sum
pos=pos_sum
neg=neg_sum
neu=neu_sum
result1=[]
acc=0
count=0
mini=3/(len(con_sum2)+len(con_sum3)+len(con_sum4))
#p = open('S:\\CS918\\Assignment2\\Exercise2_data\\twitter-train-cleansed-B.tsv', 'r')
p = open('S:\\CS918\\Assignment2\\Exercise2_data\\twitter-test-B.tsv', 'r')
#Naive Bayes training and apply it to the test set
for line in p:

    lon_num, short_num, result, content = line.split("\t")
    arr = {result: content}
    Sentence = content.split(" ")
    #print(line)
    pos = pos_sum
    neg = neg_sum
    neu = neu_sum
    #print(pos, neg, neu)
    for i in Sentence:
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
    #print(pos, neg, neu)
    maximum = max(pos, neg, neu)
    # Validation of known test
    '''
    if maximum == pos:

        if result == "positive":
            acc += 1
    elif maximum == neg:
        if result == "negative":
            acc += 1
    elif maximum == neu:
        if result == "neutral":
            acc += 1
    count += 1
    '''
    if maximum == pos:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "positive"))
        # if result =="positive":

        # acc+=1
    elif maximum == neg:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "negative"))
        # if result == "negative":
        # acc += 1
    elif maximum == neu:

        result_1.write('%s,%s,%s\n' % (lon_num, short_num, "neutral"))
        # if result == "neutral":
        # acc += 1
#Validation of known test
#print("accuracy",acc,count,acc/count)

positive.close()
negative.close()
neutural.close()
p.close()