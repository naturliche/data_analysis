# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:06:25 2020

@author: natur
"""

from collections import Counter
import pandas as pd
import math
import csv
import numpy as np
import copy
import re

paci = pd.read_csv('pacifier_chugao.csv')  
paci_seg = paci['review'].values

def lowWord(np_train_seg):   #转为小写
    low_words1 = []
    for i in range(len(np_train_seg)):
        low_words0 = []
        words = np_train_seg[i]
        raw_words = re.split(' |\n',words)
        for word in raw_words:
            word = str(word)
            #print(word)
            #print(type(word))
            word = re.sub('[^a-zA-Z]', '',word)
            word = word.lower()
            low_words0.append(word)  
        low_words1.append(low_words0)
    return low_words1

paci_seg = lowWord(paci_seg)  #每条评论划分好单词并全部转为小写


keyWord = pd.read_csv('sum_tf_idf_del2.csv') 
keyword0 = keyWord['word'].values
tf_i0 = keyWord['tf_idf'].values
keyword = keyword0.tolist()
tf_i = tf_i0.tolist()


#print(keyword)    
#print(tf_i)
#print(type(keyword))    
#print(type(tf_i))

match_dict = dict(zip(keyword,tf_i))  #一个关键字和tf—idf对应
                                        #的字典
#count = 0
#row1 = []

len_key2 = []  
for i in range(len(paci_seg)):  #多条评论的循环
    len_key1 = copy.deepcopy(match_dict)
    for key in len_key1:
        len_key1[key] = 0
        #len_key1变为key=keyword，value=tf_idf的字典
    for j in range(len(paci_seg[i])):#每一天评论的长度
        #for key in match_dict:
        if paci_seg[i][j] in match_dict.keys():
            len_key1[key] +=match_dict[key]
            #print(ad)
            #print(type(ad))
            #ad = str(ad)
            #print(type(ad)
            #if i not in row1:
                #row1.append(i)
        #else:
         #   len_key1[key]
        #print(len_key1)
    len_key2.append(len_key1) #长度为i
    #print(i)
    #print(seg)
    #print(matr)
    
TARGET_FILE = 'cos_pre66667'

csv_file_name = "C:/Users/natur/Desktop/A寒假/美赛/test_b/"+TARGET_FILE+".csv"

with open(csv_file_name,"w",encoding="utf-8") as f:
    csv_file = csv.writer(f,delimiter = "|",quotechar='"',quoting = csv.QUOTE_MINIMAL)
    for j in range(len(paci_seg)):
        a3 = []
        a4 = []
        for m in range(len(keyword)):
            a1 = keyword[m]
            a2 = len_key2[j][key] 
            a3.append(a1)
            a4.append(a2)
        csv_file.writerow([a3,a4])

