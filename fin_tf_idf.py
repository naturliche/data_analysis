# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:34:47 2020

@author: natur
"""

from collections import Counter
import pandas as pd
import math
import csv
import numpy as np
import copy
import re


df_train = pd.read_csv('pacifier_chugao.csv')  


np_train_seg = df_train['review_body'].values
for i in range(len(df_train['review_headline'])):
    ap = df_train['review_headline'].values
    np_train_seg = np.append(np_train_seg,ap[i])

def idf(count,tol_count):   
    return math.log(tol_count/(count+1))

def lowWord(np_train_seg):   #转为小写
    low_words = []
    for i in range(len(np_train_seg)):
        raw_words = np_train_seg[i].split(" ")
        for word in raw_words:
            word = str(word)
            #print(word)
            #print(type(word))
            word = re.sub('[^a-zA-Z]', '',word)
            word = word.lower()
            low_words.append(word)          
    return low_words

def tf_idf(ptf,pidf):
    fin = ptf*pidf
    return fin

fina_word1 = lowWord(np_train_seg)
#print(fina_word1)

c = Counter(fina_word1)   #对每一个词进行统计次数，是一个字典
#print(c)
#逆文档频率
idf1 = {}
for key,value in c.items():
    idf1[key] = idf(value,38562)


c1 = copy.deepcopy(c)
for value in c1:
    c1[value] = c1[value]/sum(c1.values())    #对每一个词统计频率

#词频分析
final_tf_idf = copy.deepcopy(c)
for key in final_tf_idf:
    ptf = c1[key]
    pidf = idf1[key]
    final_tf_idf[key] = tf_idf(ptf,pidf)


#TOP_NUM = 38562 
TARGET_FILE = 'fin_tf_idf3'
'''
top_word = c.most_common(TOP_NUM)
top_word1 = c1.most_common(TOP_NUM)
top_word2 = final_tf_idf.most_common(TOP_NUM)
'''

csv_file_name = "C:/Users/natur/Desktop/A寒假/美赛/testa/"+TARGET_FILE+".csv"

with open(csv_file_name,"w",encoding="utf-8") as f:
    csv_file = csv.writer(f,delimiter = "|",quotechar='"',quoting = csv.QUOTE_MINIMAL)
    for key in final_tf_idf:
        count = c[key]
        frequent = c1[key]
        value = final_tf_idf[key]
        csv_file.writerow([key,value,count,frequent])





