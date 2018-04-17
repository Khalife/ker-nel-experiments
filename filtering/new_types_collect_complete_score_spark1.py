# coding: utf8
import sys
import json
import pdb
import numpy as np
import sys
from nltk.corpus import stopwords
import string
import unicodedata
import re
#import gensim
import scipy.spatial.distance as sd
#import optimization as opt
import sklearn 
import unicodedata
from sklearn.externals import joblib
#import stringMetrics
import pandas
import time
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import operator
import string
punctuation = string.punctuation
from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
import pyspark
import unicodedata
sc = SparkContext(appName="nel-system")
sess = pyspark.sql.SparkSession.builder.appName("nel-system").getOrCreate()
punctuation = string.punctuation


nb_words_text = 50
hidden_size = 300

def returnLetterNgram1(str1,n):
    l3grams1 = []
    str2 = str1.replace(" ", "")
    for i in range(max([len(str2)-n+1,1])):
        l3grams1.append(str2[i:i+n])
    return l3grams1

def scoreLetterNgram(str1, str2, n):
    LNgram1 = returnLetterNgram1(str1,n)
    LNgram2 = returnLetterNgram1(str2,n)
    L3 = set(LNgram1).intersection(LNgram2)
    L4 = set(LNgram1).union(set(LNgram2))
    return float(len(L3))/len(L4)

def scoreMixture(name_score, context_score):
    if name_score == 0:
        return context_score
    else:
        return context_score  + 0.05*name_score

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

levenshtein("DHHS", "DHS")

punctuation = string.punctuation
def scoreAcronym(str1, str2):
    # Str1 potential acronym
    # Str2 potential target
    # N gram Jaccard scoring        
    str1_ = str1
    str2_ = str2
    for punct_symb in punctuation:
        str1_ = str1_.replace(punct_symb, " ")
        str2_ = str2_.replace(punct_symb, " ")
    
    str1_ = [st for st in str1_] # acronym letters 
    str2_ = str2_.split() # mention name
    str2_words = [word for word in str2_ if (word not in stop_words) and (word != "")]
    first_letters_1 = "".join([st[0] for st in str1_])
    first_letters_2 = "".join([st[0] for st in str2_words])
    
    len_longest_letters = float(max([len(first_letters_1), len(first_letters_2)]))
    return (len_longest_letters - levenshtein(first_letters_2, first_letters_1))/len_longest_letters
    
    #FL1 = set(first_letters_1).intersection(first_letters_2)
    #FL2 = set(first_letters_1).union(set(first_letters_2))
    #return float(len(FL1))/len(FL2)


defaultDataFolder = sys.argv[1]

id_list = []
name_list = []
type_list = []
text_list = []
#categories = ["PER", "ORG", "GPE", "UKN"]
#category = sys.argv[1]
#print(category)
write_file_name = sys.argv[1]


#entityIdToType = json.load(open("entityIdToOntologyType.json", "r"))
entityIdToType = json.load(open("entityIdToOntologyType-Updated-27-02-18.json", "r"))
entityIdToMainType = json.load(open("entityIdToType-Updated-2010.json", "r"))
categories = list(set([val for val in entityIdToType.values()]))


entityUKNIndexToDegree = json.load(open("entityUKNIndexToDegree.json", "r"))
edgesIdFile = open("entityIdToIndexFile.json","r")
for line in edgesIdFile:
    entityIdToIndex = json.loads(line) 


#kbFile = open(defaultDataFolder + "knowledgeBaseFile-CorrectTitle.json", "r")
#kbFile = open("knowledgeBaseFile-Updated-2010.json", "r")
kbFile = open("knowledgeBaseFile-Updated-2010-TRAIN.json", "r")
knowledgeBase = []

for line in kbFile:
    dic_line = json.loads(line)
    local_dic = {}
    local_dic["entity_id"] = dic_line["entity_id"]
    local_dic["entity_name"] = dic_line["entity_name"]
    local_dic["entity_type"] = dic_line["entity_type"]      
    #if local_dic["entity_type"] in categories:
    try:
        if dic_line["entity_type"] == "UKN":
            if entityUKNIndexToDegree[str(entityIdToIndex[dic_line["entity_id"]])] < 10:
                continue

    except:
        continue
    knowledgeBase.append(local_dic)


print("Compute TFIDF...")

nb_max_features = 2000000


DIC_KB_TFIDF = {}
#mentionsTfIdf = sp.load_npz("mentions_tfidf-2010-UPDATE.npz") 
mentionsTfIdf = sp.load_npz("mentions_tfidf-2010-TRAINING-UPDATE.npz")
#M_KB_TFIDF = sp.load_npz("matrices_tfidf-2010-UPDATE.npz")
M_KB_TFIDF = sp.load_npz("matrices_tfidf-2010-TRAINING-UPDATE.npz")

DIC_KB_TFIDF = {}
index_kb = 0
indexes_kb_to_keep = []
for i_dic in knowledgeBase:
    try:
        if dic_line["entity_type"] == "UKN":
            if entityUKNIndexToDegree[str(entityIdToIndex[dic_line["entity_id"]])] < 10:
                 continue
    except:
        continue
    indexes_kb_to_keep.append(index_kb)
    index_kb += 1

matrix_kb = M_KB_TFIDF[indexes_kb_to_keep,:]


print("TFIDF done")


stop_words = stopwords.words('english')
def processFullText(str1):
    mention_context_text = re.sub(rx, " ", str1.lower())
    words = mention_context_text.split()
    bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
    return " ".join(bag_of_words)


class DistributedTfidfVectorizer(TfidfVectorizer):
    TfidfVectorizer.idf_ = None #idfs_category
    TfidfVectorizer.vocabulary_ = None #vocabulary_category
    def __init__(self, prebuilt_vocabulary=None, prebuilt_idfs=None): #, prebuilt_max_features=2000000):
    #def __init__(self, value=None):
        super(DistributedTfidfVectorizer, self).__init__()#max_features=max_features)
        #self.idf_ = value
        self.idf_ = prebuilt_idfs
        self.vocabulary_ = prebuilt_vocabulary

def reverse_insort_index(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
        
    if lo < 0 or hi <0:
        raise ValueError('lo and hi must be non-negative')
        
        
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    return lo

def reverse_insort(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    
    if lo < 0 or hi <0:
        raise ValueError('lo and hi must be non-negative')
    
    
    while lo < hi:
        mid = (lo+hi)//2
        #print(x)
        #print(a[mid])
        if x > a[mid]: hi = mid
        else: lo = mid+1
    #a.insert(lo, x)
    original_length = len(a)
    
    if lo == len(a):
        a.append(x)
    else:
        a[lo+1:] = a[lo:len(a)]
        a[lo] = x
    return a

def reverse_insort_dic(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    
    if lo < 0 or hi <0:
        raise ValueError('lo and hi must be non-negative')
    
    while lo < hi:
        mid = (lo+hi)//2
        #print(x)
        #print(a[mid])
        if x["score"] > a[mid]["score"]: hi = mid
        else: lo = mid+1
    #a.insert(lo, x)
    original_length = len(a)
    
    if lo == len(a):
        a.append(x)
    else:
        a[lo+1:] = a[lo:len(a)]
        a[lo] = x
    return a

def reverse_insort_key(a, x, key="total_score", lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.
    
    If x is already in a, insert it to the right of the rightmost x.
    
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if hi is None:
        hi = len(a)
    
    if lo < 0 or hi <0:
        raise ValueError('lo and hi must be non-negative')
    
    
    while lo < hi:
        mid = (lo+hi)//2
        if x[key] > a[mid][key]: hi = mid
        else: lo = mid+1
    #a.insert(lo, x)
    original_length = len(a)
    a[lo+1:] = a[lo:(len(a)-1)]
    a[lo] = x
    return a

    
def acronymTest(str1, main_type):
    #score1 = 1/float(len(str1))
    #uppers = ([l for l in str1 if l.isupper()])


    # Get sequence of lenghts between upper letters
    distances = []
    nb_letters = len(str1)
    i = 0
    index = -1
    while i < nb_letters:
        if index < 0:
            if str1[i].isupper():
                index += 1
        else:
            if str1[i].isupper():
                distances.append(i-index)
                index = i
        i += 1

    if main_type == "PER":
        max_distances = 4

    if main_type == "ORG":
        max_distances = 5

    if main_type == "GPE":
        max_distances = 4


    #return (score1 + len(uppers))*10
    return distances, (len(distances) > 0) and ( sum(distances) <= len(distances) ) and ( len(distances) < max_distances ) #and ( len(distances) <= 5 )

#def acronymScore(str1, str2):
#   capital_letters = "".join([l for l in str2 if l.isupper()])
#   #len_longest_letters = float(max([len(str1), len(str2)])    
#   #norm_nb_letters = float(len(str1))
#   norm_nb_letters = float(max([len(str1), len(str2)]))
#   acronym_score = 1 - levenshtein(str1, capital_letters)/norm_nb_letters  
#   ## parameter ??? norm_nb_letters
#   return acronym_score

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return longest
        #return s1[x_longest - longest: x_longest]


def acronymScore1(str1, str2):
    capital_letters = "".join([l for l in str2 if l.isupper()])
    first_words = str2.split()
    first_letters = "".join([fw[0] for fw in first_words])
    norm_nb_letters = min([len(str1), len(str2)])
    
    total_letters = capital_letters
    for letter in first_letters:
        if letter not in capital_letters:
            total_letters = total_letters + letter
    
    lcs1 = longest_common_substring(str1.lower(),total_letters.lower())
    acronym_score = lcs1/norm_nb_letters
    return acronym_score


def longest_common_substring2(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest, y_longest = 0, 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0

    M = [sum(m_) for m_ in m]
    x_longest_start = x_longest - longest + 1
    y_longest_start = m[x_longest].index(longest) - longest + 1
    return longest, [x_longest_start, y_longest_start], x_longest



def unicodeName(name):
    #name_u = unicode(name, 'utf-8')
    name_clean = unicodedata.normalize('NFD', name)
    name_clean = name_clean.encode('ascii', 'ignore')
    return name_clean

import re
rx = re.compile(r"[\W]")

def nelRankingSystem(dic_line):
    
    mention_name = dic_line["mention_name"]
    mention_id = dic_line["mention_id"]
    mention_gold_entity_id = dic_line["gold_entity_id"]
    mention_gold_entity_type = entityIdToType[mention_gold_entity_id]
    mention_index = int(dic_line["mention_index"])
    M_mention_tfidf = mentionsTfIdf[mention_index]  

    list_ranks_categories = {}
    nb_collect = 100
    first_order = False
    row_index = -1
    rank_dic = {}
    #categories = ["ORG", "PER", "GPE", "UKN"]
    list_ranks = []
    gold_dic = {}
    for category in categories:
    #for category in ["ORG", "PER", "GPE", "UKN"]:
        list_ranks_categories[category] = []

    index_kb = -1
    #pdb.set_trace()
    results_tfidf = matrix_kb.dot(M_mention_tfidf.T)
    for i_dic in knowledgeBase:
        index_kb += 1
        context_score = results_tfidf[index_kb][0,0] 

        if mention_gold_entity_type == entityIdToType[i_dic["entity_id"]]:
            #if mention_id == "EL3808":
            #    pdb.set_trace()
            _, acronym_test = acronymTest(mention_name, entityIdToMainType[mention_gold_entity_id])
            _, acronym_test_entity = acronymTest(i_dic["entity_name"], entityIdToMainType[mention_gold_entity_id])
            #closest_clean_entity_name = i_dic["entity_name"].split("_(")[0] 
            closest_clean_entity_name = i_dic["entity_name"]
            closest_clean_entity_name =  unicodeName(closest_clean_entity_name).replace("_", " ")
            closest_clean_entity_name = re.sub(rx, " ", closest_clean_entity_name)
            closest_clean_mention_name = unicodeName(mention_name).replace("_", " ")
            closest_clean_mention_name = re.sub(rx, " ", closest_clean_mention_name)

            if acronym_test or acronym_test_entity:
                #name_score = acronymScore1(mention_name, i_dic["entity_name"])
                #if mention_gold_entity_type == "City":
                #    closest_clean_entity_name = closest_clean_entity_name.split(",_")[0]
                if acronym_test:
                    if acronym_test_entity:
                        #norm_name_factor = min([len(closest_clean_entity_name), len(closest_clean_mention_name)])
                        #name_score = longest_common_substring(closest_clean_entity_name, closest_clean_mention_name)/float(norm_name_factor)
                        name_score = acronymScore1(closest_clean_mention_name, closest_clean_entity_name)
                    else:
                        name_score = acronymScore1(closest_clean_mention_name, closest_clean_entity_name)
 
                else:
                    name_score = acronymScore1(closest_clean_entity_name, closest_clean_mention_name)

            else:
                closest_clean_entity_name = closest_clean_entity_name.lower()
                closest_clean_mention_name = closest_clean_mention_name.lower()
                score2 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 2)
                score3 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 3)
                score4 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 4)
                name_score = (score2 + score3 + score4)/3
                #name_score = longest_common_substring(mention_name.lower(), i_dic["entity_name"].lower())/float(min([len(mention_name),len(i_dic["entity_name"])]))
    
            if i_dic["entity_id"] == mention_gold_entity_id:
                gold_dic = {"mention_gold_entity_id":mention_gold_entity_id}
                gold_score = name_score + context_score
                
        
            #list_ranks_categories[i_dic["entity_type"]] = reverse_insort_dic(list_ranks_categories[i_dic["entity_type"]], {"entity_id" : i_dic["entity_id"], "score" : name_score + context_score})
            list_ranks_categories[entityIdToType[i_dic["entity_id"]]] = reverse_insort_dic(list_ranks_categories[entityIdToType[i_dic["entity_id"]]], {"entity_id" : i_dic["entity_id"], "score" : name_score + context_score}) 
        #else:
            #list_ranks_categories[i_dic["entity_type"]] = reverse_insort_dic(list_ranks_categories[i_dic["entity_type"]], {"entity_id" : i_dic["entity_id"], "score" : context_score})
            #list_ranks_categories[entityIdToType[i_dic["entity_id"]]] = reverse_insort_dic(list_ranks_categories[entityIdToType[i_dic["entity_id"]]], {"entity_id" : i_dic["entity_id"], "score" : context_score}) 
            
        #list_ranks = reverse_insort_dic(list_ranks, name_score + context_score)
        #rank_dic[i_dic["entity_id"]] = name_score + context_score
        #prediction_rank = reverse_insort_index(list_ranks, gold_score)


    return {"mention_id" : mention_id, "gold_dic" : gold_dic, "prediction_ranks" : list_ranks_categories}   
    #return [mention_id, mention_name, gold_dic, gold_score, prediction_rank]
    

mentions_category = []
mentions_file = open(write_file_name, "r")
for line in mentions_file:
    dic_line = json.loads(line)
    #if category in dic_line["gold_entity_type"] and "NIL" not in dic_line["gold_entity_id"]:
    mentions_category.append(dic_line)


print(len(knowledgeBase))
print(mentionsTfIdf.shape)
print(len([dv for dv in DIC_KB_TFIDF]))

## CPU
#mc = None
#for me in mentions_category:
#    if me["mention_id"] == "EL01313":
#        mc = me 
#
##mc = mentions_category[0]
##nelRankingSystem(mc, knowledgeBase, mentionsTfIdf, DIC_KB_TFIDF)
#results1 = nelRankingSystem(mc)
#print(write_file_name)
#pdb.set_trace()

## Spark

mentions_category_spark = sc.parallelize(mentions_category, 200)
mentions_category_results =  mentions_category_spark.map(lambda line: nelRankingSystem(line))

writeFileNameSplit = write_file_name.split("JsonFile")
print(writeFileNameSplit)
writeFileName = writeFileNameSplit[0] + "-NEW-TYPES-02-04-18-1-OWT-Scores-" + writeFileNameSplit[1] 
mentions_category_results.saveAsTextFile(writeFileName)

