import sklearn
import scipy.sparse as sp
import numpy as np
import sys
import json
import pdb
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

#defaultDataFolder = "/home/khalife/ai-lab/data/LDC2015E19_TAC_KBP_English_Entity_Linking_Comprehensive_Training_and_Evaluation_Data_2009-2013/json/backup/"
#category = sys.argv[1] 
#kbFile = open(defaultDataFolder + "knowledgeBaseFile-CorrectTitle.json", "r")
#kbFile = open("knowledgeBaseFile-Updated-2010.json", "r")
kbFile = open("knowledgeBaseFile-Updated-2010-TRAIN.json", "r")

entityUKNIndexToDegree = json.load(open("../entityUKNIndexToDegree.json", "r"))

edgesIdFile = open("../entityIdToIndexFile.json","r")
for line in edgesIdFile:
    entityIdToIndex = json.loads(line)

entityIndexToId = {v: k for k, v in entityIdToIndex.items()}


categories = ["ORG", "PER", "GPE", "UKN"]
text_list = []
knowledgeBase = []

index = 0
for line in kbFile:
    dic_line = json.loads(line)
    if dic_line["entity_type"] in categories:
        try:
            if dic_line["entity_type"] == "UKN":
                if entityUKNIndexToDegree[str(entityIdToIndex[dic_line["entity_id"]])] < 10:
                    continue
            local_dic = {}
            local_dic["entity_id"] = dic_line["entity_id"]
            local_dic["entity_name"] = dic_line["entity_name"]
            local_dic["entity_type"] = dic_line["entity_type"]
            knowledgeBase.append(local_dic)
            text_list.append(dic_line["entity_text"])
            #if index == 218160:
            #    pdb.set_trace()
            index += 1 

        except:
            continue

#pdb.set_trace()
print("Compute TFIDF...")

stop_words = stopwords.words('english')
def processNameString(str1):
       str1_ = str1
       for char in string.punctuation:
              if ( str1_.find(char + " ") > -1 ) or ( str1_.find(" " + char) > -1 ):
                     str1_ = str1_.replace(char, "")
              else:
                     str1_ = str1_.replace(char, " ")
       mention_name_words = str1_.split()
       mention_words = [m_w for m_w in mention_name_words if m_w not in stop_words]      
       return " ".join(mention_words) 
               

nb_max_features = 2000000

Vectorizer = TfidfVectorizer(max_features = nb_max_features)
M_KB_TFIDF = Vectorizer.fit_transform(text_list)
#idfs_category = Vectorizer.idf_
#np.savetxt("idfs-.txt")
#pdb.set_trace()
print("Preparing mentions...")
#np.savetxt("idfs-" + category + ".txt", idfs_category)
sp.save_npz("matrices_tfidf-2010-TRAINING-UPDATE.npz", M_KB_TFIDF)
rx = re.compile(r"[\W]")
def processFullText(str1):
       mention_context_text = re.sub(rx, " ", str1.lower())
       words = mention_context_text.split()
       bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
       return " ".join(bag_of_words)


mentionsText = []
mentionsFile = open("mentionsJsonFileComplete-2010-TRAINING-UPDATE.json", "r")
#mentionsFile = open("mentionsJsonFileComplete-2010-UPDATE.json", "r")
#mention_index = 0
nb_words = 10
for line in mentionsFile:
       dic_line = json.loads(line)
       #mention_text = dic_line["mention_text"]
       mention_text = processFullText(dic_line["mention_full_text"])
       mentionsText.append(mention_text)

mentionsTfIdf = Vectorizer.transform(mentionsText)
#sp.save_npz("mentions_tfidf-2010-UPDATE.npz", mentionsTfIdf)
sp.save_npz("mentions_tfidf-2010-TRAINING-UPDATE.npz", mentionsTfIdf)
