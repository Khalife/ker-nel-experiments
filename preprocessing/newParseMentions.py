import json
from lxml import etree
import pandas
from nltk.corpus import stopwords
import os
import sys
import pdb
import time
import re
import unicodedata
import string

defaultDataFolder = "/home/khalife/ai-lab/data/"
years = ["2009", "2010"]
year = years[0]
sequences = ["training", "eval"]
sequence = sequences[1]
sequence1 = sequence
if sequence == "eval":
    sequence1 = "evaluation"

challenge2015 = "LDC2015E19_TAC_KBP_English_Entity_Linking_Comprehensive_Training_and_Evaluation_Data_2009-2013"
#challenge2017 = "LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016"
kbLinksFile =  defaultDataFolder + challenge2015 + "/data/" + year + "/" + sequence +"/tac_kbp_" + year + "_english_entity_linking_" + sequence1 + "_KB_links.tab"
knowledgeBaseFile = defaultDataFolder + "tac_kbp_ref_know_base/data/"
queriesFile = defaultDataFolder + challenge2015 + "/data/" + year + "/" + sequence + "/tac_kbp_" + year + "_english_entity_linking_" + sequence1 + "_queries.xml" 
sourceFile = defaultDataFolder + challenge2015 + "/data/" + year + "/" + sequence + "/source_documents/"
#tac_kbp_2010_english_entity_linking_evaluation_queries.xml
#tac_kbp_2010_english_entity_linking_evaluation_KB_links.tab


#entityIdToTypeFile =  open("entityIdToType.json", "r")

import pdb
import json

#mentionsTypeFile = open("tac_kbp_2010_english_entity_linking_evaluation_KB_links.tab", "r")


#mentionsToType = {}
#for line in mentionsTypeFile:
#   mention_id, gold_entity_id, mention_type = line.split()
#   mentionsToType[mention_id] = mention_type



#mentionsNewFile = open("mentionsJsonFileComplete-2009-UPDATE.json", "w")


#for line in mentionsFile:
    #dic_line = json.loads(line)
    #mention_id = dic_line["mention_id"]
    #dic_line["gold_entity_type"] = mentionsToType[mention_id]
    #mentionsNewFile.write(json.dumps(dic_line) + "\n")
                                                           





nb_words = 20
stop_words = stopwords.words('english')
punctuation = string.punctuation
#text_.strip(punctuation)



def processMentionText(str1, mention_name):
    rx = re.compile(r"[\W]")
    if str1.find(mention_name) > -1 :
        str1_ = str1.replace(mention_name, "REPLACE_STRING_NEL")
        mention_document_words = str1_.split()
        mention_word_indexes = []
        for i, j in enumerate(mention_document_words):
            if j == "REPLACE_STRING_NEL":
                mention_word_indexes.append(i)
        mention_context_words_index = []
        for m_w_i in mention_word_indexes:
            for w in range(max([0, m_w_i - nb_words]), min([len(mention_document_words), m_w_i + nb_words])):
                if mention_document_words[w] != "REPLACE_STRING_NEL" and w not in mention_context_words_index:
                    mention_context_words_index.append(w)
        
        mention_context_words = [mention_document_words[mi] for mi in mention_context_words_index]
        #pdb.set_trace()
        mention_context_text = " ".join(mention_context_words)
        mention_context_text = re.sub(rx, " ", mention_context_text.lower())
        words = mention_context_text.split()
        bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
    
    else:
        mention_context_text = re.sub(rx, " ", str1.lower())
        words = mention_context_text.split()
        bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
    
    return " ".join(bag_of_words)
                
                                    
#def processMentionText(str1, mention_name):
#    rx = re.compile(r"[\W]")
#    pdb.set_trace()
#    try:
#        mention_left, mention_right = str1.split(mention_name)[:2]
#        mention_left_words = mention_left.split()
#        mention_right_words = mention_right.split()
#        nb_left_words = min([nb_words, len(mention_left_words)])
#        mention_left_text = " ".join(mention_left_words[-nb_left_words:])
#        nb_right_words = 2*nb_words - nb_left_words
#        mention_right_text = " ".join(mention_right_words[:min(nb_right_words,len(mention_right_words))])
#        mention_text_context = mention_left_text + " " + mention_right_text
#        
#        mention_text_context = re.sub(rx, " ", mention_text_context.lower())
#        words = mention_text_context.split()                                                                
#        bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
#   
#    except:
#        mention_text_context = re.sub(rx, " ", str1.lower())
#        words = mention_text_context.split()
#        bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]    
#    
#    return " ".join(bag_of_words)

text_test = " Cartoon Network characters to hit cell phones by year ' s end  LOS ANGELES 2007-02-14 22:30:46 UTC  Fans of Cartoon Network shows will be able to interact with their favorite characters through their cell phones by the end of the year.  Cartoon Network New Media, a division of Turner Broadcasting System Inc., announced Wednesday that it is rolling out CallToons, software that takes over a cell phone's main functions and replaces them with character voices, ringtones, wallpaper and other features.  The technology can come packaged with a mobile handset as well as be available as downloads from a carrier, the company, a unit of Time Warner Inc., said.  Unlike typical ringtones or wallpapers, CallToons characters are more tightly integrated with cell phone functions, the company said.  For example, if a phone's battery is weak, instead of the usual static display on the phone's screen, a specific character could appear saying, \"Plug me in, I'm fading fast,\" said Ross Cox, senior director of entertainment products at Cartoon Network New Media.  In another example, a character voice could serve as a ringtone, but deliver a more frustrated response if a user decides to ignore calls from a particular person.  \"It's a narrative voice that responds in real time to the state of the phone and behavior of the owner of the phone,\" Cox said.  Turner is in talks with Swedish mobile phone maker Ericsson regarding delivery options. The service is expected to launch in the fourth quarter.  CallToons will use Cartoon Network characters in phones aimed at children, while characters from the network's \"Adult Swim\" programs will be in phones aimed at adults, the company said.  In the kid's version, characters can also be programmed to provide guidance, Cox said.  \"If a stranger were to call and the number is not in the phone book, the ringtone could say, \"Hey, a stranger is calling. Maybe you should hand this to your parent or let the call go into voice mail,\" he said.  Turner will also license its technology to other companies in the future, the company said.  Pricing details and deals with carriers were not released Wednesday. "

print(processMentionText(text_test, "Cartoon Network"))
from itertools import chain
from lxml.etree import tostring
#pdb.set_trace()

def loadMentions(dataFolder=defaultDataFolder):
    mentionToEntity = {}
    mentionToType = {}
    with open(kbLinksFile) as fileLinks:
        linksLines = fileLinks.readlines()
        kbLinks = [lL.strip().split('\t') for lL in linksLines]

    for kbLink in kbLinks:
        mentionToEntity[kbLink[0]] = kbLink[1]
        mentionToType[kbLink[0]] = kbLink[2]
    ####################### Queries #########################
    start = time.time()
    queriesName = {}
    mentionToText = {}
    lxmlFile = etree.parse(queriesFile)
    root = lxmlFile.getroot()
    queries = root.xpath("query")
    values = []
    for qu in queries:
        queriesName[qu.get("id")] = qu.find("name").text.replace("\n", " ")
        #mentionToText[qu.get("id")] = qu.find("docid").text.lower()
        mentionToText[qu.get("id")] = qu.find("docid").text
    end = time.time()
    print("Queries built, took " + str(end-start))

    mentionsJsonFile = open("mentionsJsonFileComplete-" + year + "-" + sequence +  "-FullText.json", "w")

    ####################### Mentions ########################
    cuTime = 0
    start = time.time()
    i = 0
    mention_index = 0
    for m in mentionToText.keys():
        print(sourceFile + mentionToText[m] + ".xml")
        #pdb.set_trace()
        #with open(sourceFile + mentionToText[m].upper() + ".xml") as mentionFile:
        with open(sourceFile + mentionToText[m] + ".xml") as mentionFile:
            lxmlMentionsHeadline = etree.iterparse(sourceFile + mentionToText[m] + ".xml", events=('end',), tag='HEADLINE')
            mentionTextHeadline = ""

            try:
                lxmlMentionsHeadlineElements = [obj for obj in lxmlMentionsHeadline]
                if ( len(lxmlMentionsHeadlineElements) > 0 ): 
                    if len(lxmlMentionsHeadlineElements[0]) > 1:
                        mentionTextHeadline = lxmlMentionsHeadlineElements[0][1].text.replace("\n", " ")
            
            except:
                pass
            
            P_code_names = ["AFP", "CNA", "APW", "LTW", "NYT", "WPB", "XIN"]
            if any(code_name in mentionToText[m] for code_name in P_code_names):
                lxmlMentionsP = etree.iterparse(sourceFile + mentionToText[m] + ".xml", events=('end',), tag='P')
                mentionTexts = [lm[1] for lm in lxmlMentionsP]

            Post_code_names = ["eng-"] 
            if any(code_name in mentionToText[m] for code_name in Post_code_names):
                #if mentionToText[m] in "eng-NG-31-108089-8957009.xml":
                #    pdb.set_trace()
                #lxmlMentionsP = etree.iterparse(sourceFile + mentionToText[m].upper() + ".xml", events=('end',), tag='POST')
                lxmlMentionsP = etree.iterparse(sourceFile + mentionToText[m] + ".xml", events=('end',), tag='POST', recover=True)
                node = [lm[1] for lm in lxmlMentionsP][0]
                parts = ([node.text] + list(chain(*([c.text, c.tail] for c in node.getchildren()))) + [node.tail])
                #mentionTexts = [lm[1] for lm in lxmlMentionsP]
                parts = ''.join(filter(None, parts))
                mentionTexts = parts.replace("\n", " ")        
                        
        
            post_code_names = ["bolt-e"]
            if any(code_name in mentionToText[m] for code_name in post_code_names):
                lxmlMentionsP = etree.iterparse(sourceFile + mentionToText[m] + ".xml", events=('end',), tag='post')
                mentionTexts = [lm[1] for lm in lxmlMentionsP]


        
            mentionText = mentionTextHeadline

            if any(code_name in mentionToText[m] for code_name in Post_code_names):
                mentionText = mentionText +  mentionTexts
            else:
                for mt in mentionTexts:
                    mentionText = mentionText + mt.text.replace("\n", " ")

            mentionName = queriesName[m]
            mentionEntity = mentionToEntity[m]
            Mention = {}
            Mention["mention_id"] = m
            Mention["mention_name"] = mentionName
            Mention["mention_index"] = mention_index
            Mention["mention_full_text"] = mentionText
            Mention["mention_text"] = processMentionText(mentionText, mentionName)
            Mention["gold_entity_id"] = mentionEntity
            #if "NIL" in mentionEntity:
            #   Mention["gold_entity_type"] = mentionEntity
            #else:
            #   Mention["gold_entity_type"] = entityIdToType[mentionEntity]
            Mention["gold_entity_type"] = mentionToType[Mention["mention_id"]]
            mentionsJsonFile.write(json.dumps(Mention) + "\n")
            mention_index += 1
            #if Mention["mention_id"] == "EL1142":
            #   print(Mention["mention_text"])
            #   pdb.set_trace()
        i += 1
    print("Mentions base built, took : " + str(time.time()-start))
        

    return 1 
    ####################################################################################




loadMentions()
pdb.set_trace()
