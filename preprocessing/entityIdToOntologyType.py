from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pdb
from hashlib import sha1
from datasketch.minhash import MinHash
import json

#test_dic = json.load(open("entityIdToOntologyType.json", "r"))
#pdb.set_trace()
 
entityIdToWikiTitle = json.load(open("entityIdToWikiTitle.json", "r"))
print("loaded 1")
entityWikiTitleToType = json.load(open("wiki_title_toType.json", "r"))
original_types = json.load(open("entityIdToType-Updated-2010.json","r"))

choices = [ew for ew in entityWikiTitleToType.keys()]
process.extract("new york jets", choices, limit=2)
pdb.set_trace()


with open("gold_ids_2010.txt", "r") as f:
    gold_ids = f.readline().strip().split(", ")



#recent_remaining_titles = [key for key in entityWikiTitleToType.keys()]
#recent_remaining_titles = entityWikiTitleToType
print("loaded")

with open("PER.txt", "r") as f:
    PER_subtypes = f.readline().strip().split(", ")

with open("ORG.txt", "r") as f:
    ORG_subtypes = f.readline().strip().split(", ")

with open("GPE.txt", "r") as f:
    GPE_subtypes = f.readline().strip().split(", ")



error_ids = []
error_titles = []
used_titles = []
entityIdToOntologyType = {}
for key_id in entityIdToWikiTitle.keys():
    wiki_title = entityIdToWikiTitle[key_id]
    #if key_id == "E0344712":
    #    pdb.set_trace()
    try:
        wiki_type = entityWikiTitleToType[wiki_title]
        if original_types[key_id] == "PER":
            possible_subtypes = PER_subtypes        

        if original_types[key_id] == "ORG":
            possible_subtypes = ORG_subtypes

        if original_types[key_id] == "GPE":
            possible_subtypes = GPE_subtypes
    
        if wiki_type in possible_subtypes:
            entityIdToOntologyType[key_id] = wiki_type
         
        else:
            raise Exception("Not found suitable type") 

        #del recent_remaining_titles[]
        #entityWikiTitleToType.pop(wiki_title)       
    except Exception:
        entityIdToOntologyType[key_id] = original_types[key_id]
        error_ids.append(key_id)
        error_titles.append(wiki_title)


mis_gold_ids = [gi for gi in gold_ids if gi in error_ids]
mis_gold_names = [entityIdToWikiTitle[gi] for gi in mis_gold_ids]
mis_gold_types = [original_types[gi] for gi in mis_gold_ids]
mis_gold_indexes = [[i for i, x in enumerate(gold_ids) if x == gi] for gi in [x for x in set(mis_gold_ids)]]


ranksFile = open("test-predictionRanks-NEW-TYPES-1-OWT-Scores-Complete-2010-UPDATE-NNIL.json", "r")
prediction_ranks_entities = []
for line in ranksFile:
    dic_line = json.loads(line)
    prediction_ranks_entities.append(dic_line["gold_dic"]["mention_gold_entity_id"])


print(set(mis_gold_ids).intersection(set(prediction_ranks_entities)))
pdb.set_trace()
#PER_subtypes_title = [key for key in entityWikiTitleToType.keys() if entityWikiTitleToType[key] in PER_subtypes]
#ORG_subtypes_title = [key for key in entityWikiTitleToType.keys() if entityWikiTitleToType[key] in ORG_subtypes]
#GPE_subtypes_title = [key for key in entityWikiTitleToType.keys() if entityWikiTitleToType[key] in GPE_subtypes]
#
#import difflib
#for mis_gold_name, mis_gold_id in zip(mis_gold_names, mis_gold_ids):
#    if original_types[mis_gold_id] == "PER":
#        subtypes_title = PER_subtypes_title
#   
#    if original_types[mis_gold_id] == "ORG":
#        subtypes_title = ORG_subtypes_title
#
#    if original_types[mis_gold_id] == "GPE":
#        subtypes_title = GPE_subtypes_title
#
#    print("original_name : " + mis_gold_name)
#    result_diff = difflib.get_close_matches(mis_gold_name, subtypes_title)
#    print("results_diff : [%s]"   % ", ".join(map(str, result_diff)))
#pdb.set_trace()


#json.dump(entityIdToOntologyType, open("entityIdToOntologyType.json", "w"))















