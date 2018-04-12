# coding: utf8
import codecs
import argparse
import re
import pdb
import os
import json
import pandas
from lxml import etree
import csv
import time
import numpy as np
import sys
import random
import pickle
#Example 
#doc = etree.parse('content-sample.xml')




parser = argparse.ArgumentParser(description="Machine")
parser.add_argument('-m', help='foo help')
args = parser.parse_args()
defaultDataFolder = "/home/khalife/ai-lab/data/"
knowledgeBaseFile = defaultDataFolder + "tac_kbp_ref_know_base/data/"
writeFile = open(defaultDataFolder +  "tac_kbp_ref_know_base/analysis/entityIdToTypeFile.json", "w")


############################# Knowledge Base ######################################	
def loadKnowledgeBaseTypes():
	for subdir, dirs, files in os.walk(knowledgeBaseFile):
		kbFiles = files
	kbList = [knowledgeBaseFile + kb for kb in kbFiles if "sw" not in kb]
	entityIdToType = {}
	entityIdToName = {}
	entityIdToText = {}
	start = time.time()
	for kb in kbList:
		print(kb)
		with open(kb) as kbFile:
			lxmlKb = etree.iterparse(kb, events=('end',), tag='entity')#, encoding="utf-8")
			lxmlKbs = [lk for lk in lxmlKb]
			for kbes in lxmlKbs:
				entityId = kbes[1].get("id")
				entityType = kbes[1].get("type")
				entityIdToType[entityId] = entityType
				entityName = kbes[1].get("name")
				entityIdToName[entityId] = entityName
				entityText = kbes[1].find("wiki_text").text.replace("\n", " ")
				entityIdToText[entityId] = entityText
	#print(len([key for key in entityIdToType.keys()]))
	#with open(defaultDataFolder +  "tac_kbp_ref_know_base/analysis/entityIdToType" + ".json", "w") as entityToTypeJson:
	#	json.dump(entityIdToType, entityToTypeJson)                                                 	
	#with open(defaultDataFolder +  "tac_kbp_ref_know_base/analysis/entityIdToText" + ".json", "wb") as entityToTextJson:
	#	json.dump(entityIdToText, codecs.getwriter('utf-8')(entityToTextJson), ensure_ascii=False)
	
	
				#writeTextFile.write(entityId + " _E_ " + entityText + "\n")
	json.dump(entityIdToType, writeFile)

	print("Knowledge base built, took : " + str(time.time()-start))

	return 1 
####################################################################################

loadKnowledgeBaseTypes()
writeFile.close()














































































































































































































































































































































