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
import networkx as nx
import matplotlib.pyplot as plt
import sys
#Example 
#doc = etree.parse('content-sample.xml')

#parser = argparse.ArgumentParser(description="Machine")
#parser.add_argument('-m', help='foo help')
#args = parser.parse_args()
defaultDataFolder = sys.argv[1]
knowledgeBaseFile = defaultDataFolder + "tac_kbp_ref_know_base/data/"
entityIdToIndexFile = defaultDataFolder + "tac_kbp_ref_know_base/analysis/entityIdToIndexFile.json"
entityIdToTypeFile = defaultDataFolder + "tac_kbp_ref_know_base/analysis/entityIdToTypeFile.json"

categories = ["PER", "ORG", "GPE"]
############################# Knowledge Base ######################################	
def loadEntityIdToIndex():
	with open(entityIdToIndexFile, "r") as f:
		entityIdToIndex = json.load(f)

	with open(entityIdToTypeFile, "r") as f:
		entityIdToType = json.load(f)

	return entityIdToIndex, entityIdToType

def loadKnowledgeBase(entityIdToIndex_, entityIdToType_, kbList_):

	raw = []
	col = []
	print("Second go")
	node_index = 0
	
	edges_seen = []
	counter_edge = 0
	for kb in kbList_:
		print(kb)
		with open(kb) as kbFile:
			lxmlKb = etree.iterparse(kb, events=('end',), tag='entity')
			lxmlKbs = [lk for lk in lxmlKb]
			for kbes in lxmlKbs:
				entityId = kbes[1].get("id")
				entityType = kbes[1].get("type") 
				if entityType in categories:
					entityOutLinks = []
					for li in kbes[1].xpath(".//facts/fact/link"):
						if "entity_id" in li.attrib.keys():
							link_id = li.attrib["entity_id"]
							if entityIdToType_[link_id] in categories:
								raw.append(entityIdToIndex_[entityId])
								col.append(entityIdToIndex_[link_id])
	
	#print("Knowledge base graph built, took : " + str(time.time()-start))
	return raw, col 								

####################################################################################
indexes = range(88)
index = int(sys.argv[1])
extent = [11*index, 11*(index+1)]
for subdir, dirs, files in os.walk(knowledgeBaseFile):
	kbFiles = files

kbList = [knowledgeBaseFile + kb for kb in kbFiles if "sw" not in kb][extent[0]:extent[1]]
entityIdToIndex, entityIdToType = loadEntityIdToIndex()
raw, col = loadKnowledgeBase(entityIdToIndex, entityIdToType, kbList)
assert(len(raw) == len(col))
with open(defaultDataFolder + "tac_kbp_ref_know_base/analysis/EDGES/edgesIndexFile-" + str(index) + ".txt", "w") as writeEdgesFile:
	for i in range(len(raw)):
		writeEdgesFile.write(str(raw[i]) + " " + str(col[i]) + "\n")	

writeEdgesFile.close()	










