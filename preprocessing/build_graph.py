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
defaultDataFolder = "/home/khalife/ai-lab/data/"
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
#def loadKnowledgeBase(kbList):	
	#entityIdToType = {}
	#entityIdToName = {}
	#entityIdToText = {}
	#entityIdToNode = {}
	#entityIdToIndex = {}
	#entityOutlinks = []
	#entityNodes = []
	#start = time.time()
	#counter_type = 0
	#categories = ["PER", "ORG", "GPE"]

	raw = []
	col = []
	#print("First go")
	#nodeIndex = 0
	#nodes_seen = []
	#for kb in kbList:
	#	print(counter_type)
	#	with open(kb) as kbFile:
	#		lxmlKb = etree.iterparse(kb, events=('end',), tag='entity')#, encoding="utf-8")
	#		lxmlKbs = [lk for lk in lxmlKb]
	#		for kbes in lxmlKbs:
	#			entityType = kbes[1].get("type")
	#			entityId = kbes[1].get("id")
	#			entityName = kbes[1].get("name")
	#			entityIdToType[entityId] = entityType
	#			entityIdToName[entityId] = entityName
	#			entityIdToIndex[entityId] = nodeIndex
	#			nodes_seen.append(nodeIndex)
	#			#G.add_node(nodeId, {"entity_id" : entityId, "entity_name" : entityName})
	#			#G.add_node(entityId, {"entity_name" : entityName})
	#			#entityIdToNode[entityId] = nodeId	
	#			nodeIndex += 1		
	#	counter_type += 1
	#
	#with open(entityIdToIndexFile, "w") as f:
	#	json.dump(entityIdToIndex, f)
	#return 1
	
	print("Second go")
	#edges_limit = 100
	node_index = 0
	
	edges_seen = []
	counter_edge = 0
	for kb in kbList_:
		print(kb)
		with open(kb) as kbFile:
			lxmlKb = etree.iterparse(kb, events=('end',), tag='entity')
			lxmlKbs = [lk for lk in lxmlKb]
			#pdb.set_trace()
			for kbes in lxmlKbs:
				#pdb.set_trace()
				entityId = kbes[1].get("id")
				entityType = kbes[1].get("type") 
				if entityType in categories:
					#entityName = kbes[1].get("name")
					#entityIdToName[entityId] = entityName
					#entityText = kbes[1].find("wiki_text").text.replace("\n", " ")
					#entityIdToText[entityId] = entityText
					# get entity outgoing links
					#local_iterator = kbes[1].getiterator()
					#entityNodes.append({"entity_id" : entityId, "entity_name" : entityName, "entity_text" : entityText})
					entityOutLinks = []
					for li in kbes[1].xpath(".//facts/fact/link"):
					#for li in local_iterator:
						if "entity_id" in li.attrib.keys():
							link_id = li.attrib["entity_id"]
							if entityIdToType_[link_id] in categories:
								raw.append(entityIdToIndex_[entityId])
								#targetEntityName = entityIdToName[link_id]
								col.append(entityIdToIndex_[link_id])
								#edges_seen.append((entityId, link_id))
	
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










