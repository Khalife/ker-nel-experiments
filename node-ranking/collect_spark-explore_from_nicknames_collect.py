import pdb
import sys
import json
import networkx as nx
import time
from collections import defaultdict, deque
import operator
import numpy as np
import scipy.sparse as sp
from string_metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from igraph import *

############################## Spark #########################################
#from pyspark import SparkContext, SQLContext
#from pyspark.conf import SparkConf
#import pyspark
#sc = SparkContext(appName="nel-system")
#sess = pyspark.sql.SparkSession.builder.appName("nel-system").getOrCreate()
##############################################################################


def generateData(line):
    #for line in ranks_file:
    #    dic_line = json.loads(line.replace('u', '').replace('\'','\"'))
    #    gold_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    #    mention_id = dic_line["mention_id"]
    #    return {"sorted_RANKS": dic_line["prediction_ranks"], "gold_entity_id": gold_id, "mention_id" : mention_id }
    dic_line = json.loads(line.replace('u', '').replace('\'','\"'))
    gold_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    mention_id = dic_line["mention_id"]
    return {"sorted_RANKS": dic_line["prediction_ranks"], "gold_entity_id": gold_id, "mention_id" : mention_id }

def generateDataCpu(ranks_filename):
    ranks_file = open(ranks_filename, "r")
    for line in ranks_file:
        dic_line = json.loads(line.replace('u', '').replace('\'','\"'))
        gold_id = dic_line["gold_dic"]["mention_gold_entity_id"]
        mention_id = dic_line["mention_id"]
        yield {"sorted_RANKS": dic_line["prediction_ranks"], "gold_entity_id": gold_id, "mention_id" : mention_id }


def listNeighbors(Graph, L):
    neighbors = [] 
    for node in L:
        node_neighbors = Graph.neighbors(node)
        for node_neighbor in node_neighbors:
            if node_neighbor not in neighbors:
                neighbors.append(node_neighbor)
    return neighbors    



def igraph_neighbors_iter(self, node):
    return iter(self.adj[node])
    

def bfs_edges_level(G, max_depth, source, reverse=False):
    if reverse and isinstance(G, nx.DiGraph):
        neighbors = G.predecessors_iter
    else:
        neighbors = G.neighbors_iter
    visited = set([source])
    
    if max_depth > 0:
        queue = deque([(source, 0, neighbors(source))])
    else:
        queue = deque([])
    while queue:
        parent, depth, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
            if depth+1 < max_depth:
                queue.append((child, depth+1, neighbors(child)))
        except StopIteration:
            queue.popleft()

#G.neighbors_iter
#return(self.adj[n])



def bfs_edges_level_standard(G, max_depth, source):
    #neighbors = G.neighbors_iter
    visited = set([source])
    
    if max_depth > 0:
        queue = deque([(source, 0, iter(G[source]))])
    else:
        queue = deque([])
    while queue:
        parent, depth, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
            if depth+1 < max_depth:
                queue.append((child, depth+1, iter(G[child])))
        except StopIteration:
            queue.popleft()



def bfs_edges_level_myGraph(G, max_depth, source):
    #neighbors = G.neighbors_iter
    visited = set([source])
    
    if max_depth > 0:
        queue = deque([(source, 0, iter(G.graph[source]))])
    else:
        queue = deque([])
    while queue:
        parent, depth, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
            if depth+1 < max_depth:
                queue.append((child, depth+1, iter(G.graph[child])))
        except StopIteration:
            queue.popleft()



print("Building graph")
edgesIdFile = open("../entityIdToIndex-GRAPH-File.json","r")
for line in edgesIdFile:
    entityIdToIndex = json.loads(line)

# For igraph
for key in entityIdToIndex.keys():
    entityIdToIndex[key] = entityIdToIndex[key]

entityIndexToId = {v: k for k, v in entityIdToIndex.items()}

entityIdToTypeFile = open("../../entityIdToType-Updated.json", "r")
for line in entityIdToTypeFile:
    entityIdToType = json.loads(line)

entityIndexToType = {}
for key in entityIdToType.keys():
    entityIndexToType[entityIdToIndex[key]] = entityIdToType[key]



edgesFile = open("../../edgesIndexFile.txt", "r")
edgesList = []
for line in edgesFile:
    id1, id2 = line.split()
    #edgesList.append([int(id1), int(id2)])
    edgesList.append((int(id1), int(id2)))

#nodesList = []
#for node in entityIdToIndex.values():
#for node in range(nb_nodes)
#    nodesList.append(node)

nb_nodes = len([val for val in entityIdToIndex.values()])
#G = Graph()

class myGraph(dict):
    def __init__(self):
        self.graph = {}
        
    def add_nodes_from(self, nodes):
        for node in nodes:
            self.graph[node] = set([])
    
    def add_edges_from(self, edges):
        for edge in edges:
            node1, node2 = edge
            self.graph[node1].add(node2)
            self.graph[node2].add(node1)


    def neighbors(self, node):
        return list(self.graph[node])

    def degree(self, node):
        return len(self.graph[node])


    
    def subgraph(self, nodes):
        new_edges = []
        new_subgraph = myGraph() 
        for node in nodes:
            new_subgraph.graph[node] = set([])
            neighbor_nodes = self.graph[node]
            for neighbor_node in neighbor_nodes:
                new_subgraph.graph[node].add(neighbor_node)
                try:
                    new_subgraph.graph[neighbor_node].add(node)
                except:
                    new_subgraph.graph[neighbor_node] = set([])
                    new_subgraph.graph[neighbor_node].add(node)
        return new_subgraph

    #def shortest_path_length(self, source, target):
    #    return dijkstra(self.graph, source, target)






#graph = {}
#for node in nodesList:
#    graph[node] = set([])
#
#for edge in edgesList:
#    node1, node2 = edge
#    graph[node1].add(node2)
#    graph[node2].add(node1)    
#
#graph_degrees = {}
#for node in graph.keys():
#    graph_degrees[node] = len(graph[node])
#        
#
##G.add_nodes_from(nodesList)
##G.add_edges_from(edgesList)
#
#G = graph
#G = nx.Graph()
#G = myGraph()
#G.add_nodes_from(nodesList)
#G.add_edges_from(edgesList)

G = Graph()
G.add_vertices(nb_nodes)
G.add_edges(edgesList)

entityIdToOntologyType = json.load(open("../entityIdToOntologyType-Updated-27-02-18.json", "r"))
# Government agency ['E0476856', 'E0793281', 'E0071026']
# Town ['E0664098', 'E0459068', 'E0189873', 'E0133014', 'E0437638', 'E0437638'] 
# BaseBallPlayer 'E0219212' 
# Country {'E0315956', 'E0679687', 'E0363137'}
# TradeUnion {'E0468402'}

list_ids = ['E0468402']
frequency_type = {}
neighbors_type = sum([[entityIdToOntologyType[entityIndexToId[es]] for es in G.neighbors(entityIdToIndex[ei])] for ei in list_ids],[])
for nt in neighbors_type:
    frequency_type[nt] = 0

for nt in neighbors_type:
    frequency_type[nt] += 1



#pdb.set_trace()
#data2010Folder = "/home/khalife/ai-lab/data/LDC2015E19_TAC_KBP_English_Entity_Linking_Comprehensive_Training_and_Evaluation_Data_2009-2013/json/backup/ORG/NEW_TESTS/MENTIONS_UPDATE/2010/"
try:
    data2010Folder = sys.argv[1]
except Exception as e:
    print(e)
    print("Please provide a default data folder")

entityUKNIndexToDegree = json.load(open(data2010Folder + "entityUKNIndexToDegree.json", "r"))
#kbFile = open(data2010Folder + "knowledgeBaseFile-Updated-2010.json", "r")
#entityIdToKBIndex = {}
#index = 0
#for line in kbFile:
#    dic_line = json.loads(line)
#    entity_id = dic_line["entity_id"]
#    try:
#        if dic_line["entity_type"] == "UKN":
#            if entityUKNIndexToDegree[str(entityIdToIndex[dic_line["entity_id"]])] < 10:
#                continue
#
#    except:
#        continue
#    entityIdToKBIndex[entity_id] = index
#    index += 1




entityIdToKBIndex = json.load(open("../entityIdToKBIndex.json", "r"))

# Memphis example
#id_list = ['E0607811', 'E0065958', 'E0530902', 'E0757454', 'E0271607', 'E0412261', 'E0698308', 'E0304910', 'E0787907', 'E0642032']
id_list = ['E0029958', 'E0377527', 'E0550618', 'E0639496', 'E0748134', 'E0689273', 'E0367940', 'E0370036', 'E0762371', 'E0003347']
# Democratic party example




#entityIdToOntologyType = json.load(open("entityIdToOntologyType.json", "r"))

with open("../PER.txt", "r") as f:
    PER_subtypes = f.readline().strip().split(", ")

with open("../ORG.txt", "r") as f:
    ORG_subtypes = f.readline().strip().split(", ")

with open("../GPE.txt", "r") as f:
    GPE_subtypes = f.readline().strip().split(", ")

#mention_mis = json.load(open("example1-mis-mention.json", "r"))

#entities_mis_file = open("example1-mis.json", "r")
#entityIdToFeatures = {}
#for line in entities_mis_file:
#    dic_line = json.loads(line)
#    entity_id = dic_line["entity_id"]
#    entity_name = dic_line["entity_name"]
#    #entity_text = dic_line["entity_text"]
#    #entity_index = dic_line["entity_index"]
#    entityIdToFeatures[entity_id] = {"entity_name": entity_name}

mentionsTfIdf = sp.load_npz("../mentions_tfidf-2010-UPDATE.npz")
M_KB_TFIDF = sp.load_npz("../matrices_tfidf-2010-UPDATE.npz")


def unicodeName(name):
    #name_u = unicode(name, 'utf-8')

    name_clean = unicodedata.normalize('NFD', name)
    name_clean = name_clean.encode('ascii', 'ignore')
    return name_clean



GPE_subtypes = ["AdministrativeRegion"]
neighbors_all_type = [[entityIdToOntologyType[entityIndexToId[node]] for node in G.neighbors(entityIdToIndex[id_l]) if entityIdToOntologyType[entityIndexToId[node]]] for id_l in id_list]
neighbors_all_id = [[entityIndexToId[node] for node in G.neighbors(entityIdToIndex[id_l]) if entityIdToOntologyType[entityIndexToId[node]]] for id_l in id_list]
neighbors_type = [[entityIdToOntologyType[entityIndexToId[node]] for node in G.neighbors(entityIdToIndex[id_l]) if entityIdToOntologyType[entityIndexToId[node]] in GPE_subtypes] for id_l in id_list]
neighbors_id = [[entityIndexToId[node] for node in G.neighbors(entityIdToIndex[id_l]) if entityIdToOntologyType[entityIndexToId[node]] in GPE_subtypes] for id_l in id_list]


#context_score = results_tfidf[index_kb][0,0]
entityIdToMainType = json.load(open(data2010Folder + "entityIdToType-Updated-2010.json", "r"))

mentionsFileTest = open(data2010Folder + "predictionRanks-NEW-TYPES-27-02-18-1-OWT-Scores-Complete-2010-UPDATE-NNIL.json", "r") 


mentionToRank = {}
cityTest = ["E0757454", "E0401466", "E0238320", "E0171454", "E0294319", "E0600130", "E0519583", "E0152710"]
#frequency_2_ids = ['E0769300', 'E0189873', 'E0629007', 'E0420830', 'E0216703', 'E0275249', 'E0363137', 'E0613782', 'E0399313', 'E0103032', 'E0227897', 'E0739132', 'E0408938', 'E0476856', 'E0435757', 'E0396272', 'E0664098', 'E0371427', 'E0642032', 'E0791416', 'E0437638']

#cityTest = ["E0593713"]
#cityTest = ["E0468402"] # Free trade union
cityTest = ["E0593713"] # US Senate
frequency_2_types = ['AdministrativeRegion', 'University', 'PER', 'PER', 'Town', 'MilitaryUnit', 'GPE', 'AdministrativeRegion', 'City', 'City', 'Settlement', 'AdministrativeRegion', 'City', 'GovernmentAgency', 'Settlement', 'City', 'University', 'ORG', 'City', 'ORG', 'University']


with open("../gold_ids_2010.txt", "r") as f:
    gold_ids = f.readline().strip().split(", ")


mentionsList = {}
for entity in gold_ids:
#for entity in frequency_2_ids:
    mentionsList[entity] = []

for line in mentionsFileTest:
    #pdb.set_trace()
    dic_line = json.loads(line)
    mention_gold_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    #if mention_gold_id in cityTest:
    #if mention_gold_id in frequency_2_ids:
    #    mentionsList[mention_gold_id].append(dic_line["mention_id"])
    #    mentionToRank[dic_line["mention_id"]] = dic_line["rank"]
    mentionsList[mention_gold_id].append(dic_line["mention_id"])
    mentionToRank[dic_line["mention_id"]] = dic_line["rank"]


all_mentions_id = sum([value for value in mentionsList.values()], [])

mentionsAllFile = open(data2010Folder + "mentionsJsonFileComplete-2010-UPDATE-NNIL.json" , "r")
mentionToFeature = {}
mentionToText = {}
for line in mentionsAllFile:
    dic_line = json.loads(line)
    line_id = dic_line["mention_id"]
    gold_id = dic_line["gold_entity_id"]
    #if line_id in all_mentions_id:
    #    mentionToFeature[line_id] = {"mention_id": line_id, "mention_name" : dic_line["mention_name"], "mention_index" : dic_line["mention_index"], "gold_entity_id": gold_id}
    mentionToFeature[line_id] = {"mention_id": line_id, "mention_name" : dic_line["mention_name"], "mention_index" : dic_line["mention_index"], "gold_entity_id": gold_id}
    mentionToText[line_id] = dic_line["mention_full_text"]

import re
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

rx = re.compile(r"[\W]")
def processFullText(str1):
       mention_context_text = re.sub(rx, " ", str1.lower())
       words = mention_context_text.split()
       bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
       return " ".join(bag_of_words)


mentionToTopEntities = {}
mentionsScores = open(data2010Folder + "mentions-NEW-TYPES-27-02-18-1-OWT-Scores-Complete-2010-UPDATE-NNIL.json", "r")
for line in mentionsScores:
    dic_line = json.loads(line.replace('u', '').replace('\'','\"'))
    #pdb.set_trace()
    #dic_line = json.loads(line)    
    #mention_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    mention_id = dic_line["mention_id"]
    #if mention_id in all_mentions_id:
    #    gold_entity_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    #    gold_entity_type = entityIdToOntologyType[gold_entity_id]
    #    top_10_entities = dic_line["prediction_ranks"][gold_entity_type.replace('u', '')][:10]
    #    mentionToTopEntities[mention_id] = top_10_entities    

    gold_entity_id = dic_line["gold_dic"]["mention_gold_entity_id"]
    gold_entity_type = entityIdToOntologyType[gold_entity_id]
    top_10_entities = dic_line["prediction_ranks"][gold_entity_type.replace('u', '')][:10]
    mentionToTopEntities[mention_id] = top_10_entities    


entityIdToFeatures = json.load(open("../entityIdToIndexName.json", "r"))



def score_name(mention_name, entity_name):
    _, acronym_test = acronymTest(mention_name, "ORG")
    if acronym_test:
        name_score = acronymScore1(mention_name, entity_name)
    else:
        closest_clean_entity_name = entity_name.lower()
        closest_clean_mention_name = mention_name.lower()
        score2 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 2)
        score3 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 3)
        score4 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 4)
        name_score = (score2 + score3 + score4)/3

    return name_score


gold_types = ['BusCompany', 'ChessPlayer', 'ORG', 'AdministrativeRegion', 'Settlement', 'TennisPlayer', 'Town', 'PER', 'Politician', 'PoliticalParty', 'Airline', 'Band', 'Writer', 'PublicTransitSystem', 'Lake', 'AmericanFootballPlayer', 'Village', 'Actor', 'Comedian', 'University', 'OfficeHolder', 'TradeUnion', 'President', 'GPE', 'Governor', 'Country', 'SoccerClub', 'MilitaryUnit', 'MilitaryPerson', 'Non-ProfitOrganisation', 'Senator', 'Legislature', 'TelevisionStation', 'City', 'GovernmentAgency', 'BroadcastNetwork', 'Company', 'MusicalArtist', 'Model', 'BaseballPlayer', 'RadioStation', 'Congressman']


mis_ontologies = ['GovernmentAgency', 'MusicalArtist', 'AdministrativeRegion', 'Scientist', 'Non-ProfitOrganisation', 'Newspaper', 'Newspaper', 'ORG', 'PoliticalParty', 'City', 'Judge', 'GovernmentAgency', 'Company', 'ORG', 'Actor', 'Actor', 'GeopoliticalOrganisation', 'Newspaper', 'Company', 'PlayboyPlaymate', 'Hospital', 'Country', 'School', 'AmericanFootballPlayer']

entityIdToName = json.load(open("../entityIdToIndexName.json", "r"))
typeToThreshold = json.load(open("../typeToThreshold.json", "r"))

import re
rx = re.compile(r"[\W]")
def reScore(mention_mis_, alpha): #, neighbors_id_):
    #print(mention_mis_)
    #id_list = ['E0029958', 'E0377527', 'E0550618', 'E0639496', 'E0748134', 'E0689273', 'E0367940', 'E0370036', 'E0762371', 'E0003347']

    # 1- Load top 10 ranked entities
        # for each of them load neighbors
    # 2- Compute agregated score for each of the neigbhor

    
    mention_name = mention_mis_["mention_name"]
    top10ranked_entities = mentionToTopEntities[mention_mis_["mention_id"]]
    gold_id =  mentionToFeature[mention_mis_["mention_id"]]["gold_entity_id"]
    gold_type = entityIdToOntologyType[gold_id]

    top10ranked_entities_id = [tre["entity_id"] for tre in top10ranked_entities]
    if gold_id in top10ranked_entities_id:
        original_rank = top10ranked_entities_id.index(gold_id)
    else:
        original_rank = 100
    
    
    #######################################################################################
    #if gold_type in ['TradeUnion', 'City', 'Legislature', 'PoliticalParty', 'University', 'Country']:
    #if gold_type in ['City', 'Settlement', 'Company', 'AdministrativeRegion', 'University', 'Town', 'GovernmentAgency', 'BaseballPlayer', 'Country', 'TradeUnion', 'Legislature']:
    ##if gold_type in ["City"]:
    ##if 0:
    #    # Homonymy detection
    #    #if len(top10ranked_entities) < 3:
    #    #    threshold_type = 1
    #    #    std_name_score = 0
    #    threshold_type = 1
    #    std_name_score = 0
    #    if 0:
    #    #else:
    #        name_scores = []
    #        for i in range(min([12, len(top10ranked_entities)])):
    #            name_i = entityIdToName[top10ranked_entities[i]["entity_id"]]["entity_name"]
    #            _, acronym_test = acronymTest(mention_name, entityIdToMainType[gold_id])
    #            _, acronym_test_entity = acronymTest(name_i, entityIdToMainType[gold_id])

    #            #closest_clean_entity_name = name_i.split("_(")[0]
    #            closest_clean_entity_name =  unicodeName(name_i).replace("_", " ")
    #            closest_clean_entity_name = re.sub(rx, " ", closest_clean_entity_name)
    #            closest_clean_mention_name = unicodeName(mention_name).replace("_", " ")
    #            closest_clean_mention_name = re.sub(rx, " ", closest_clean_mention_name)

    #            if acronym_test or acronym_test_entity:
    #                #name_score = acronymScore1(mention_name, i_dic["entity_name"])
    #                if acronym_test:
    #                    if acronym_test_entity:
    #                        norm_name_factor = min([len(closest_clean_entity_name), len(closest_clean_mention_name)])
    #                        name_score = longest_common_substring(closest_clean_entity_name, closest_clean_mention_name)/norm_name_factor
    #                    else:
    #                        name_score = acronymScore1(closest_clean_mention_name, closest_clean_entity_name)

    #                else:
    #                    name_score = acronymScore1(closest_clean_entity_name, closest_clean_mention_name)
    #                name_scores.append(name_score)

    #            else:
    #                closest_clean_entity_name = closest_clean_entity_name.lower()
    #                closest_clean_mention_name = closest_clean_mention_name.lower()
    #                #score2 = scoreLetterNgram(mention_name.lower(), i_dic["entity_name"].lower(), 2)            
    #                #score3 = scoreLetterNgram(mention_name.lower(), i_dic["entity_name"].lower(), 3)
    #                #score4 = scoreLetterNgram(mention_name.lower(), i_dic["entity_name"].lower(), 4)
    #                #score2 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 2)
    #                #score3 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 3)
    #                #score4 = scoreLetterNgram(closest_clean_mention_name, closest_clean_entity_name, 4)
    #                #name_score = (score2 + score3 + score4)/3
    #                norm_name_factor = min([len(closest_clean_entity_name), len(closest_clean_mention_name)])
    #                name_score = longest_common_substring(closest_clean_mention_name, closest_clean_entity_name)/norm_name_factor
    #                #name_score = longest_common_substring(mention_name.lower(), i_dic["entity_name"].lower())/float(min([len(mention_name),len(i_dic["entity_name"])]))
    #                name_scores.append(name_score)

    #        threshold_type = typeToThreshold[gold_type]
    #        std_name_score = np.std(name_scores)
    #else:
    #    threshold_type = 0 
    #    std_name_score = 1
    #########################################################################################
    
    #if gold_type in ['TradeUnion', 'City', 'Legislature']:
    
    #    pdb.set_trace() 
   
    nb_max_features = 50000 
    Vectorizer = TfidfVectorizer(max_features = nb_max_features)
     
    #if std_name_score > threshold_type:
    #    return [[0, top10ranked_entities[k]["score"], top10ranked_entities[k]["score"]] for k in range(len(top10ranked_entities))] 
    nb_neighbors_per_candidate = []
    #else:
    if 1:
        neighbors_id_ = []
        top_scores = []
        neighbor_frequency = {}
        #for top_entity in top10ranked_entities[:12]:
        #    neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
        #    for nte in neighbors_top_entity:
        #        neighbor_frequency[nte] = 0


        #for top_entity in top10ranked_entities[:12]:
        #    neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
        #    for nte in neighbors_top_entity:
        #        neighbor_frequency[nte] += 1

        ##pdb.set_trace()

        for top_entity in top10ranked_entities[:7]:
        #for top_entity in top10ranked_entities[:12]:
        #    neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
        #    #pdb.set_trace()
        #    
        #    if gold_type == "City":
        #        filtered_neighbors_ = [ne for ne in neighbors_top_entity if entityIndexToId[ne] in real_country_ids and neighbor_frequency[ne] <= 3]
        #    else:
        #        filtered_neighbors_ = [ne for ne in neighbors_top_entity if entityIdToOntologyType[entityIndexToId[ne]] in GPE_subtypes1[gold_type] and neighbor_frequency[ne] <= 3]
        #    
        #    #filtered_neighbors_ = [ne for ne in neighbors_top_entity if entityIdToOntologyType[entityIndexToId[ne]] in GPE_subtypes1[gold_type] and G.degree(ne) <= 100]
        #    filtered_neighbors = []
        #    for fn in filtered_neighbors_:
        #        try:
        #            test = entityIdToName[entityIndexToId[fn]]
        #            filtered_neighbors.append(fn)
        #        except:
        #            assert(1)
        #            #print("One neighbor removed")
 
        #    #print([[entityIndexToId[fn], entityIdToName[entityIndexToId[fn]]] for fn in filtered_neighbors])
        #    neighbors_id_.append(filtered_neighbors)
            top_scores.append(top_entity["score"])        

        
        

        ########## Country information aggregation ##########
        counter_debug = 0 
        #try: 
        if 1:         
   
            #pdb.set_trace()
            mention_vector = mentionsTfIdf[mention_mis_["mention_index"]]
            
            countries_and_co_index_to_keep = [entityIdToFeatures[cid]["entity_index"] for cid in real_ids] 
            countries_and_co_matrix = M_KB_TFIDF[countries_and_co_index_to_keep] 

            M0 = countries_and_co_matrix.dot(mention_vector.transpose())
            #for i in range(min([len(new_scores_total), len(top_scores)])):
            no_neighbor_candidate = []
            argmins_s_tau = []
            neighbors_id = []
            neighbors_type = []
            new_temp_scores_total_main = []
            #for percentile_neighbors in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            
            if 1:
            #for nb_max_neighbors in range(1, 20):
                scores_total = []        
                new_scores_total = []
                neighbor_k_text = []     

                for i in range(7):
                #for top_entity in top10ranked_entities[:12]:
                    #if percentile_neighbors == 10:
                    #if nb_max_neighbors == 1: 
                    if 1:
                        top_entity = top10ranked_entities[i]
                        entities_index_to_keep = []
                        neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
                        filtered_neighbors_ = [entityIndexToId[ne] for ne in neighbors_top_entity if entityIndexToId[ne] in real_ids]
                        ############## Get closest country #################
                        test_set = set(filtered_neighbors_).intersection([rci for rci in real_countries_id])
                        test = len(test_set)
                        if test < 1:
                            shortest_path_countries = []
                            counter_country = 0
                            for real_country_id in real_countries_id:
                                #if entityIdToIndex[real_country_id] in nodes1:
                                #    countries.append(entityIdToFeatures[real_country_id]["entity_name"])
                                #counter_country += 1
                                #if counter_country % 50 == 0:
                                #    print(real_country_id)
                                #pdb.set_trace()
                                shortest_path_length_value = G.shortest_paths_dijkstra(entityIdToIndex[top_entity["entity_id"]], entityIdToIndex[real_country_id])[0][0]
                                shortest_path_countries.append(shortest_path_length_value)
                            shortest_path_countries_min = min(shortest_path_countries)
                            if shortest_path_countries_min < 3:
                                countries_candidate = [real_countries_id[j] for j in range(len(real_countries_id)) if shortest_path_countries[j] == shortest_path_countries_min]
                                if len(countries_candidate) > 0:
                                    if len(countries_candidate) > 1:
                                        countries_and_co_index_to_keep_local = [entityIdToFeatures[cid]["entity_index"] for cid in countries_candidate]
                                        entity_vector = M_KB_TFIDF[entityIdToFeatures[top_entity["entity_id"]]["entity_index"]]
                                        countries_and_co_matrix_local = M_KB_TFIDF[countries_and_co_index_to_keep_local]
                                        dot_result = countries_and_co_matrix_local.dot(entity_vector.transpose())
                                        index_final_candidate = np.argmax([x[0,0] for x in dot_result])
                                        final_country_candidate = countries_candidate[index_final_candidate]
                                    
                                    else: 
                                        final_country_candidate = countries_candidate[0]
    
                                    filtered_neighbors_.append(final_country_candidate)
                            
                        neighbors_type_i = [entityIdToOntologyType[fn] for fn in filtered_neighbors_]                           
                        neighbors_type.append(neighbors_type_i)
                        neighbors_id.append(filtered_neighbors_)
                                       

                        ###################################################
                        #assert(len(filtered_neighbors_) > 0)
                        nb_neighbors_per_candidate.append(len(filtered_neighbors_))
                        ############## Filter by popularity ##############
                        if len(filtered_neighbors_) > 1:
                            neighbors_top_entity_degrees = [G.degree(entityIdToIndex[nte]) for nte in filtered_neighbors_]
                            #neighbors_top_entity_indexes = np.argsort(neighbors_top_entity_degrees)[::-1].tolist()
                            #filtered_neighbors_  = [filtered_neighbors_[ntei] for ntei in neighbors_top_entity_indexes[:min([nb_max_neighbors, len(neighbors_top_entity_indexes)])]]
                        ##################################################                

                        for ne in filtered_neighbors_:
                            entities_index_to_keep.append(entityIdToFeatures[ne]["entity_index"])

                        
                        new_temp_scores_total = []
                        if len(entities_index_to_keep) > 0:
                            entities_matrix = M_KB_TFIDF[entities_index_to_keep]
                            #if mention_name == "Richmond":
                            #    counter_debug += 1
                            #    if counter_debug >= 2:
                            #        pdb.set_trace()
                            #try:
                            EC = entities_matrix.dot(countries_and_co_matrix.transpose())
                            mention_country_and_co_entity_comparison = EC.dot(M0)
                            #except:
                            #    pdb.set_trace()    
                        
                            ############################################################

                            for i_n in range(EC.shape[0]):
                                new_temp_score_total = EC[i_n].data.dot(EC[i_n].data)
                                #cross_product = 2.*[x[0,0] for x in mention_country_and_co_entity_comparison][i_n] 
                                cross_product = [x[0,0] for x in mention_country_and_co_entity_comparison][i_n]
                                m0_square = M0.data.dot(M0.data)
                                #new_temp_scores_total.append(new_temp_score_total - cross_product + m0_square)
                                #new_temp_scores_total.append(-cross_product/np.sqrt(new_temp_score_total*m0_square))
                                cross_product = (1 - cross_product/np.sqrt(new_temp_score_total*m0_square))/2.
                                new_temp_scores_total.append(cross_product)                   

                        #pdb.set_trace()
                        #if EC.shape[0] == 0: # then no neighbors
                        else:
                            if i == 0:
                                no_neighbor_candidate.append(i)
                                new_temp_scores_total.append(0.5)
                            else:
                                #new_temp_scores_total.append(M0.data.dot(M0.data))
                                new_temp_scores_total.append(0.5)

                        new_temp_scores_total_main.append(new_temp_scores_total)

        else:
            assert(1)





        #####################################################
        
        #percentile_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        #diff_argmins_s_tau = [original_rank - ast for ast in argmins_s_tau]
        #min_value = np.min([np.abs(dast) for dast in diff_argmins_s_tau])
        ##optimal_neighbors_numbers = [i+1 for i, dast in zip(range(len(diff_argmins_s_tau)), diff_argmins_s_tau) if np.abs(dast) == min_value]
        #optimal_percentile_values = [percentile_values[i] for i, dast in zip(range(len(diff_argmins_s_tau)), diff_argmins_s_tau) if np.abs(dast) == min_value]       

        #return_dic = mention_mis_
        #return_dic["gold_rank"] =  original_rank
        #return_dic["final_rank_diff"] = min_value
        #return_dic["nb_neighbors"] = nb_neighbors_per_candidate
        ##return_dic["optimal_neighbors_numbers"] = optimal_neighbors_numbers
        #return_dic["optimal_percentile"] = optimal_percentile_values
        return_dic = {}
        for i in range(7):
            dic_entity  = top10ranked_entities[i]
            return_dic_ = {}
            return_dic_["score"] = dic_entity["score"]
            return_dic_["neighbors"] = [{"score" : ns, "neighbor_id" : ni, "neighbor_type" : nt} for ns, ni, nt in zip(new_temp_scores_total_main[i], neighbors_id[i], neighbors_type[i])]
            return_dic[dic_entity["entity_id"]] = return_dic_
            return_dic["gold_id"] = gold_id
        
        return return_dic


def returnNewScores(i, alpha):
    result_i = reScore(mentionToFeature[all_mentions_id[i]], alpha)
    rank_i = mentionToRank[all_mentions_id[i]]
    return rank_i, result_i

#['City', 'Settlement', 'Company', 'AdministrativeRegion', 'University']
mentions_cities_id = ["EL012254","EL012253","EL000634","EL002619","EL012826","EL013447","EL012823","EL012820","EL012825","EL013470","EL013471","EL013476","EL013475","EL001719","EL012833","EL013723","EL013462","EL013734","EL013739","EL000156","EL002634","EL001946","EL013762","EL013760","EL000449","EL012203","EL000680","EL001222","EL003255","EL012210","EL012044","EL012264","EL012266","EL012262","EL012263","EL001938","EL013058","EL012270","EL012831","EL012830"]

mentions_settlement_id = ["EL013755","EL012857","EL013744","EL013749","EL000836","EL000831","EL012193","EL013753","EL013750","EL000825","EL013773","EL012206","EL012201","EL012866","EL007215"] 

mentions_company_id = ["EL005455","EL012385","EL010113","EL011418","EL006532","EL006051","EL013621","EL013620","EL013619","EL006748","EL012107"]

mentions_administrativeRegion = ["EL012006","EL000549","EL000547","EL012037","EL000014","EL000019","EL002161","EL000029","EL002170","EL011216"]

mentions_university = ["EL013239","EL013237","EL013232","EL011037","EL010491","EL005505","EL005506","EL011027"]

mentions_town = ["EL013463", "EL012821", "EL012845", "EL012837", "EL013455", "EL013456"]
mentions_governmentAgency = ["EL013251", "EL005427", "EL005435", "EL005593", "EL005591"]
mentions_baseballPlayer = ["EL011839", "EL011838", "EL011835", "EL011836"] 
mentions_country = ["EL001632", "EL007679", "EL007673", "EL003201"]
mentions_tradeUnion = ["EL012927", "EL012929", "EL012928", "EL012925"]
mentions_legislature = ["EL010973", "EL010971", "EL010966", "EL010968"]



#('Settlement', 15), ('Company', 11), ('AdministrativeRegion', 10),

def finalReturn(alpha, return_types):
    become_good_entities = []
    become_mis_entities = []
    left_mis_entities = []
    final_count = []
    for i in range(len(all_mentions_id)):
        if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] in return_types:
        #if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "AdministrativeRegion":
        #if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "AdministrativeRegion" and  all_mentions_id[i] == "EL012034":
        #if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "Company":       
        #if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "City":    
        #if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "City" and all_mentions_id[i] == 'EL013770': 
        #if all_mentions_id[i] in mentions_cities_id:
        #if all_mentions_id[i] in mentions_cities_id and all_mentions_id[i] == "EL003255":
        #if all_mentions_id[i] in mentions_cities_id and all_mentions_id[i] == "EL013762":
        #if all_mentions_id[i] in mentions_cities_id and all_mentions_id[i] == "EL013760":
        #if all_mentions_id[i] in mentions_settlement_id:
        #if all_mentions_id[i] in mentions_company_id:
        #if all_mentions_id[i] in mentions_administrativeRegion:
        #if all_mentions_id[i] in mentions_legislature:
        #if 1:
            ok = returnNewScores(i, alpha)
            #print(ok[0])
            #print(np.argmax([ok_[2] for ok_ in ok[1]]))
            
            #pdb.set_trace()
            #if ok[0] == np.argmax([ok_[2] for ok_ in ok[1]]):
            if ok[0] == np.argmin([ok_[2] for ok_ in ok[1]]):
                #print("working")
                #print(all_mentions_id[i])
                final_count.append(1)
                if ok[0] != 0:
                    become_good_entities.append(all_mentions_id[i])
            else:
                if ok[0] == 0:
                    become_mis_entities.append(all_mentions_id[i])
                else:
                    left_mis_entities.append(all_mentions_id[i])        
                #print(all_mentions_id[i]) 
                #print(np.argmax([ok_[2] for ok_ in ok[1]]))

    return final_count, left_mis_entities, become_mis_entities, become_good_entities


#for alpha_ in [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
final_alphas = []
final_become_mis = []
final_become_good = []
final_mis = []
#for alpha_ in [0.2, 0.3, 0.75, 0.90, 0.95, 1]:ok = reScore(mentionToFeature['EL0022161'], 0.3)
countries_id = [key for key in entityIdToOntologyType.keys() if entityIdToOntologyType[key] in ['Country']]
settlements_id = [key for key in entityIdToOntologyType.keys() if entityIdToOntologyType[key] in ["Settlement"]]
adrs_id = [key for key in entityIdToOntologyType.keys() if entityIdToOntologyType[key] in ["AdministrativeRegion"]]

def filterIds(countries_id_, threshold):    
    cids_name = []
    for cid in countries_id_:
        cid_neighbors= G.neighbors(entityIdToIndex[cid])
        try:
            cid_name = entityIdToFeatures[cid]["entity_name"]
            cids_n_type = []
            for cid_n in cid_neighbors:
                try:
                    cid_n_type = entityIdToOntologyType[entityIndexToId[cid_n]]
                    if entityIdToOntologyType[cid] == "Country":
                        if cid_n_type not in ["UKN", "PER", "ORG", "GPE"]:
                            cids_n_type.append(cid_n_type)

                    else:
                        cids_n_type.append(cid_n_type)

                except:
                    print(cid_name)
    
            if "City" in cids_n_type and len(cids_n_type) > threshold:
                cids_name.append({"id": cid, "name": cid_name, "neighbors_type": cids_n_type})
             
        except:
            assert(1)
    
    return cids_name


counter_names = ['Durrani_Empire', 'Republic_of_Texas', 'Gold_Coast_(British_colony)'] 
cids_name = filterIds(countries_id, 1)
real_countries_id = [cn["id"] for cn in cids_name if cn["name"] not in counter_names]

sets_name = filterIds(settlements_id, 1)
sets_id = [sn["id"] for sn in sets_name if ",_" not in sn["name"]]

adrs_name = filterIds(adrs_id, 1)
adrs_id = [an["id"] for an in adrs_name]

real_ids = real_countries_id + sets_id + adrs_id
for key in entityIdToOntologyType.keys():
    if entityIdToOntologyType[key] in ['School', 'Politician', 'OfficeHolder', 'University', 'Company', 'AmericanFootballPlayer', 'SoccerPlayer', 'RadioStation', 'Wrestler', 'Band', 'MilitaryPerson', 'MusicalArtist', 'Congressman', 'IceHockeyPlayer', 'BaseballPlayer', 'Road']:
    #if entityIdToOntologyType[key] in ['School', 'Politician', 'OfficeHolder', 'University', 'Company', 'AmericanFootballPlayer', 'SoccerPlayer', 'Wrestler', 'Band', 'MilitaryPerson', 'MusicalArtist', 'Congressman', 'IceHockeyPlayer', 'BaseballPlayer', 'Road']:  # remove radio station 
        real_ids.append(key)

real_ids = set(real_ids).intersection(entityIdToFeatures.keys())



######################################################### CPU #############################################################
type_concerned = sys.argv[1]
mentions_concerned_id = [all_mentions_id[i] for i in range(len(all_mentions_id)) if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == type_concerned]
print(len(mentions_concerned_id))
mention_mis_list = [mentionToFeature[mci] for mci in mentions_concerned_id]
results1 = []
for i in range(len(mention_mis_list)):
    results1.append(reScore(mention_mis_list[i], 0.5))

pdb.set_trace()
###########################################################################################################################

######################################################### Spark ###########################################################
#type_concerned = sys.argv[1]
#mentions_concerned_id = [all_mentions_id[i] for i in range(len(all_mentions_id)) if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == type_concerned]
#print(len(mentions_concerned_id))
##nb_tasks = len(mentions_concerned_id)/2
#mention_mis_list = [mentionToFeature[mci] for mci in mentions_concerned_id]
#mention_mis_spark_list = sc.parallelize(mention_mis_list, 80)
#alpha_ = 0.5
#results_spark = mention_mis_spark_list.map(lambda x : reScore(x, alpha_))
#results_spark.saveAsTextFile("rescore_neighbors_collect_spark_" + type_concerned + ".json")
############################################################################################################################
