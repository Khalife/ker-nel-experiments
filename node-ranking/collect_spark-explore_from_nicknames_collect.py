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
from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
import pyspark
sc = SparkContext(appName="nel-system")
sess = pyspark.sql.SparkSession.builder.appName("nel-system").getOrCreate()

#{"mention_id" : mention_id, "mention_name" : mention_name, "gold_dic" : gold_dic, "gold_score" : gold_score, "prediction_ranks" : list_ranks[:10], "gold_rank" : predictio    n_rank}

#ranks_file = open("newTestA-OWT-ScoresNelMentions-2009.json", "r")
#ranks_file = open("SUBSAMPLES/GPE1-sample-10-500-JsonFileComplete-ORG-2009.json")
#ranks_file = open("subMentionsGPE-OWT-ScoresNelMentions-2009.json", "r")
CATEGORY = "GPE"
YEAR = "2009"
#ranks_file = open("sample-above-20-OWT-Scores-Complete-2009-" + category + "-UPDATE-NNIL.json", "r")
#ranks_file = open("sample-ranks-above-10-JsonFileUpdate-" + CATEGORY + "-" + YEAR + ".jsonc", "r")
#ranks_file = open("SUBSAMPLES/subMentions-above-500-JsonFileComplete-ORG-2009.json", "r")
#ranks_file = open("mentions-OWT-Scores-Complete-2014-GPE-TRAINING-UPDATE-NNIL.jsonaa", "r")
#ranks_filename = sys.argv[1]
#input_alpha_level = sys.argv[2]
#ranks_file = open(ranks_filename, "r")
#ranks_file = sc.textFile(ranks_filename)

#min_id_mention = int(sys.argv[1])
#max_id_mention = int(sys.argv[2])
#config = int(sys.argv[1])
#
#if config == 4:
#    seed_indexes = [i for i in range(10)]
#
#if config == 5:
#    seed_indexes = [i for i in range(5)] + [10, 20, 30, 50, 100]
#
#if config == 6:
#    seed_indexes = [i for i in range(5)] + [10, 15, 20, 25, 30]
#
#if config == 7:
#    seed_indexes = [i for i in range(5)] + [10, 15, 20, 30, 40]    

seed_indexes = [i for i in range(10)]    


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


#GPE_subtypes1 = ['Region', 'Settlement', 'City', 'Capital', 'CapitalOfRegion', 'CityDistrict', 'HistoricalSettlement', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
#GPE_subtypes1 = ['Region', 'City', 'Capital', 'CapitalOfRegion', 'CityDistrict', 'HistoricalSettlement', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
#GPE_subtypes1 = ['Country']

gold_types = ['BusCompany', 'ChessPlayer', 'ORG', 'AdministrativeRegion', 'Settlement', 'TennisPlayer', 'Town', 'PER', 'Politician', 'PoliticalParty', 'Airline', 'Band', 'Writer', 'PublicTransitSystem', 'Lake', 'AmericanFootballPlayer', 'Village', 'Actor', 'Comedian', 'University', 'OfficeHolder', 'TradeUnion', 'President', 'GPE', 'Governor', 'Country', 'SoccerClub', 'MilitaryUnit', 'MilitaryPerson', 'Non-ProfitOrganisation', 'Senator', 'Legislature', 'TelevisionStation', 'City', 'GovernmentAgency', 'BroadcastNetwork', 'Company', 'MusicalArtist', 'Model', 'BaseballPlayer', 'RadioStation', 'Congressman']

GPE_subtypes1 = {}
for gold_type in gold_types:
    GPE_subtypes1[gold_type] = ['City', 'Country']  


mis_ontologies = ['GovernmentAgency', 'MusicalArtist', 'AdministrativeRegion', 'Scientist', 'Non-ProfitOrganisation', 'Newspaper', 'Newspaper', 'ORG', 'PoliticalParty', 'City', 'Judge', 'GovernmentAgency', 'Company', 'ORG', 'Actor', 'Actor', 'GeopoliticalOrganisation', 'Newspaper', 'Company', 'PlayboyPlaymate', 'Hospital', 'Country', 'School', 'AmericanFootballPlayer']

for mo in mis_ontologies:
    GPE_subtypes1[mo] = ['City', 'Country']


GPE_subtypes1['City'] = ['Region', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
#GPE_subtypes1['City'] = ['AdministrativeRegion']
GPE_subtypes1['Settlement'] = ['GovernmentAgency'] + GPE_subtypes1['City']
GPE_subtypes1['TradeUnion'] = ['Country']
GPE_subtypes1['Legislature'] = ['Region', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
GPE_subtypes1['University'] = ['Region', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country', 'City']
GPE_subtypes1['Country'] = ['Region', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
GPE_subtypes1['Company'] = ['Country', 'Company', 'PER', 'MusicalArtist', 'Village', 'ORG', 'Settlement']
GPE_subtypes1['AdministrativeRegion'] = ['Settlement']
#GPE_subtypes1['AdministrativeRegion'] = ['Settlement', 'City', 'School', 'RadioStation', 'ORG', 'University', 'Town', 'Company', 'OfficeHolder', 'MusicalArtist', 'BaseballPlayer', 'PER']
#GPE_subtypes1['University'] = ['ORG', 'OfficeHolder', 'Judge', 'Congressman', 'AmericanFootballPlayer', 'Economist', 'PER']

GPE_subtypes1['Town'] = ['ORG', 'OfficeHolder', 'PER', 'Judge', 'AmericanFootballPlayer', 'SportsTeam', 'GPE']

GPE_subtypes1['BaseballPlayer'] = ['GPE']

GPE_subtypes1['Country'] = ['Town', 'PER', 'City', 'Settlement', 'MilitaryPerson', 'ORG', 'Village', 'MilitaryUnit', 'School', 'University', 'GPE', 'Company', 'OfficeHolder', 'MusicalArtist', 'Artist', 'SoccerPlayer', 'IceHockeyPlayer', 'Boxer', 'Writer', 'Wrestler', 'RecordLabel', 'SoccerManager', 'Comedian', 'Congressman', 'Politician', 'TradeUnion', 'TelevisionStation', 'TennisPlayer', 'Senator', 'Swimmer', 'ComicsCreator', 'Governor', 'Airline', 'Non-ProfitOrganisation', 'Economist', 'BroadcastNetwork', 'Model', 'RacingDriver', 'BaseballLeague', 'ProtectedArea', 'Judge', 'President', 'Mayor', 'IceHockeyLeague', 'SoccerLeague', 'BasketballLeague', 'BaseballPlayer', 'MartialArtist', 'RadioStation', 'Country', 'NascarDriver', 'AdministrativeRegion']

#('GovernmentAgency', 5), ('BaseballPlayer', 4), ('Country', 4), ('TradeUnion', 4), ('Legislature', 4)


# TradeUnion {u'Country': 1} 

# Country [(u'Town', 14761), (u'PER', 9797), (u'City', 9450), (u'Settlement', 7296), (u'MilitaryPerson', 3190), (u'ORG', 3146), (u'Village', 2614), (u'MilitaryUnit', 2223), (u'School', 1674), (u'University', 1290), (u'GPE', 943), (u'Company', 714), (u'OfficeHolder', 670), (u'MusicalArtist', 629), (u'Artist', 555), (u'SoccerPlayer', 349), (u'IceHockeyPlayer', 317), (u'Boxer', 246), (u'Writer', 208), (u'Wrestler', 195), (u'RecordLabel', 191), (u'SoccerManager', 171), (u'Comedian', 163), (u'Congressman', 155), (u'Politician', 145), (u'TradeUnion', 137), (u'TelevisionStation', 130), (u'TennisPlayer', 124), (u'Senator', 104), (u'Swimmer', 88), (u'ComicsCreator', 87), (u'Governor', 71), (u'Airline', 65), (u'Non-ProfitOrganisation', 64), (u'Economist', 56), (u'BroadcastNetwork', 56), (u'Model', 55), (u'RacingDriver', 55), (u'BaseballLeague', 47), (u'ProtectedArea', 42), (u'Judge', 32), (u'President', 27), (u'Mayor', 27), (u'IceHockeyLeague', 24), (u'SoccerLeague', 23), (u'BasketballLeague', 23), (u'BaseballPlayer', 21), (u'MartialArtist', 16), (u'RadioStation', 14), (u'Country', 14), (u'NascarDriver', 14), (u'AdministrativeRegion', 13), (u'LacrossePlayer', 13), (u'AmericanFootballLeague', 12), (u'Murderer', 11), (u'MemberOfParliament', 9), (u'SportsLeague', 8), (u'AmericanFootballPlayer', 8), (u'Philosopher', 7), (u'Publisher', 7), (u'BritishRoyalty', 7), (u'CollegeCoach', 7), (u'Actor', 6), (u'Building', 6), (u'InlineHockeyLeague', 5), (u'PlayboyPlaymate', 5), (u'FormulaOneRacer', 5), (u'AmateurBoxer', 5), (u'Saint', 5), (u'PoliticalParty', 4), (u'GovernmentAgency', 4), (u'AutoRacingLeague', 4), (u'GridironFootballPlayer', 4), (u'LacrosseLeague', 4), (u'Cricketer', 3), (u'AdultActor', 3), (u'Library', 3), (u'RugbyLeague', 3), (u'ChristianBishop', 3), (u'VideogamesLeague', 3), (u'River', 3), (u'Medician', 2), (u'HistoricPlace', 2), (u'GolfPlayer', 2), (u'BasketballPlayer', 2), (u'TennisLeague', 2), (u'RugbyPlayer', 2), (u'VolleyballLeague', 2), (u'SoccerClub', 1), (u'CricketLeague', 1), (u'Bodybuilder', 1), (u'Legislature', 1), (u'PrimeMinister', 1), (u'MotorcycleRider', 1), (u'Scientist', 1), (u'CricketTeam', 1), (u'TelevisionHost', 1), (u'Curler', 1), (u'Ambassador', 1), (u'TableTennisPlayer', 1), (u'GeopoliticalOrganisation', 1), (u'Island', 1), (u'LawFirm', 1)]



# BaseballPlayer  {u'GPE': 1}

#'GovernmentAgency' {u'GovernmentAgency': 2, u'Settlement': 2, u'OfficeHolder': 2}


# Town 
#{u'City': 1, u'Writer': 1, u'Artist': 1, u'SportsTeam': 1, u'GPE': 1, u'University': 1, u'CollegeCoach': 1, u'AmericanFootballPlayer': 18, u'PER': 114, u'Mayor': 1, u'LacrossePlayer': 4, u'Congressman': 2, u'Philosopher': 1, u'OfficeHolder': 9, u'Country': 1, u'Judge': 3, u'ORG': 9, u'Economist': 2, u'SoccerPlayer': 1}

#Company {u'City': 1, u'CollegeCoach': 1, u'GPE': 1, u'Artist': 2, u'AmericanFootballPlayer': 21, u'University': 2, u'Writer': 1, u'SportsTeam': 1, u'PER': 214, u'LacrossePlayer': 8, u'Philosopher': 2, u'OfficeHolder': 17, u'Country': 1, u'Judge': 5, u'Congressman': 4, u'Economist': 4, u'Mayor': 1, u'SoccerPlayer': 2, u'ORG': 16}


#['GovernmentAgency', 'City', 'Wrestler', 'Swimmer', 'Murderer', 'MotorcycleRider', 'Writer', 'Non-ProfitOrganisation', 'PER', 'TradeUnion', 'MilitaryPerson', 'Saint', 'Congressman', 'Politician', 'InlineHockeyLeague', 'MartialArtist', 'TennisPlayer', u'RecordLabel', u'Cricketer', u'Company', u'NascarDriver', u'BroadcastNetwork', u'Boxer', u'MusicalArtist', u'Judge', u'President', u'IceHockeyPlayer', u'LawFirm', u'Actor', u'Town', u'School', u'EducationalInstitution', u'ProtectedArea', u'BritishRoyalty', u'Artist', u'GPE', u'University', u'SoccerManager', u'AmericanFootballPlayer', u'Governor', u'MilitaryUnit', u'Architect', u'Philosopher', u'OfficeHolder', u'Senator', u'Model', u'RadioStation', u'RacingDriver', u'Airline', u'Legislature', u'SoccerClub', u'TelevisionStation', u'SoccerPlayer', u'Mayor', u'Comedian', u'Village', u'ORG', 'Settlement', 'Economist', 'BaseballPlayer', 'ComicsCreator']

#with open("ALL-TYPES.txt", "r") as f:
#    GPE_subtypes1 = f.readline().strip().split(", ")
entityIdToName = json.load(open("../entityIdToIndexName.json", "r"))
typeToThreshold = json.load(open("../typeToThreshold.json", "r"))

#pdb.set_trace()
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
                        
                            ######################### Sum ###############################

                            # Compute min over all neighbors and keep lowest#
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
                        #pdb.set_trace() 
                    
                    #pdb.set_trace()
                    #new_temp_scores_total = new_temp_scores_total_main[i]

                    # Neighbors selection
                    # a. type selection
                    #pdb.set_trace()
                    #neighbors_type_selection = set(neighbors_type[0]).intersection(*neighbors_type[:1])
                    #neighbors_type_selection = list(neighbors_type_selection) + ["Country", "Settlement", "Town", "AdministrativeRegion"]
                    #new_temp_scores_total  = [ntst for ntst, nt in zip(new_temp_scores_total, neighbors_type[i]) if nt in neighbors_type_selection] 
                     
                    # b. neighbors selection
                    #new_temp_scores_total_sorted = np.sort(new_temp_scores_total).tolist()[:min([nb_max_neighbors, len(new_temp_scores_total)])]
                    #new_temp_scores_total = new_temp_scores_total_sorted 

                    # Percentile selection
                    #threshold = np.percentile(new_temp_scores_total, percentile_neighbors)
                    #new_temp_scores_total = [ntst for ntst in new_temp_scores_total if ntst <= threshold]         
            
                    #alpha_level = input_alpha_level 
                    #alpha_percentile = np.percentile(new_temp_scores_total, alpha_level)                
                    
                    #new_temp_scores_total_filtered = [ntst for ntst in new_temp_scores_total if ntst <= alpha_percentile]
                    #if len(new_temp_scores_total_sorted) < len(new_temp_scores_total_filtered):
                    #    new_temp_scores_total = new_temp_scores_total_filtered
                    #else:
                    #    assert(1)

                    #pdb.set_trace()
                    #offset = -min([0, min(new_temp_scores_total)])
                    #new_temp_scores_total = [ntst + offset for ntst in new_temp_scores_total] # get positive scores
                    #normalization_factor = float(max([1, EC.shape[0]]))
                    #normalization_factor = float(max([1, len(new_temp_scores_total)]))
                    #original_modified_degree = EC.shape[0]
                    #score_total = (np.log(i+2)/normalization_factor)*sum(new_temp_scores_total)
                    #score_total = (1/normalization_factor)*sum(new_temp_scores_total)
                    #log_factor = np.log(5+i)/np.log(5)
                    #log_factor = 1
                    #score_total = (log_factor/normalization_factor)*sum(new_temp_scores_total)
                    
                    #new_scores_total.append(score_total)
                    #################################################

                #max_new = max(new_scores_total)                                                                                                   
                #pdb.set_trace()
                #for i in range(7):
                #    top_score = top_scores[i]
                #    final_score = alpha*new_scores_total[i]/(max_new*2) - (1-alpha)*top_score/4.
                #    scores_total.append([new_scores_total[i]/(max_new), top_score, final_score])
                #                                                                                                                                  
                #                                                                                                                                  
                #if len(no_neighbor_candidate) > 0:
                #    #pdb.set_trace()
                #    no_neighbor_value = np.percentile([scores_total[i][0] for i in range(len(scores_total)) if i not in no_neighbor_candidate],7)
                #    for i in no_neighbor_candidate:
                #        scores_total[i][0] = no_neighbor_value
                #        final_score = alpha*scores_total[i][0]/2. - (1-alpha)*top_scores[i]/4.
                #        scores_total[i][2] = final_score 

                #argmin_s_tau = np.argmin([sc[2] for sc in scores_total])
                #argmins_s_tau.append(argmin_s_tau)

                #############################################################

                ######################## Max ################################
                #new_score_total_ = []
                #for i_n in range(EC.shape[0]):
                #    new_score_total_.append(EC[i_n].data.dot(EC[i_n].data) - 2*mention_country_and_co_entity_comparison[i_n][0,0])

                #if len(new_score_total_) == 0:
                #    if i  == 0:
                #        no_neighbor_candidate.append(i)
                #        new_score_total_ = 0 
                #    else:
                #        new_score_total = M0.data.dot(M0.data)
    
                #else:
                #    new_score_total_ = min(new_score_total_)
                #    new_score_total_ += M0.data.dot(M0.data)

                #new_score_total = new_score_total_
                #score_total = new_score_total
                #new_scores_total.append(score_total) 
                #############################################################


                #top_score = top_scores[i]
                #new_score = alpha*(-top_score/4. + score_total/2.) - (1-alpha)*top_score/4.
                #new_score = alpha*score_total/2 - (1-alpha)*100*top_score/4.
                #new_score = min([alpha*(top_score/4. + score_total/2.) + (1-alpha)*top_score/4., 1]) 
                #scores_total.append([score_total, top_score, new_score])
            #print(mention_name)            

 


        #except:
        #    print("WARNIIIIIIINNNNG ISSUEEEEEE")
        #    pdb.set_trace()
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

def returnNeighbors(mention_id, frequency_threshold):
    gold_id = mentionToFeature[mention_id]["gold_entity_id"]
    gold_type = entityIdToOntologyType[gold_id]
    neighbors = G.neighbors(entityIdToIndex[gold_id])
    clean_neighbors = []
    corrupted_neighbors = []

    neighbor_frequency = {}
    top10ranked_entities = mentionToTopEntities[mention_id][:12]
    for top_entity in top10ranked_entities[:12]:
        neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
        for nte in neighbors_top_entity:
            neighbor_frequency[nte] = 0
                                                                                         
                                                                                         
    for top_entity in top10ranked_entities[:12]:
        neighbors_top_entity = G.neighbors(entityIdToIndex[top_entity["entity_id"]])
        for nte in neighbors_top_entity:
            neighbor_frequency[nte] += 1

    for nei in neighbors:
        try:
            if entityIdToOntologyType[entityIndexToId[nei]] in GPE_subtypes1[gold_type] and neighbor_frequency[nte] <= frequency_threshold:
                clean_neighbors.append([entityIndexToId[nei], entityIdToOntologyType[entityIndexToId[nei]], entityIdToName[entityIndexToId[nei]]])
        except:
            assert(1)

    for corrupted_entity in top10ranked_entities:
        corrupted_neighbors_ = []
        if corrupted_entity["entity_id"] != gold_id:
            entity_id = corrupted_entity["entity_id"]
            neighbors = G.neighbors(entityIdToIndex[entity_id])
            for nei in neighbors:
                try:
                    if entityIdToOntologyType[entityIndexToId[nei]] in GPE_subtypes1[gold_type] and neighbor_frequency[nte] <= frequency_threshold:
                        corrupted_neighbors_.append([entityIndexToId[nei], entityIdToOntologyType[entityIndexToId[nei]], entityIdToName[entityIndexToId[nei]]])
                except:
                    assert(1)
        corrupted_neighbors.append(corrupted_neighbors_)

    return {"gold_neighbors": clean_neighbors, "corrupted_neighbors": corrupted_neighbors}


#for alpha_ in [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
final_alphas = []
final_become_mis = []
final_become_good = []
final_mis = []
#for alpha_ in [0.2, 0.3, 0.75, 0.90, 0.95, 1]:ok = reScore(mentionToFeature['EL0022161'], 0.3)
#pdb.set_trace()
#city_gold_neighbors_name = [nei[2]["entity_name"] for nei in returnNeighbors('EL012254', 1)["gold_neighbors"]]
#city_corrupted_neighbors_name =  [" ".join([nei[2]["entity_name"] for nei in nej]) for nej in returnNeighbors('EL012254', 1)["corrupted_neighbors"]]
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



#ok = reScore(mentionToFeature["EL012254"], 0.3)
['Region', 'Town', 'Village', 'State', 'AdministrativeRegion', 'Country']
GPE_subtypes1["City"] = ['AdministrativeRegion', 'Country', 'RadioStation', 'Road', 'OfficeHolder', 'MusicalArtist', 'School', 'BaseballPlayer', 'MilitaryPerson', 'Settlement', 'Company', 'University', 'Building', 'SoccerPlayer', 'IceHockeyPlayer', 'AmericanFootballPlayer', 'Wrestler', 'Politician', 'Congressman', 'Band', 'Writer', 'Governor', 'TelevisionStation', 'Artist', 'Airport', 'HistoricPlace', 'BasketballPlayer', 'SoccerManager', 'Airline', 'ProtectedArea', 'MemberOfParliament', 'Senator', 'PoliticalParty', 'Boxer', 'Cricketer', 'MilitaryUnit', 'Town', 'TradeUnion', 'NascarDriver', 'Stadium', 'BritishRoyalty', 'TennisPlayer', 'Bridge', 'City', 'Judge', 'ComicsCreator', 'ShoppingMall', 'RugbyPlayer', 'Comedian', 'Mayor', 'GridironFootballPlayer', 'RecordLabel', 'President', 'Lake', 'Non-ProfitOrganisation', 'Museum', 'Station', 'Swimmer', 'SoccerClub', 'Model', 'GaelicGamesPlayer', 'Saint', 'PrimeMinister', 'ChristianBishop', 'GovernmentAgency', 'Island', 'Legislature', 'LawFirm', 'PlayboyPlaymate', 'RailwayStation', 'Philosopher', 'RacingDriver', 'SkiArea', 'Park', 'Economist', 'Gymnast', 'MartialArtist', 'Monarch', 'AustralianRulesFootballPlayer', 'Theatre', 'RailwayLine', 'River', 'Planet', 'Lighthouse', 'FigureSkater', 'Architect', 'Actor', 'ReligiousBuilding', 'PublicTransitSystem', 'HistoricBuilding', 'BusCompany', 'MilitaryStructure', 'LacrossePlayer', 'Mountain', 'ChessPlayer', 'CollegeCoach'][:20]



GPE_subtypes1["City"] = ['School', 'Politician', 'OfficeHolder', 'University', 'Country', 'Company', 'AmericanFootballPlayer', 'SoccerPlayer', 'RadioStation', 'Wrestler', 'Band', 'AdministrativeRegion', 'MilitaryPerson', 'MusicalArtist', 'Congressman', 'IceHockeyPlayer', 'Settlement', 'BaseballPlayer', 'Road'] # intersection gold and corrupted


#mentions_city_id = [all_mentions_id[i] for i in range(len(all_mentions_id)) if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == "City"]

#potential_mis_city_ids = []
#for mci in mentions_city_id:
#    mentionTopEntites = [me["entity_id"] for me in mentionToTopEntities[mci]]
#    mentionGoldEntity = mentionToFeature[mci]["gold_entity_id"]
#    if mentionGoldEntity in mentionTopEntites:
#        if mentionTopEntites.index(mentionGoldEntity) == 0:
#            potential_mis_city_ids.append(mci)

#pdb.set_trace()
#ok = returnNewScores(18, 0.3)
#pdb.set_trace()
#ok = finalReturn(0.5, ["City"])
#pdb.set_trace()
#neighbors_ok = [[ne for ne in G.neighbors(entityIdToIndex[mentionToFeature[k]["gold_entity_id"]]) if entityIndexToId[ne] in real_ids] for k in ok[3]]

##### Spark debug #####
#mention_mis_list = [mentionToFeature[mci] for mci in mentions_city_id]#[:20]
#results_spark = []
#for i in range(5):
#    results_spark.append(reScore(mention_mis_list[i], 0.3))

#ok = returnNewScores(all_mentions_id.index('EL013061'), 0.5)
#pdb.set_trace()
#################

##### Spark #####
type_concerned = sys.argv[1]
mentions_concerned_id = [all_mentions_id[i] for i in range(len(all_mentions_id)) if entityIdToOntologyType[mentionToFeature[all_mentions_id[i]]["gold_entity_id"]] == type_concerned]
print(len(mentions_concerned_id))
#nb_tasks = len(mentions_concerned_id)/2
mention_mis_list = [mentionToFeature[mci] for mci in mentions_concerned_id]
mention_mis_spark_list = sc.parallelize(mention_mis_list, 80)
alpha_ = 0.5
results_spark = mention_mis_spark_list.map(lambda x : reScore(x, alpha_))
results_spark.saveAsTextFile("rescore_neighbors_collect_spark_" + type_concerned + ".json")
#################
