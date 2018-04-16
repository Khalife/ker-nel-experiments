import pdb
import numpy as np
import json

type_concerned = "AdministrativeRegion"
type_concerned = "University"
type_concerned = "Company"
type_concerned = "GovernmentAgency"
type_concerned = "BaseballPlayer"


datafile = open("rescore_neighbors_collect_spark_" + type_concerned + ".json", "r")
datafile_train = open("rescore_neighbors_collect_spark_train_" + type_concerned + ".json", "r")
datafile_train_2014 = open("rescore_neighbors_collect_spark_train_2014_" + type_concerned + ".json", "r")

example = {u'E0503975': {'neighbors': [], 'score': 0.45657054936916297}, 'gold_id': u'E0710271', u'E0374326': {'neighbors': [], 'score': 0.2947042095338972}, u'E0012327': {'neighbors': [], 'score': 0.3125166860296959}, u'E0710271': {'neighbors': [], 'score': 1.1141315129872948}, u'E0234156': {'neighbors': [], 'score': 0.2673218635357505}, u'E0241257': {'neighbors': [], 'score': 0.3464330949537297}, u'E0291064': {'neighbors': [], 'score': 0.28223286174030066}}



types = ['AdministrativeRegion', 'Country', 'RadioStation', 'Road', 'OfficeHolder', 'MusicalArtist', 'School', 'BaseballPlayer', 'MilitaryPerson', 'Settlement', 'Company', 'University', 'Building', 'SoccerPlayer', 'IceHockeyPlayer', 'AmericanFootballPlayer', 'Wrestler', 'Politician', 'Congressman', 'Band', 'Writer', 'Governor', 'TelevisionStation', 'Artist', 'Airport', 'HistoricPlace', 'BasketballPlayer', 'SoccerManager', 'Airline', 'ProtectedArea', 'MemberOfParliament', 'Senator', 'PoliticalParty', 'Boxer', 'Cricketer', 'MilitaryUnit', 'Town', 'TradeUnion', 'NascarDriver', 'Stadium', 'BritishRoyalty', 'TennisPlayer', 'Bridge', 'City', 'Judge', 'ComicsCreator', 'ShoppingMall', 'RugbyPlayer', 'Comedian', 'Mayor', 'GridironFootballPlayer', 'RecordLabel', 'President', 'Lake', 'Non-ProfitOrganisation', 'Museum', 'Station', 'Swimmer', 'SoccerClub', 'Model', 'GaelicGamesPlayer', 'Saint', 'PrimeMinister', 'ChristianBishop', 'GovernmentAgency', 'Island', 'Legislature', 'LawFirm', 'PlayboyPlaymate', 'RailwayStation', 'Philosopher', 'RacingDriver', 'SkiArea', 'Park', 'Economist', 'Gymnast', 'MartialArtist', 'Monarch', 'AustralianRulesFootballPlayer', 'Theatre', 'RailwayLine', 'River', 'Planet', 'Lighthouse', 'FigureSkater', 'Architect', 'Actor', 'ReligiousBuilding', 'PublicTransitSystem', 'HistoricBuilding', 'BusCompany', 'MilitaryStructure', 'LacrossePlayer', 'Mountain', 'ChessPlayer', 'CollegeCoach'][:20]

#types = [types[0], types[1], "Town", "City", "Settlement"]

nb_entities = 4 


def returnXData(dataFile_):
    X = []
    counter_data = 0
    nb_cities_not_first = 0
    for line in dataFile_:
        counter_data += 1
        X_mention = []
        try:
            dic_line = json.loads(line)
        except:
            pdb.set_trace()
        gold_id = dic_line["gold_id"]
        candidate_entities = [key for key in dic_line.keys() if "E" in key]
        dic_candidates_score = [dic_line[ce]["score"] for ce in candidate_entities]
        dic_entities = [(ce, sc) for ce, sc in zip(candidate_entities, dic_candidates_score)]
        dic_entities_sorted = sorted(dic_entities, key = lambda x :x [1], reverse = True)
        candidate_entities = [des[0] for des in dic_entities_sorted][:nb_entities]  # number of elements considered 
        if gold_id in candidate_entities:
            index_gold = candidate_entities.index(gold_id)
            if index_gold > 0: 
                gold_candidate = candidate_entities[index_gold]
                candidate_entities[index_gold] = candidate_entities[0]
                candidate_entities[0] = gold_candidate
        
        



            for ce in candidate_entities:
                X_line = []    
                s0 = dic_line[ce]["score"]
                X_line.append(s0)
                for type_ in types:
                    neighbors_of_type = [nei for nei in dic_line[ce]["neighbors"] if nei["neighbor_type"] == type_]
                    neighbors_of_type_scores = [nei["score"] for nei in neighbors_of_type]
                    if len(neighbors_of_type_scores) == 0:
                        X_line.append(0)
                    else:
                        X_line.append(min(neighbors_of_type_scores))
                X_mention.append(X_line) 
            X.append(X_mention)

        else:
            nb_cities_not_first += 1    

    print("nb_data : " + str(counter_data)) 
    print(nb_cities_not_first)
    return X

X_train1 = returnXData(datafile_train)
X_train2 = returnXData(datafile_train_2014)
X_train2 = [[[x_t_ if x_t_ != "nan" else 0.5 for x_t_ in x_t] for x_t in x_t_2] for x_t_2 in X_train2]
X_train = X_train1 + X_train2

X_test = returnXData(datafile)

Y_check = [int(np.argmax([x_[0] for x_ in x]) == 0) for x in X_test]




#Y = [[0.7*x_[0] - 0.3*sum(x_[1:])/2. for x_ in x] for x in X]


# 1st method : LP 

#XX = [0 for x in range(len(X[0][0]))]
#for i in range(len(X)):
#    for j in range(1, len(X[0])):
#        for k in range(len(X[0][0])):
#            XX[k] += ( X[i][0][k] - X[i][j][k] ) 
#
#
#from cvxopt import matrix, solvers

#A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
#b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
#c = matrix([ 2.0, 1.0 ])

# Solves min <c, x> , s.t Ax <= b
#sol=solvers.lp(c,A,b)

#c = -matrix(XX)
#n = len(XX)
#
#def lp_matrix(i, j, n_):
#    if i <= n_-1:
#        if i == j:
#            return -1.0
#        else:
#            return 0.0
#    if i == n_:
#        return 1.0
#    if i == n_+1:
#        return -1.0
#    
#def lp_vector(i, n_):
#    if i <= n_ - 1:
#        return 0.0
#    if i == n_:
#        return 1.0
#    if i == n_+1:
#        return -1.0
#
#
#A = matrix([[lp_matrix(i, j, n) for i in range(n+2)] for j in range(n)])
#b = matrix([lp_vector(i, n) for i in range(n+2)])
#
#pdb.set_trace()
#sol=solvers.lp(c,A,b)

# 2nd method : SVM

#from sklearn import svm
#
#clf = svm.SVC()
#Y = sum([[1] + [0 for i in range(nb_entities-1)] for j in range(len(X))], []) 
#XX = sum(X, []) 
#clf.fit(XX, Y)
#print(clf.score(XX, Y))
#pdb.set_trace()

# 3rd method : regression trees

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
#pdb.set_trace()
Y_train = sum([[1] + [0 for i in range(nb_entities-1)] for j in range(len(X_train))], []) 
XX_train = sum(X_train, []) 

Y_test = sum([[1] + [0 for i in range(nb_entities-1)] for j in range(len(X_test))], [])
X_test = sum(X_test, [])


from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor

# All dataset
results_gold = []

for penal in [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]:
#for depth in range(2, 30):
#for alpha_ in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5, 10]:
#for n_estimator in [5, 10, 50, 100, 200]:
    #predictor = 1
    #regr_2 = DecisionTreeRegressor(max_depth=depth)
    #regr_2.fit(XX_train,Y_train)
    #results = regr_2.predict(X_test).tolist()

    #predictor = 1
    #regr_2 = DecisionTreeClassifier(max_depth=depth, class_weight = "balanced" )
    #regr_2.fit(XX_train,Y_train)
    #results = regr_2.predict_proba(X_test).tolist()
    #results = [res[1] for res in results]

    predictor = 6
    regr_2 = linear_model.LogisticRegression(C = penal, class_weight = "balanced")
    regr_2.fit(XX_train,Y_train)
    results = regr_2.predict_proba(X_test).tolist()
    results = [res[1] for res in results]



    #predictor = 2
    #regr_2 = Ridge(alpha=alpha_)
    #regr_2.fit(XX_train, Y_train)
    #results = regr_2.predict(X_test).tolist()

    #predictor = 3
    #regr_2 = linear_model.LinearRegression()
    #regr_2.fit(XX_train, Y_train)
    #results = regr_2.predict(X_test).tolist()

    #predictor  = 4
    #regr_2 = RandomForestRegressor(n_estimators=n_estimator, oob_score=True, random_state=0, max_features="log2")
    #regr_2.fit(XX_train, Y_train)
    #results = regr_2.predict(X_test).tolist()

    #if alpha_ >= 2:
    #if n_estimator >= 50:
        #pdb.set_trace()
    result_gold = 0
    if predictor == 1 or predictor == 6:
        for i in range(len(Y_test)/nb_entities):
            #pdb.set_trace()
            max_regression = np.max([res for res in results[(nb_entities*i):(nb_entities*(i+1))]])
            if len([res for res in results[(nb_entities*i):(nb_entities*(i+1))] if res == max_regression]) == 1:
            #if len([res for res in results[(nb_entities*i):(nb_entities*(i+1))] if res == 1.0]) == 1:
            #if len(np.argwhere(results[(nb_entities*i):(nb_entities*(i+1))]) == np.argmax(results[(nb_entities*i):(nb_entities*(i+1))])) == 1:
                if np.argmax(results[(nb_entities*i):(nb_entities*(i+1))]) == 0:
                    result_gold += 1
        #result_gold = sum([int( np.argwhere(results[(nb_entities*i+1):(nb_entities*(i+1))]) == np.amax(results[(nb_entities*i+1):(nb_entities*(i+1))]) == 0) for i in range(len(Y_test)/nb_entities)])

    else:
        for i in range(len(Y_test)/nb_entities):
            if np.argmax(results[(nb_entities*i):(nb_entities*(i+1))]) == 0:
                 
                result_gold += 1
            else:
                print(np.argmax(results[(nb_entities*i):(nb_entities*(i+1))]))

    results_gold.append(result_gold)


pdb.set_trace()

#By batch
#grid_results_alpha = []
#grid_results_max = []
#grid_results_strict_max = []
#for j in range(len(Y_test)/(10*nb_entities)):
##for j in range(len(Y_test)/nb_entities):
#    width = 10*nb_entities
#    #width = 1*nb_entities
#    if j == 0:
#        index_test = [k for k in range(width*j,width*(j+1))]
#        index_train = [k for k in range(width*(j+1),len(Y_test))]
#    elif j == ( len(Y_test)/nb_entities - 1):
#        index_test = [k for k in range(width*j, len(Y_test))]
#        index_train = [k for k in range(width*j)]
#    else:
#        index_train = [k for k in range(width*j)] + [k for k in range(width*(j+1),len(Y_test))]
#        index_test = [k for k in range(width*j, width*(j+1))] 
#
#    #if j == 15:
#    #    pdb.set_trace()
#    YY_test = [Y_test[i_te] for i_te in index_test]
#    Y_train = [Y_test[i_tr] for i_tr in index_train] # self train
#    #Y_train = [Y[i_tr] for i_tr in index_train]
#    XX_test = [X_test[i_te] for i_te in index_test]
#    XX_train = [X_test[i_tr] for i_tr in index_train] # self train
#    #XX_train = [XX[i_tr] for i_tr in index_train]
#
#    grid_results_alpha_ = []
#    grid_results_max_ = [] 
#    grid_results_strict_max_ = []
#
#    for depth in range(10, 30):
#        #regr_2 = DecisionTreeRegressor(max_depth=depth)
#        #regr_2.fit(XX_train, Y_train)
#        #results = regr_2.predict(XX_test).tolist()
#        
#        predictor = 2
#        regr_2 = Ridge(alpha=1.0)
#        regr_2.fit(XX_train, Y_train)
#        results = regr_2.predict(XX_test).tolist()
#    
#        def returnAlphaDecision(alpha):
#            corrupted = [sum([int(res_ < alpha) for res_ in results[(nb_entities*i+1):(nb_entities*(i+1))]]) for i in range(len(YY_test)/nb_entities)] 
#            gold = [int(results[nb_entities*i] >= alpha) for i in range(len(YY_test)/nb_entities)]
#            return len([co + go for co, go in zip(corrupted, gold) if co + go == nb_entities])
#      
#        def returnMaxDecision():
#            result_gold = [int(np.argmax(results[(nb_entities*i+1):(nb_entities*(i+1))]) == 0) for i in range(len(YY_test)/nb_entities)] 
#            return sum(result_gold)
#
#        #print("alpha decision : ")
#        #print([returnAlphaDecision(alpha) for alpha in np.arange(0,1, 0.05)])
#        #print("max decision : ")
#        #print(returnMaxDecision())
#   
#        def returnStrictMaxDecision():
#            result_gold = 0
#            for i in range(len(YY_test)/nb_entities):                                                                                                                          
#                if predictor == 1:
#                    if len([res for res in results[(nb_entities*i):(nb_entities*(i+1))] if res == 1.0]) == 1:
#                        if np.argmax(results[(nb_entities*i):(nb_entities*(i+1))]) == 0:
#                            result_gold += 1
#                    else:
#                        if depth == 29:
#                            print(results[(nb_entities*i):(nb_entities*(i+1))])
#                else:
#                    if np.argmax(results[(nb_entities*i):(nb_entities*(i+1))][:4]) == 0:
#                        result_gold += 1
#                    else:
#                        print(results[(nb_entities*i):(nb_entities*(i+1))])
#            return result_gold 
#            #result_gold = sum([int( np.argwhere(results[(nb_entities*i+1):(nb_entities*(i+1))]) == np.amax(results[(nb_entities*i+1):(nb_entities*(i+1))]) == 0) for i in range(len(Y_test)/nb_entities)])
# 
# 
#        grid_results_alpha_.append(np.max([returnAlphaDecision(alpha) for alpha in np.arange(0,1, 0.05)]))
#        grid_results_max_.append(returnMaxDecision())
#        grid_results_strict_max_.append(returnStrictMaxDecision())
#
#    grid_results_alpha.append(np.max(grid_results_alpha_))
#    grid_results_max.append(np.max(grid_results_max_))
#    grid_results_strict_max.append(np.max(grid_results_strict_max_))
#
#pdb.set_trace()
#from sklearn.tree import _tree
#
#
#def tree_to_code(tree, feature_names):
#    tree_ = tree.tree_
#    feature_name = [
#        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#        for i in tree_.feature
#    ]
#    print "def tree({}):".format(", ".join(feature_names))
#
#    def recurse(node, depth):
#        indent = "  " * depth
#        if tree_.feature[node] != _tree.TREE_UNDEFINED:
#            name = feature_name[node]
#            threshold = tree_.threshold[node]
#            print "{}if {} <= {}:".format(indent, name, threshold)
#            recurse(tree_.children_left[node], depth + 1)
#            print "{}else:  # if {} > {}".format(indent, name, threshold)
#            recurse(tree_.children_right[node], depth + 1)
#        else:
#            print "{}return {}".format(indent, tree_.value[node])
#
#    recurse(0, 1)
#
#
#
#
#pdb.set_trace()
#
#
#
#                
#tree.export_graphviz(regr_2, out_file='tree_el.dot') 
