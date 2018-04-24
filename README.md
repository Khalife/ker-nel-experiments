This repository is dedicated to research experiments for **Graph based named of entities**.
Please find below our comments to make these experiments reproducible.

In particular, we invite reviewers to refer to the detailed comment section concerning our implementation of preprocessing, filtering and score features extraction.

These modules were designed as one global routine for Named entity identification/linking problem, they are relatively independant though. 

[Some of these modules are compatible with Spark, please refer to comment section]

# Dependencies  
These experiments require

- Python [>=2.7, >= 3.5]
- NumPy [>= 1.8.2]
- SciPy [>= 0.13.3]
- scikit-learn [>= 0.15.2, >= 0.18.1]
- nltk [>= 3.2.3]

# Routine

- Parse NIST TAC-KBP Datasets, write .json files (preprocessing/tac-kbp.py, preprocessing/newParseMentions.py) and generate knowledge graph represented by nodes and egdes in a text file) (preprocessing/build_graph.py) 
- Create dictionnary entity id to ontology type using DBPedia (preprocessing/extract_ontology_dbpedia.py, preprocessing/entityIdToOntologyType.py
, preprocessing/update-ontology.py)
- Generate TF-IDF sparse matrices files .npz (filtering/prepare_all_data.py)
- Entity filtering (filtering/new_types_collect_complete_score_spark1.py)
- Graph features extraction with KER-NEL for train and test data (node-ranking/collect_spark-explore_from_nicknames_collect.py)
- Entity identification with supervised learning (node-ranking/try_rank.py)


## 1 - Preprocessing module

This module is used to :  
- parse NIST TAC-KBP (2009 to 2014) datasets and build queries and knowledge base/graph dataset.
- provide a fine-grained ontology entity classification using DBPedia 2016
- compute TF-IDF matrices on the knowledge base, and apply on mentions datasets




## 2 - Filtering method [Hadoop/Spark compatible]

This module provides a filtering algorithm to discard less similar entities to a given query. It allows to return a smaller set of entity candidates.
Input must be in .json format, and can be generated with preprocessing module, following template :


{"mention_name": "Cambridge", "mention_index": 2, "mention_id": "EL000634", "gold_entity_id": "E0272861"}

Also,
 
- Knowledge base in .json format
- TF-IDF matrices in .npz format 
- Entity id to ontology types dictionnary in json format

must have been generated. 

*Example:*

python new_types_collect_complete_score_spark1.py mentions_file.json 


## 3 - Graph mining for new score features and identification [Hadoop/Spark compatible]
 
KER-NEL (Knowledge graph exploration and node ranking for named entity linking) uses the knowledge graph to extract semantic information for a given entity node.  
Then, capitalizing on a filtering method to take as input a reduced set of entity candidates, it returns the most probable underlying entity to a given query. 
We added an example of regression/classification training using these new entity features.

To extract score features from, run
python collect_spark-explore_from_nicknames_collect.py ontology_type

*Example:*

python collect_spark-explore_from_nicknames_collect.py City


# Detailed comments

For transparency and reproducibility:

## a - Preprocessing and filtering

In our experiments on NIST TAC KBP datasets, TF-IDF matrix is computed on a subpart of the knowledge base, then applied on each mention dataset. 
A mention index represent the corresponding row in TF-IDF matrices. These indexes have been computed on all mentions datasets (including NIL mentions)

Fine-grained ontology classification is achieved by joining DBPedia 2016. Some titles have changed between 2009 and 2016. For a list of 15 entities , we manually annoted their ontology type with preprocessing/update-ontology.py 

**Ontology types**

We refer to the ontology tree:
http://mappings.dbpedia.org/server/ontology/classes/

For our experiments, ontology types are children of nodes Person, Organization and Place in the ontology tree, but we did not consider hierarchical relations between these types.
We check these new ontology types are compatible with NIST Datasets annotations (PER, ORG, GPE, in entityIdToOntologyType.py). 

## b - Graph based scores extraction

The most important idea of our algorithm is the extraction of scores in the graph neighborhood of one entity node.
We present the parameters and neighborhood score computation in details.

**Type mapping function**

We considered a constant mapping function.
The type mapping function we used is equal to 1 for types 
- "City", "Settlement", "Company", "University", "OfficeHolder"


As mentioned in our paper, we considered these types because they were most concerned by mis-identification, and enough training data were available for these types in TAC10-TRAIN and TAC14-TRAIN

The corresponding types mapping are the following On the following entity types : 
- "AdministrativeRegion", "Country", "RadioStation", "Road", "OfficeHolder", "MusicalArtist", "School", "BaseballPlayer", "MilitaryPerson", "Settlement", "Company", "University", "Building", "SoccerPlayer", "IceHockeyPlayer", "AmericanFootballPlayer", "Wrestler", "Politician", "Congressman", "Band"
- On other ontology types, our type mapping function value is 0.


**Neighborhood definition**

A neighborhood of a node entity is defined as his direct neighbors.

We made an exception consistent with our type mapping function.
Indeed, for node entities for which ontology type is a city, and if there is no country in their neighborhood, we computed shortest path between countries nodes and the entity node and kept the closest node. (We first built the list of all countries nodes, so that it avoids to re-compute Breadth first search).


**Neighbor score computation**

In our experiments, for each query, we built a matrix EC containing cosine similarity scores between neighbors entity and a set of arbitrary "informative" entities (cf next subsection).
EC has shape (N x E).
Then, we computed the TF-IDF cosine similarity of the query context with informative entities in a vector M, shape (E x 1). 
The final scores vector is computed with euclidian norm difference between vector M and column vectors of matrix EC.

However, this returns a variable size vector N, depending on the degree of the entity node.
To set a constant size we consider the following procedure:

- order scores using type mapping function (cf subsection Type mapping function) 
- if there are multiple neighbors for one type, compute minimum over the scores for these type
- if there is no neighbor for one type, set 0.

This yields a vector of constant size T since we considered a constant type mapping function.
The final scores vector (size T+1) is composed of the filtering score of the entity, and new neighborhood scores.


**Set of informative entities**

This set of informative entities has been chosen through our experiments (called "real_ids" in node_ranking/collect_spark-explore_from_nicknames_collect.py).
It is composed of a list of entities:

- Entities of type "Settlement"
- Entities of type "AdministrativeRegion"
- A subset of Entities of type "Country"


Finally, all of these parameters can possibly be improved. (Informative set of entities, type mapping function, number of final candidates).
We did not achieve full parametere optimization, to avoid over-fitting on NIST Datasets (we tried a dozen of configuration), but kept a configuration that was relatively performant. 

**Supervised learning for named entity identification**

We tried basic classifiers and regression models. Using scikit-learn, we finally kept logistic regression (node-ranking/try_rank.py) with penalization parameter equal to 0.0001 and class_weight = "balanced".


## c - Spark compatibility

The following scripts :

- filtering/new_types_collect_complete_score_spark1.py 
- node-ranking/collect_spark-explore_from_nicknames_collect.py

Respectively entity filtering and graph-based scores extraction scripts are spark/hadoop-compatible.

To use spark, you should :

- uncomment spark import headers 
- comment CPU and uncomment Spark code blocks at the end of files

An example of pyspark command (using hadoop) :
spark-submit --master yarn --driver-memory 15g --num-executors 70 --executor-memory 5G --conf spark.local.dir=/home/usr/tmp collect_complete_score_spark1.py   


## d - TAC-KBP inconsistencies (main types)

We noticed on NIST TAC-KBP 2010 dataset 2 entities type in contradiction with Wikipedia dump used for this challenge.
For more details : mention with ID EL004107 with gold entity ID E0466642 which is presented as a person (PER) wheras it is a localization (GPE); and mention with ID EL004411 with gold entity ID E0793726 presented as a per- son (PER) wheras it is an organization (ORG). We considered mention annotation as the ground truth and run experiments accordingly, though these types are often replaced by fine-grained classification.

