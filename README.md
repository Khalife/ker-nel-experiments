This repository is dedicated to research experiments for **Graph based named of entities**.

These modules were conceived as part of a Named entity linking problem. There implementation here are relatively independant.

[Some of these modules use Spark, we are currently updating the code to provide a CPU version]

1 - **Preprocessing module**

This module is used to :  
- parse NIST TAC-KBP (2009 to 2014) datasets and build queries and knowledge base/graph dataset.
- provide a fine-grained ontology entity classification using DBPedia 2016


2 - **Filtering method** [Uses Spark, CPU-version in development]

This module provides a filtering algorithm to discard less similar entities to a given query. It allows to return a smaller set of entity candidates.

3 - **Graph mining for new score features and identification** [Uses Spark, CPU-version in development]  
KER-NEL (Knowledge graph exploration and node ranking for named entity linking) uses the knowledge graph to extract semantic information for a given entity node.  
Then, capitalizing on a filtering method to take as input a reduced set of entity candidates, it returns the most probable underlying entity to a given query. 
We added an example of regression/classification training using these new entity features. 


Comments

a - Type mapping function

We considered a constant mapping function.

The type mapping function we used is equal to 1 for type 'City' on the following entity types : 'AdministrativeRegion', 'Country', 'RadioStation', 'Road', 'OfficeHolder', 'MusicalArtist', 'School', 'BaseballPlayer', 'MilitaryPerson', 'Settlement', 'Company', 'University', 'Building', 'SoccerPlayer', 'IceHockeyPlayer', 'AmericanFootballPlayer', 'Wrestler', 'Politician', 'Congressman', 'Band', and 0 otherwise. 
We also used graph mining with the same mapping function for entity types : "Settlement", "Company", "University", "OfficeHolder", because these types were most concerned by mis-identification, and enough training data were available for these types in TAC10-TRAIN and TAC14-TRAIN

b - TAC-KBP inconsistencies 


