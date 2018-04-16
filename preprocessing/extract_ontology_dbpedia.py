import pdb
import json


instance_file = open("instance_types_en.nt", "r")
index = -1


wiki_title_toType = {}
previous_wiki_title = ""
with open("instance_types_en.nt", "r") as instance_file:
    header = instance_file.readline()
    for line in instance_file:
        index += 1
        
        #<http://dbpedia.org/resource/Autism> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Disease>
        try:
            link_title, link_type, link_ontology, _ = line.strip().split()
        except:
            print("Line format error : " + str(index))
            continue
    
        wiki_title_ = link_title.split("/")[-1]
        wiki_title = wiki_title_[:len(wiki_title_)-1]
        #if wiki_title == "MILAN":
        #    pdb.set_trace()
        if index % 5000000 == 0:        
            print(wiki_title)
            print(index)

        if wiki_title == previous_wiki_title:
            previous_wiki_title = wiki_title
            continue
        else:
            if wiki_title.split("__")[0] == previous_wiki_title.split("__")[0]:
                previous_wiki_title = wiki_title
                continue
            else:
                try:
                    if "w3.org" in link_ontology:
                        ontology_type_ = link_ontology.split("owl#")[1]            
                        ontology_type = ontology_type_[:len(ontology_type_)-1]
                        wiki_title_toType[wiki_title] = ontology_type
                        previous_wiki_title = wiki_title
                        #print(link_ontology)
                    
                    else:
                        ontology_type_ = link_ontology.split("ontology/")[1]
                        ontology_type = ontology_type_[:len(ontology_type_)-1]
                        wiki_title_toType[wiki_title] = ontology_type
                        previous_wiki_title = wiki_title
                        #print(link_ontology)
                except:
                    pdb.set_trace()


print("writing...")
pdb.set_trace()
json.dump(wiki_title_toType, open("wiki_title_toType.json","w"))

