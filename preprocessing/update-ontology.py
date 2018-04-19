import json


entityIdToOntologyType = json.load(open("entityIdToOntologyType-Updated-10-04-18.json", "r"))



#updateIds = ['E0609298', 'E0694758', 'E0446101', 'E0165696', 'E0496876', 'E0645236', 'E0707513', 'E0745616', 'E0499632', 'E0678761', 'E0243349']
#updateOntologyTypes = ["Company", "PoliticalParty","ORG", "MilitaryUnit",  "Newspaper",  "PoliticalParty", "ORG", "SoccerClub", "TelevisionStation", "SportsTeam", "PoliticalParty"]


updateIds = ['E0199103', 'E0257593', 'E0593333', 'E0619026']
updateOntologyTypes = ["Country", "City", "Town", "Country"]

for i in range(len(updateIds)):
    assert(entityIdToOntologyType[updateIds[i]] == "GPE")
    entityIdToOntologyType[updateIds[i]] = updateOntologyTypes[i]

json.dump(entityIdToOntologyType, open("entityIdToOntologyType-Updated-11-04-18.json", "w"))









