This repository is dedicated to for Graph based named of entities.

1 - Preprocessing module

This module is used to parse NIST TAC-KBP (2009 to 2014) datasets and build queries and knowledge base/graph dataset.


2 - Filtering method


3 - Graph mining for new score features and identification algorithm


These modules were conceived in as part of a Named entity linking problem but are relatively independant.



As a type mapping, we considered a constant mapping function.

The type mapping function we used is equal to 1 for type 'City' on the following entity types : 'AdministrativeRegion', 'Country', 'RadioStation', 'Road', 'OfficeHolder', 'MusicalArtist', 'School', 'BaseballPlayer', 'MilitaryPerson', 'Settlement', 'Company', 'University', 'Building', 'SoccerPlayer', 'IceHockeyPlayer', 'AmericanFootballPlayer', 'Wrestler', 'Politician', 'Congressman', 'Band', and 0 otherwise.

