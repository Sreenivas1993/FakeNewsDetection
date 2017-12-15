import codecs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os.path
import json
from pprint import pprint

stop_words = set(stopwords.words('english'))
globalVector ={}
documentVector = list()

with codecs.open('stories.json', 'r', 'UTF-8') as data_file:    
    allDocuments = json.load(data_file)
    data_file.close()
    counter = 0
    for document in allDocuments:
        
        if counter == 3000:
            break
        counter += 1
        vector = {}
        text = document['content']
        sentencesList = text.split("  ")
        for line in sentencesList:
            result = ""
            re.sub(r'[^a-zA-Z ]+', '',line)
            word_tokens = word_tokenize(line)
            filtered_line = [w for w in word_tokens if not w in stop_words]
  
            result =  nltk.pos_tag(filtered_line)
  
            for obj in result:
                if obj[1].lower() not in vector:
                    vector[obj[1].lower()] = 1
                else:
                    vector[obj[1].lower()] += 1

                if obj[1].lower() not in globalVector:
                    globalVector[obj[1].lower()] = 1
                else:
                    globalVector[obj[1].lower()] += 1

        documentVector.append(vector)


globalProbVector = list()
for vector in documentVector:
    probablityVector={}

    for key,value in vector.items():
        if key in globalVector.keys():
            probablityVector[key] = 1.0*value/(1.0*globalVector[key])
    globalProbVector.append(probablityVector)        


with codecs.open('pos_tag.json', 'w+', 'UTF-8') as write_json:
    json.dump(globalProbVector,write_json)