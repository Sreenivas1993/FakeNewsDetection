# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:06:36 2017

@author: shalin
"""
import spacy
from spacy.matcher import Matcher
from spacy.attrs import ORTH
import json
import nltk
import pandas as pd
from tqdm import tqdm,trange
from textblob import TextBlob 
from gensim.models import ldamodel
from gensim import corpora
from nltk.stem.porter import PorterStemmer
import numpy as np
import http.client,urllib.parse,json
import bs4 as bs
from glob import glob
import re
import urllib.request
p_stem = PorterStemmer()

subscriptionKey = "893f8cf0e542499790e274a511d43d76"
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/search"
    

parse = spacy.load('en_core_web_md')


def mod(num):
    if num >= 0:
        return num
    else:
        return num*-1
    

def jaccardSimilarity(query,doc):
    intersect_set = set(query).intersection(set(doc))
    union_set = set(query).union(set(doc))
    if len(union_set) != 0:
        jaccard = len(intersect_set)/len(union_set)
    return jaccard


def BingWebSearch(search):
        
    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    conn = http.client.HTTPSConnection(host)
    query = urllib.parse.quote(search)
    conn.request("GET", path + "?q=" + query, headers=headers)
    response = conn.getresponse()
    headers = [k + ": " + v for (k, v) in response.getheaders()
                   if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
    return headers, response.read().decode("utf8")


def extractFeatures(title,body):
    
    if title == '' or body == '':
        print("Invalid Tweets....")
        return False
    
    else:
        trainDataFrame = pd.DataFrame()
        trainDataList = [] 
        
        body = body
        title = title
        parsed_body = parse(body)
        body_nouns = [x.text for x in parsed_body if x.pos_ == 'NOUN' or x.pos_ == 'PROPN']
        noun_len = len(body_nouns)
        body_verbs = [x.text for x in parsed_body if x.pos_ == 'VERB']
        verb_len = len(body_verbs)
        appData = pd.Series([body,title,noun_len,verb_len])
        trainDataList.append(appData)
        trainDataFrame = pd.DataFrame(trainDataList)
        trainDataFrame.columns = ['Body','Title','Body_Noun_No','Body_Verb_No']
    
        title_token_count = []
        title = title
        doc_title = parse(title)
        count = len(doc_title.count_by(ORTH))
        title_token_count.append(count)
        trainDataFrame['n_tokens_title'] = title_token_count

        body_token_count = []
        body = body
        count = len(nltk.word_tokenize(body))
        body_token_count.append(count)
        trainDataFrame['n_tokens_content'] = body_token_count

        n_unique_tokens = []
        body = body
        doc_body = parse(body)
        u_count = len(doc_body.count_by(ORTH))
        u_count_ratio = u_count/count
        n_unique_tokens.append(u_count_ratio)
        trainDataFrame['n_unique_tokens'] = n_unique_tokens

        n_non_stop_words = []
        n_non_stop_unique_tokens = []
        stopwords = nltk.corpus.stopwords.words('english')
        body = body
        count = len(nltk.word_tokenize(body)) + 1
        non_stop_word_count = 0
        for token in nltk.word_tokenize(body):
            if token not in stopwords:
                non_stop_word_count+=1
        rat = non_stop_word_count/count
        n_non_stop_words.append(rat)
    
        doc = parse(body)
        u_non_s_w_count = 0
        for token in doc:
            if token.is_stop == False:
                u_non_s_w_count+=1
        rat = u_non_s_w_count/count
        n_non_stop_unique_tokens.append(rat)    
        trainDataFrame['n_non_stop_words'] = n_non_stop_words
        trainDataFrame['n_non_stop_unique_tokens'] = n_non_stop_unique_tokens

        average_token_length = []
        body = body
        count = len(nltk.word_tokenize(body)) + 1
        sum = 0
        for word in nltk.word_tokenize(body):
            sum += len(word)
        avg = sum/count
        average_token_length.append(avg)
        trainDataFrame['average_token_length'] = average_token_length


        #LDA - Latent Dirichlet Allocation using gensim. No of topics are 2 for now....

        ldamodel1 = ldamodel.LdaModel.load("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/ldamodel.bin") 
        
        corpus = []
        corpus.append(str(body))    
        token_corpus = []
        for doc in tqdm(corpus):
            raw = doc.lower()
            tokens = nltk.word_tokenize(raw)
            tokens_ws = []
            for t in tokens:
                if t not in stopwords:
                    tokens_ws.append(t)
            token_stem = []
            for t in tokens_ws:
                token_stem.append(p_stem.stem(t))        
            token_corpus.append(token_stem)    

        dictionary = corpora.Dictionary(token_corpus)
        bow = [dictionary.doc2bow(doc) for doc in token_corpus]

        lda = []
        for doc in tqdm(bow):
            l = ldamodel1[doc]
            ldadata = {}
            if len(l) == 2:
                for tup in l:
                    ldadata[tup[0]] = tup[1]    
                lda.append(ldadata)
            else:
                for i in trange(1):
                    tup = l[0]
                    if tup[0] == 0:
                        l.append((1,0.0000))
                    else:
                        l.append((0,0.0000))

                for tup in l:
                    ldadata[tup[0]] = tup[1]    
                lda.append(ldadata)
                
        lda_dataframe = pd.DataFrame(lda)
        lda_dataframe.columns = ['LDA_Topic_00','LDA_Topic_01']
        templist = ['LDA_Topic_00','LDA_Topic_01']
        for t in templist:
            trainDataFrame[t] = pd.Series(lda_dataframe[t])
       
    
        
        text_sentiment_polarity = []
        body = body
        blob = TextBlob(str(body))
        text_sentiment_polarity.append(blob.sentiment.polarity)        
        trainDataFrame['text_sentiment_polarity'] = text_sentiment_polarity


        text_subjectivity = []
        body = body
        blob = TextBlob(str(body))
        text_subjectivity.append(blob.sentiment.subjectivity)        
        trainDataFrame['text_subjectivity'] = text_subjectivity


        rate_of_positive_words = []
        rate_of_negative_words = []
        rate_of_positive_words_nn = []
        rate_of_negative_words_nn = []
        avg_pos = []
        min_pos = []
        max_pos = []
        avg_neg = []
        min_neg = []
        max_neg = []
    
        n_pos=1
        n_neg=1
        n_neu=1
        count=1
        sum_pos_polarity=0
        sum_neg_polarity=0
        max_pos_polarity=[0.0000]
        min_pos_polarity=[99.0000]
        max_neg_polarity=[-99.000]
        min_neg_polarity=[0.0000]
    
        body_tokens = nltk.word_tokenize(str(body))
        count = len(body_tokens)+1
        for word in  body_tokens:
            blob = TextBlob(word)
            pol = blob.sentiment.polarity
            if pol > 0:
                sum_pos_polarity+=pol
                max_pos_polarity.append(pol)
                min_pos_polarity.append(pol)
                n_pos+=1
            if pol < 0:
                sum_neg_polarity+=pol
                max_neg_polarity.append(pol)
                min_neg_polarity.append(pol)
                n_neg+=1
            else:
                n_neu+=1
    
        no_nn = (count-n_neu) + 1 
        r_pos = n_pos/count
        r_neg = n_neg/count
        r_pos_non_neu = n_pos/no_nn
        r_neg_non_neu = n_neg/no_nn
        avg_pos_polarity = sum_pos_polarity/n_pos
        avg_neg_polarity = sum_neg_polarity/n_neg
        max_pos_pol_index = np.argmax(max_pos_polarity)
        max_pos_polarity1 = max_pos_polarity[max_pos_pol_index]
        min_pos_pol_index = np.argmin(min_pos_polarity)
        min_pos_polarity1 = min_pos_polarity[min_pos_pol_index]
        max_neg_pol_index = np.argmax(max_neg_polarity)
        max_neg_polarity1 = max_neg_polarity[max_neg_pol_index]
        min_neg_pol_index = np.argmin(min_neg_polarity)
        min_neg_polarity1 = min_neg_polarity[min_neg_pol_index]
    
        rate_of_positive_words.append(r_pos)
        rate_of_negative_words.append(r_neg)
        rate_of_positive_words_nn.append(r_pos_non_neu)
        rate_of_negative_words_nn.append(r_neg_non_neu)
        avg_pos.append(avg_pos_polarity)
        max_pos.append(max_pos_polarity1)
        min_pos.append(min_pos_polarity1)
        avg_neg.append(avg_neg_polarity)
        max_neg.append(max_neg_polarity1)
        min_neg.append(min_neg_polarity1)

        trainDataFrame['rate_of_positive_words'] = rate_of_positive_words
        trainDataFrame['rate_of_negative_words'] = rate_of_negative_words
        trainDataFrame['rate_of_positive_words_nn'] = rate_of_positive_words_nn
        trainDataFrame['rate_of_negative_words_nn'] = rate_of_negative_words_nn
        trainDataFrame['avg_positive_polarity'] = avg_pos
        trainDataFrame['max_positive_polarity'] = max_pos
        trainDataFrame['min_positive_polarity'] = min_pos
        trainDataFrame['avg_negative_polarity'] = avg_neg
        trainDataFrame['max_negative_polarity'] = max_neg
        trainDataFrame['min_negative_polarity'] = min_neg
    
        title_subjectivity = []
        blob = TextBlob(str(title))
        title_subjectivity.append(blob.sentiment.subjectivity)        
        trainDataFrame['title_subjectivity'] = title_subjectivity

        title_sentiment_polarity = []
        blob = TextBlob(str(title))
        title_sentiment_polarity.append(blob.sentiment.polarity)        
        trainDataFrame['title_sentiment_polarity'] = title_sentiment_polarity

        title_abs_sentiment_polarity = []
        blob = TextBlob(str(title))
        title_abs_sentiment_polarity.append(mod(blob.sentiment.polarity))        
        trainDataFrame['title_abs_sentiment_polarity'] = title_abs_sentiment_polarity

        ari = []
        no_of_char = 0
        no_of_words = 0
        no_of_sents = 0
    
        no_of_words = len(nltk.word_tokenize(str(body)))
        no_of_sents = len(nltk.sent_tokenize(str(body)))
        for token in nltk.word_tokenize(str(body)):
            no_of_char += len(token)
        
        if no_of_sents != 0 and no_of_words != 0:
            doc_ari = 4.71*(no_of_char/no_of_words) + 0.5*(no_of_words/no_of_sents) - 21.43
        else:
            doc_ari = -21.43
        ari.append(doc_ari)
        trainDataFrame['ARI'] = ari    


        #ORS...    
        term_list = []
        term = str(title)
        term_list.append(term)
        if len(subscriptionKey) == 32:
            headers, result = BingWebSearch(term)
            with open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/streamBingResults/"+title.split(" ")[0]+"-tweet.json","w+",encoding='utf-8') as write:
                json.dump(result,write)
        else:
            print("Invalid Bing Search API subscription key!")

        
        path = "C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/streamBingResults/"+title.split(" ")[0]+"-tweet.json"
        files = glob(path)

        jsondata = []
        for file in files:
            jsondata.append(json.load(open(file,encoding='utf-8')))
    
        pattern = re.compile(r', "url": "(.+?)"')
        found = []
        for d in jsondata:
            arr = re.findall(pattern,d)
            f = []
            for a in arr:
                f.append(a.replace('\\',''))    
            found.append(f)

        articles = []
        for list_link in found:
            cont_list = []
            for link in list_link:
                try:
                    myLink = urllib.request.urlopen(link).read()
                    soup = bs.BeautifulSoup(myLink,'html5lib')
                    para = soup.find_all('p')
                    cont = ''
                    for p in para:
                        cont += str(p.text)
                    cont_list.append(cont)
                except:
                    print(link+" Not Opening...")
                    print()
            articles.append(cont_list)

        clean_articles=[]
        stopwords = nltk.corpus.stopwords.words()
        for l in articles:
            clean_article=[]
            for arti in l:
                word_token = nltk.word_tokenize(str(arti))
                ws_token=[]
                for token in word_token:
                    if token not in stopwords:
                        ws_token.append(token)
                clean_article.append(ws_token)
            clean_articles.append(clean_article)

        jaccard_similarity = []
        for i in range(len(clean_articles)):
            l = clean_articles[i]
            arti_js = []
            for arti in l:
                js = jaccardSimilarity(nltk.word_tokenize(term_list[i]),arti)
                arti_js.append(js)
            jaccard_similarity.append(np.mean(arti_js))    
   
        trainDataFrame['Jaccard_Similarity'] = jaccard_similarity    
    
        
        globalVector = json.load(open("C:/Users/shalin/Desktop/Fake_News_Detection/Stance_Detection/BaseLine/globalVectorTrain.json",encoding='utf-8'))
        probablityVector={}

        vector = {}
        text = body
        sentencesList = nltk.sent_tokenize(text)
        for line in sentencesList:
            result = ""
            re.sub(r'[^a-zA-Z ]+', '',line)
            word_tokens = nltk.word_tokenize(line)
            filtered_line = [w for w in word_tokens if not w in stopwords]
  
            result =  nltk.pos_tag(filtered_line)
  
            for obj in result:
                if obj[1].lower() not in vector:
                    vector[obj[1].lower()] = 1
                else:
                    vector[obj[1].lower()] += 1


        for key,value in vector.items():
            if key in globalVector.keys():
                probablityVector[key] = 1.0*value/(1.0*globalVector[key])
            else:
                probablityVector[key] = 0.000

        nn = []
        if 'nn' in probablityVector:
            nn.append(probablityVector['nn'])
        else:
            nn.append(0.0000)

        prp = []
        if 'prp' in probablityVector:
            prp.append(probablityVector['prp'])
        else:
            prp.append(0.0000)

        vb = [] 
        if 'vb' in probablityVector:
            vb.append(probablityVector['vb'])
        else:
            vb.append(0.0000)
            
        trainDataFrame['NounProbability'] = nn
        trainDataFrame['VerbProbability'] = vb
        trainDataFrame['PrepositionProbability'] = prp
        

        return trainDataFrame
    
