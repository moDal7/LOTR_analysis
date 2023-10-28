from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

def sentiment_analysis(text): 
    prediction = dict()
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None) # type: ignore
    lines=0
    for line in text.split('\n'):
        lines+=1
        prediction_line=list()
        prediction_line=classifier(line)
        for elem in prediction_line[0]: # type: ignore
            if elem["label"] in set(prediction.keys()): # type: ignore
                prediction[elem["label"]]+=elem["score"] # type: ignore
            else:
                prediction[elem["label"]]=elem["score"] # type: ignore
    
    prediction={key:prediction[key]/lines for key in prediction.keys()}
    return prediction

def polarity(text):
    prediction = dict()
    classifier = pipeline("text-classification", model="Seethal/sentiment_analysis_generic_dataset", top_k=None) # type: ignore
    lines=0
    for line in text.split('\n'):
        lines+=1
        prediction_line=list()
        prediction_line=classifier(line)
        for elem in prediction_line[0]: # type: ignore
            if elem["label"] in set(prediction.keys()): # type: ignore
                prediction[elem["label"]]+=elem["score"] # type: ignore
            else:
                prediction[elem["label"]]=elem["score"] # type: ignore
    
    prediction={key:prediction[key]/lines for key in prediction.keys()}
    return prediction

def strongest_lines(text, top_k=10, feeling=None):
    strong_lines=list()
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None) # type: ignore
    for line in text.split('\n'):
        if len(line.split(' ')) < 3:
            continue
        prediction_line=classifier(line)
        scores=[]
        high_score = 0
        for elem in prediction_line[0]: # type: ignore
            scores.append(elem["score"]) # type: ignore
            if elem["score"] > high_score: # type: ignore
                strongest_feeling = elem
                high_score = elem["score"] # type: ignore
            
        if feeling is None:
            score_entropy=-sum([score*np.log(score) for score in scores])
            strong_lines.append([score_entropy, prediction_line[0], line]) # type: ignore
        
        else:
            if strongest_feeling["label"]==feeling: # type: ignore
                strong_lines.append([strongest_feeling["score"], prediction_line[0], line]) # type: ignore

    return sorted(strong_lines, key = lambda x: x[0], reverse=True)[:top_k]