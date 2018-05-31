# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:35:40 2017

@author: PHK
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus.reader import ChunkedCorpusReader
from collections import Counter
from nltk.corpus import stopwords
def temp(tagged):
    first,snd = zip(*tagged1)
path = '/users/PHK/python_file/panchatantra.txt'
op = open(path,'r')
sentence = op.read()
tokens = nltk.word_tokenize(sentence)
tokens
tagged = nltk.pos_tag(tokens)
tokens1 = [word for word in tokens if word not in stopwords.words('english')]
tagged[0:7]
#print(tokens)
counts = Counter(tag for word,tag in tagged)
counts
p = Counter(tokens)
path = '/users/PHK/python_file/new.txt'
op = open(path,'r')
sentence = op.read()
newstr = sentence.replace(".","")
tokens1= nltk.word_tokenize(sentence)
tokens1 = [word for word in tokens1 if word not in stopwords.words('english')]
tagged1 = nltk.pos_tag(tokens1)
entities= nltk.chunk.ne_chunk(tagged1)
print(entities)

counts1 = Counter(tag for word,tag in tagged1)
p1 = Counter(tokens1)
path = '/users/PHK/python_file/panchatantra2.txt'
op = open(path,'r')
sentence = op.read()
tokens2 = nltk.word_tokenize(sentence)
tagged2 = nltk.pos_tag(tokens2)
tokens2 = [word for word in tokens2 if word not in stopwords.words('english')]
counts2 = Counter(tag for word,tag in tagged2)
p2 = Counter(tokens2)
temp(tagged2)
path = '/users/PHK/python_file/panchatantra3.txt'
op = open(path,'r')
sentence = op.read()
tokens3 = nltk.word_tokenize(sentence)
tokens3 = [word for word in tokens3 if word not in stopwords.words('english')]
tagged3 = nltk.pos_tag(tokens3)
counts3 = Counter(tag for word,tag in tagged3)
p3 = Counter(tokens3)
path = '/users/PHK/python_file/panchatantra4.txt'
op = open(path,'r')
sentence = op.read()
tokens4 = nltk.word_tokenize(sentence)
tokens4 = [word for word in tokens4 if word not in stopwords.words('english')]
tagged4 = nltk.pos_tag(tokens4)
counts4 = Counter(tag for word,tag in tagged4)
p4 = Counter(tokens4)
path = '/users/PHK/python_file/panchatantra5.txt'
op = open(path,'r')
sentence = op.read()
tokens5 = nltk.word_tokenize(sentence)
tokens5 = [word for word in tokens5 if word not in stopwords.words('english')]
tagged5 = nltk.pos_tag(tokens5)
counts5 = Counter(tag for word,tag in tagged5)
p5 = Counter(tokens5)
#first,snd = zip(*tagged1)
new =[]


    panchatantra = BayesianModel([('wedge', 'monkey'),
('monkey', 'merchant'),
('merchant', 'mansons'),
('merchant', 'carpenters'),('carpenters','mansons'),('carpenters','garden')])#,('garden','work'),('garden','morning'),('garden','break'),('mansons','garden'),('monkey','workers'),('monkey','site'),('workers','meals'),('workers','carpenters')])
wedge = TabularCPD(
variable='wedge',
variable_card=2,
values=[[0.5, 0.5]])
#evidence=['monkey'],
#evidence_card=[2])
marchant = TabularCPD(
variable='marchant',
variable_card=2,
values=[[0.5, 0.5]])
#evidence=['monkey'],
#evidence_card=[2])
carpenters = TabularCPD(
variable='carpenters',
variable_card=2,
values=[[0.5, 0.5]])
#evidence=['wedge','merchant'],
#evidence_card=[3,3])
panchatantra.add_cpds(wedge,marchant,carpenters)

