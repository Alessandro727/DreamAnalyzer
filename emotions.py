import spacy
import json
import nltk
import re
from nltk import pos_tag
from nltk import tokenize
from nltk.tree import Tree
from nltk.parse.stanford import StanfordParser
from nltk.stem.wordnet import WordNetLemmatizer
from characters import *

"""
Script created for the extraction and encoding of emotions between characters within dream reports. 
To understand the rules of social aggression identification visit the following link: 
https://dreams.ucsc.edu/Coding/emotions.html
"""


lemmatizer = WordNetLemmatizer()


##########################
# Check if word is an adjective.
# Parameter:
# 	@item: list of list of a single string word. Ex. [['good']]
#	@tags: list of all words in the morphological tree with their label. Ex. [(good, JJ)]
# Return:
# 	True if the word is an adjective;
#	False otherwise.
##########################

def isJJ(item,tags):
	if isinstance(item, list):
		item = item[0]
	for x in tags:
		if item == x[0]:
			if(x[1] == 'JJ'):
				return True
	return False


##########################
# Check if word is an adverb.
# Parameter:
# 	@item: list of list of a single adverb.
#	@tags: list of all words in the morphological tree with their label. Ex. [(good, JJ)]
# Return:
# 	True if the word is an adverb;
#	False otherwise.
##########################

def isRB(item,tags):
	if isinstance(item, list):
		item = item[0]
	for x in tags:
		if item == x[0]:
			if(x[1] == 'RB'):
				return True
	return False

##########################
# Check if word is an name.
# Parameter:
# 	@item: list of list of a single name.
#	@tags: list of all words in the morphological tree with their label. Ex. [(good, JJ)]
# Return:
# 	True if the word is an name;
#	False otherwise.
##########################

def isNN(item,tags):
	if isinstance(item, list):
		item = item[0]
	for x in tags:
		if item == x[0]:
			if(x[1] == 'NN' or x[1] == 'NNS'):
				return True
	return False


##########################
# Check if word is a pronoun.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is a pronoun;
#	False otherwise.
##########################

def isPRP(item):
	tag = pos_tag([item])[0]
	if(tag[1] == 'PRP' or tag[1] == 'PRP$'):
		return True
	return False


##########################
# Check if word is 'each' or 'other'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is 'each' or 'other';
#	False otherwise.
##########################

def isEachOther(item):
	tag = pos_tag([item])[0]
	if(tag[0] == 'each' or tag[0] == 'other'):
		return True
	return False

##########################
# Check if word is verb.
# Parameter:
# 	@item: single string word. 
#	@tags: list of all words in the morphological tree with their label. Ex. [(have, VB)]
# Return:
# 	True if the word is a verb;
#	False otherwise.
##########################

def isVerb(item,tags):

	if not isinstance(item, list):
		tag = pos_tag([item])[0]
		for t in tags:
			if t[0] == tag[0]:
				if t[1] == 'VB' or t[1] == 'VBD' or t[1] == 'VBG' or t[1] == 'VBN' or t[1] == 'VBP' or t[1] == 'VBZ' or t[1] == 'MD':
					return True
	return False

##########################
# Take only words in the list that are in one of emotion's knowledge bases (anger.txt, sadness.txt,...).
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words in one of emotion's knowledge bases.
##########################

def inEmolex(words):
	res = []
	flag = False
	for x in words:
		if isinstance(x[0][0], list):
			x = x[0]
		for y in isHappiness([x])+isAnger([x])+isApprension([x])+isSadness([x])+isConfusion([x]): 
			if 'Emolex*' in y[0][0]:
				res.append(y)
				flag = True 
		if flag == False:
			res.append(x)

	return res

##########################
# Take only words in the list that are in anger.txt.
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words are in anger.txt.
##########################

def isAnger(words):
	result = []
	for word in words:
		flag = False
		file = open('anger.txt',encoding='utf-8')
		for line in file:
			for x in range(len(word[0])):
				if 'VBD' in word[0][x]:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'1Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'1Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'1Emolex*VBD']])
						flag = True
						break
				else:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'1Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'1Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'1Emolex*']])
						flag = True
						break
			if flag:
				break
		if not flag:
			result.append(word)
	return result


##########################
# Take only words in the list that are in apprension.txt.
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words are in apprension.txt.
##########################

def isApprension(words):
	result = []
	for word in words:
		flag = False
		file = open('apprension.txt',encoding='utf-8')
		for line in file:
			for x in range(len(word[0])):
				if 'VBD' in word[0][x]:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'4Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'4Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'4Emolex*VBD']])
						flag = True
						break
				else:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'4Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'4Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'4Emolex*']])
						flag = True
						break
			if flag:
				break
		if not flag:
			result.append(word)
	return result

##########################
# Take only words in the list that are in happiness.txt.
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words are in happiness.txt.
##########################

def isHappiness(words):
	result = []
	for word in words:
		flag = False
		file = open('happiness.txt',encoding='utf-8')
		for line in file:
			for x in range(len(word[0])):
				if 'VBD' in word[0][x]:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'5Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'5Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'5Emolex*VBD']])
						flag = True
						break
				else:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'5Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'5Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'5Emolex*']])
						flag = True
						break
			if flag:
				break
		if not flag:
			result.append(word)
	return result

##########################
# Take only words in the list that are in sadness.txt.
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words are in sadness.txt.
##########################

def isSadness(words):
	result = []
	for word in words:
		flag = False
		file = open('sadness.txt',encoding='utf-8')
		for line in file:
			for x in range(len(word[0])):
				if 'VBD' in word[0][x]:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'8Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'8Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'8Emolex*VBD']])
						flag = True
						break
				else:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'v'):
						result.append([[line+'8Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'n'):
						result.append([[line+'8Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\n')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\n')[0].lower(), 'r'):
						result.append([[line+'8Emolex*']])
						flag = True
						break
			if flag:
				break
		if not flag:
			result.append(word)
	return result

##########################
# Take only words in the list that are in confusion.txt.
# Parameter:
# 	@words: list of words. 
# Return:
# 	List of words are in confusion.txt.
##########################

def isConfusion(words):
	result = []
	for word in words:
		flag = False
		file = open('confusion.txt',encoding='utf-8')
		for line in file:
			for x in range(len(word[0])):
				if 'VBD' in word[0][x]:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'v'):
						result.append([[line.split('\t')[0]+'9Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'n'):
						result.append([[line.split('\t')[0]+'9Emolex*VBD']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'r'):
						result.append([[line.split('\t')[0]+'9Emolex*VBD']])
						flag = True
						break
				else:
					if lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'v') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'v'):
						result.append([[line.split('\t')[0]+'9Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'n') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'n'):
						result.append([[line.split('\t')[0]+'9Emolex*']])
						flag = True
						break
					elif lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == line.split('\t')[0] or lemmatizer.lemmatize(word[0][x].replace('VBD',''), 'r') == lemmatizer.lemmatize(line.split('\t')[0].lower(), 'r'):
						result.append([[line.split('\t')[0]+'9Emolex*']])
						flag = True
						break
			if flag:
				break
		if not flag:
			result.append(word)
	return result


##########################
# Method for identifying characters and groups of characters and also their associate emotion.
# Parameter:
# 	@subject: list of words extracted from the morphological tree.
#	@tags: list of all words in the morphological tree with their label. Ex. [(have, VB)]
# Return:
# 	list of lists of character/group of characters and emotions. 
##########################

def makeGroup(subject, tags):

	result = []
	subresult = []
	last = 0
	for i in range(len(subject)):
		cod = code([[subject[i]]])
		if cod != []:
			if subject[i] == '':
				last = 1 
			elif last == 0 and cod != []:
				subresult.append(cod)
				last = 1 
			elif subject[i] == 'and' and last == 1:
				cod = code([[subject[i+1]]])
				if cod != []:
					subresult.append(['and'])
					subresult.append(cod)
					subject[i+1] = '_'
			elif last == 1 and cod != []:
				result.append(subresult)
				subresult = []
				subresult.append(cod)
		elif isVerb(subject[i], tags) or isJJ(subject[i], tags) or 'VBD' in subject[i] or isRB(subject[i], tags):
			subresult.append(subject[i])
			# last = 0
		elif subject[i]=='and':
			subresult.append(subject[i])
		elif isNN(subject[i],tags):
			if 'Emolex*' in inEmolex([[[subject[i]]]])[0][0][0]:
				subresult.append(subject[i])
		elif isPRP(subject[i]):
			subresult.append(subject[i])
	result.append(subresult)

	return result

##########################
# Check if 'me' is in the list and change it with 'I D'
# Parameter:
# 	@items: list of words. 
# Return:
# 	
##########################

def checkMe(items):
	if items != [[]]:
		for x in items:
			if(x[0][0] == 'me'):
				x[0][0] = 'I D'


##########################
# Method for extracting names, adjectives, verbs and pronouns inside the morphological tree generated by the dream report.
# Parameters:
#	@tags: list of all words in the morphological tree with their label. Ex. [(have, VB)]
# Return:
# 	List of extracted elements
##########################

def getNodeFromTag(tag):

	res = []
	for item in tag:
		if item[1] == 'NN' or item[1] == 'NNS' or item[1] == 'JJ' or item[1] == 'PRP' or item[1] == 'PRP$' or item[1] == 'NNP' or item[1] == 'NNPS' or item[1] == 'RB':
			res.append(item[0])
		elif  item[1] == 'VBG' or item[1] == 'VBN' or item[1] == 'VBP':
			res.append(item[0])
		elif item[1] == 'VBD' or item[1] == 'VBZ' or item[1] == 'VB' or pos_tag('He '+item[0]+' me')[1][1] == 'VBD':
			if item[0] not in res:
				res.append(item[0]+'VBD')
	return res


##########################
# Take the last character from a list of characters.
# Parameter:
# 	@lista: list of characters;
# Return:
# 	The character extracted from 'lista' or '1ISA' if there is no character.
##########################

def takeLast(lista):
	split_list = []
	for i in range(len(lista)):
		if 'it ' in lista[i].lower() or 'its ' in lista[i].lower():
			split_list = lista[:i]
	if split_list != []:
		for k in reversed(split_list):
			if '1' in k:
				lista.remove(lista[i])
				return k
	return '1ISA'	


##########################
# Take the last male character from a list of characters.
# Parameter:
# 	@lista: list of characters;
# Return:
# 	The character extracted from 'lista' or '1MSA' if there is no male character.
##########################

def takeLastMan(lista):
	split_list = []
	for i in range(len(lista)):
		if 'he ' in lista[i].lower() or 'his ' in lista[i].lower() or 'him ' in lista[i].lower():
			split_list = lista[:i]
	if split_list != []:
		for k in reversed(split_list):
			if '1M' in k or '1I' in k:
				lista.remove(lista[i])
				return k
	return '1MSA'
		
##########################
# Take the last group of characters from a list of characters.
# Parameter:
# 	@lista: list of characters;
# Return:
# 	The group of characters extracted from 'lista' or '2ISA' if there is no group of characters.
##########################

def takeLastGroup(lista):
	split_list = []
	for i in range(len(lista)):
		if 'we ' in lista[i].lower() or 'us ' in lista[i].lower() or 'they ' in lista[i].lower():
			split_list = lista[:i]
	if split_list != []:
		for k in reversed(split_list):
			if '2' in k:
				lista.remove(lista[i])
				return k
	return '2ISA'

##########################
# Take the last female character from a list of characters.
# Parameter:
# 	@lista: list of characters;
# Return:
# 	The female character extracted from 'lista' or '1FSA' if there is no female character.
##########################

def takeLastFemale(lista):
	split_list = []
	for i in range(len(lista)):
		if 'she ' in lista[i].lower() or 'her ' in lista[i].lower():
			split_list = lista[:i]
	if split_list != []:
		for k in reversed(split_list):
			if '1F' in k or '1I' in k:
				lista.remove(lista[i])
				return k
	return '1FSA'

##########################
# Clean results and return only the codes.
# Parameter:
# 	@items: list of aggression with characters.
# Return:
# 	List of coded aggression.
##########################

def cleanCodes(items):
	res = []
	for item in items:
		emo = item.split('*')[0].split(' ')[0]
		if '(' in item and '* ' in item:
			char = item.split('* ')[1].split(' (')[1].replace('(','').replace(')','')
		elif '* ' in item:
			char = item.split('* ')[1].replace('I ','')
		else:
			char = item
		res.append(emo+' '+char)

	return res

##########################
# Removes element in a list.
# Parameter:
# 	@item: item to delete.
# Return:
# 	List without all 'item'.
##########################

def remove(item):
    final_list = []
    for num in item:
        if num not in final_list:
            final_list.append(num)
    return final_list


##########################
# Refine the list with the characters and their emotion.
# Parameter:
# 	@items: list of list of characters and emotions.
# Return:
# 	list of lists of character/group of characters with their associate emotion. 
##########################
def makeGroup2(items):
	if items != [[]]:
		group = []
		for i in range(len(items)):
			if (items[i] != '_'):
				if items[i][-1] == 'and' and len(items)<i-1:
					group.append(items[i]+items[i+1])
					items[i+1] = '_'
					items[i] = '_'
				group.append(items[i])
		return group
	return []


##########################
# Subdivides the characters and associates their emotions.
# Parameter:
# 	@items: list of list of characters/group of characters and emotions.
#	@tags: list of all verbs in the morphological tree with their label. Ex. [(have, VB)]
# Return:
# 	list of lists of character/group of characters with their associate emotion in a structured way. 
##########################

def splitSubjVerb(items,tags):
	res = []
	items2 = sum(items, [])

	for item in items:
		new = []
		good = [x for x in item if  (isVerb(x,tags) or isJJ(x, tags) or isRB(x, tags) or isNN(x, tags) or 'VBD' in x) and 'Emolex*' in inEmolex([[[x]]])[0][0][0]]
		bad  = [x for x in item if not isVerb(x,tags) and not isRB(x, tags)  and not isJJ(x, tags) and not 'VBD' in x and not isNN(x, tags)]

		temp1 = []
		temp2 = []
		temp3 = []

		if (good != []):

			for y in item:
				if y in bad:
					temp1.append(y)
					item.remove(y)
				elif y in good:
					break
			for k in item:
				if k in good:
					temp2.append([k])
					item.remove(k)
				elif k in bad:
					break
			for p in item:
				if p in bad:
					temp3.append(p)
					item.remove(p)
				elif p in good:
					break
			new.append(temp1)
			new.append(temp2)
			new.append(temp3)

		else:
			new.append(bad)
		for j in new:
			if j == []:
				new.remove(j)

		res.append(new)
	return sum(res,[])

##########################
# Check if there is the pattern 'each other'.
# Parameter:
# 	@items: list of list of list of words.
# Return:
# 	The same list without 'each other' but with the associated characters. 
##########################

def checkEachOther(items):
	if [] not in items and [[]] not in items:
		res = []
		first = 0
		second = 0
		for i in range(len(items)):
			if(items[i][0][0] == 'each'):
				first = 1;
			if(items[i][0][0] == 'other' and first == 1):
				items.remove([['each']])
				items.remove([['other']])
				second = 1
		if (second == 1):
			if('and' in items[0]):
				items[0].remove('and')
				first_char = items[0][0:(int(len(items)/2))]
				second_char = items[0][(int(len(items)/2)):len(items)]
				items = first_char+items[1]+second_char
		res.append(items)
		return res
	return []


##########################
# Identify the type of emotion.
# Parameter:
# 	@items: list of words.
# Return:
# 	The same list with coded emotions. 
##########################

def identifyEmotion(items):
	result = []

	emo = inEmolex(items[0])

	for x in emo:
		if 'Emolex*' not in x[0][0]:
			result.append(x)
		else:
			values = x[0][0].replace('\n','')
			if '4' in values:
				if 'VBD' in x[0][0]:
					result.append([['APVBD'+' '+values]])
				else:
					result.append([['AP'+' '+values]])
			elif '8' in values:
				if 'VBD' in x[0][0]:
					result.append([['SDVBD'+' '+values]])
				else:
					result.append([['SD'+' '+values]])
			elif '5' in values:
				if 'VBD' in x[0][0]:
					result.append([['HAVBD'+' '+values]])
				else:
					result.append([['HA'+' '+values]])
			elif '1' in values:
				if 'VBD' in x[0][0]:
					result.append([['ANVBD'+' '+values]])
				else:
					result.append([['AN'+' '+values]])
			elif '9' or '6' in values:
				if 'VBD' in x[0][0]:
					result.append([['COVBD'+' '+values]])
				else:
					result.append([['CO'+' '+values]])
		
	return result

##########################
# It assigns the emotion to the respective character based on the syntax of the emotion. Ex. Andrea embarrassed me. -> AP D,
# Andrea felt embarrassed. -> AP 1MKA, I was embarrassed. -> AP D (It is therefore not always valid first character pattern and then emotions)
# Parameter:
# 	@code: list of coded groups (character, emotion).
# Return:
# 	List of coded gorups (character, emotion). 
##########################

def filterCode2(code):
	res = []
	if code == []:
		return []
	for item in code:
		if len(item)==2:
			temp = []
			for y in item:
				temp.append(y.replace('VBD','').replace('I D','D'))
			res.append(' '.join(reversed(temp)))
		elif len(item)==3:
			temp = []
			if 'VBD' in item[1]:
				temp.append(item[2].replace('I D','D'))
			else:
				temp.append(item[0].replace('I D','D'))
			temp.append(item[1].replace('VBD',''))
			res.append(' '.join(reversed(temp)))
			
	return res



##########################
# It takes only the groups (character, emotion) in which the emotion has been encoded and identified.
# Parameter:
# 	@code: list of coded groups (character, emotion).
# Return:
# 	List of coded gorups (character, emotion). 
##########################

def filterCode(code):
	res = []
	temp = []
	for i in range(len(code)):
		if 'VBD' in code[i][0][0]: 
			if i < len(code)-1:
				temp.append(code[i+1][0][0])
				temp.append(code[i][0][0])
				res.append(temp)
				temp = []
		elif(('CO' in code[i][0][0] or 'SD' in code[i][0][0] or 'HA' in code[i][0][0] or 'AN' in code[i][0][0] or 'AP' in code[i][0][0])):
			if i != 0:
				temp.append(code[i-1][0][0])
				temp.append(code[i][0][0])
				res.append(temp)
				temp = []
	flag = True
	new_res = []
	for x in res:
		for y in x:
			if ' ' in y:
				flag = False
		if not flag:
			new_res.append(x)
		flag = True

	return new_res


##########################
# Codes characters following rules in https://dreams.ucsc.edu/Coding/characters.html and in https://dreams.ucsc.edu/Coding/aggression.html
# Parameter:
# 	@characters: list of possible characters. Ex. [['guy'],[Paolo]]
# Return:
# 	list of coded characters.
##########################

def code(characters):
	result = []
	last = ''
	p = inflect.engine()
	#print('>>><',characters)

	for items in characters:

		#print('>>>>>>>>>>>',items)

		if items == [''] or items == []:
			continue

		if (len(items)==1):
			if items[0] == 'I':
					result.append(items[0]+' D')
			elif isPRP(items[0]):
					result.append(items[0]+' PRP')
			elif CharacterFrequency(items[0].replace('NNP','').replace('nnp','')):
				if len(items[0]) == 1:
					return []
				elif (isEachOther(items[0])):
					result.append(items[0])
				else:
					item = items[0].replace('nnp','').replace('NNP','')
					p = inflect.engine()
					lowerword=item.lower()
					if(p.singular_noun(item)):
						lowerword = p.singular_noun(item).lower()
					if(isAnimal(item) and not isAPerson(item)):
						if(isPlural(item)):
							result.append(item+' (2ANI)')
						else:
							result.append(item+' (1ANI)')
					elif(isAPerson(item) and not p.singular_noun(item) or isEthnic(item)):
						string = ' (1'
						if(isPlural(item)):
							string = ' (2'
						if(isOccupational(item)):
							string = string+'I'
						elif(gender(item) == False): 
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else:
							string = string+'F'
						if(isOccupational(item)):
							string = string+'O'
						elif(isEthnic(item)):
							string = string+'E'
						elif(lowerword in dictionary):	
							string = string+dictionary[lowerword]
						else:
							string = string+'K'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
					elif(isAGroup(item)):
						string = ' (2I'
						if(lowerword in dictionary):	
							string = string+dictionary[lowerword]
						elif(isEthnic(item)):
							string = string+'E'
						elif(isOccupational(item)):
							string = string+'O'
						else:
							string = string+'K'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
					elif(isUndefined(item)):
						if(isPlural(item)):
							string = ' (2'
						else:
							string = ' (1'
						if(gender(item) == False):
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else: 
							string = string+'F'
						if(lowerword in dictionary):	
							string = string+dictionary[lowerword]
						elif(isEthnic(item)):
							string = string+'E'
						elif(isOccupational(item)):
							string = string+'O'
						else:
							string = string+'K'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
					elif(isImaginary(item)):
						if(isPlural(item)):
							string = ' (6'
						else:
							string = ' (5'
						if(gender(item) == False):
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else: 
							string = string+'F'
						if(lowerword in dictionary):	
							string = string+dictionary[lowerword]
						elif(isEthnic(item)):
							string = string+'E'
						elif(isOccupational(item)):
							string = string+'O'
						else:
							string = string+'K'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
					elif(isDead(item)):
						if(isPlural(item)):
							string = ' (4'
						else:
							string = ' (3'
						if(gender(item) == False):
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else: 
							string = string+'F'
						if(lowerword in dictionary):	
							string = string+dictionary[lowerword]
						elif(isEthnic(item)):
							string = string+'E'
						elif(isOccupational(item)):
							string = string+'O'
						else:
							string = string+'K'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
		else:
			for item in items:
				if type(item) is list:
					item = ''

				lowerword=item.lower().replace('nnp','')
					
				if(p.singular_noun(item)):
					lowerword = p.singular_noun(item).lower()
	
				if not (isAPerson(item.replace('NNP','').replace('nnp','')) or isAnimal(lowerword) or isUndefined(lowerword) or isAGroup(lowerword) or lowerword == 'and' or lowerword in dead_words or lowerword in imaginary_words or lowerword in changed_words or item == 'I' or code([[lowerword]]) != []) and not CharacterFrequency(lowerword):
					items.remove(item)

			for i in range(0,len(items)):
				if items[i] == []:
					items[i] = ''

			if 'and' in items and len(items)>2:
				if 'I' in items:
					items.remove('and')
					items.remove('I')
					if CharacterFrequency(items[0].replace('NNP','').replace('nnp','')):
						cod = code([[items[0]]])
						if cod != []:
							result.append(cod[0])
				else:
					string=' (2'
					if deadGroup(items):
						string = ' (4'
						string = string+defineGenderGroup(items)
						string = string+'K'
						string = string+'A)' 
						result.append((' '.join(items)+' '+'dead'+string).replace('NNP',''))
					elif imaginaryGroup(items):
						string = ' (6'
						string = string+defineGenderGroup(items)
						string = string+'K' 
						string = string+'A)'
						result.append((' '.join(items)+' '+'imaginary'+string).replace('NNP',''))
					else:
						items.remove('and')
						string = string+defineGenderGroup(items)
						string = string+'K' 
						string = string+'A)' 
						result.append((' '.join(items)+' '+string).replace('NNP',''))
			elif('and' in items and len(items)>1):
				items.remove('and')
				if CharacterFrequency(items[0].replace('NNP','').replace('nnp','')):
					cod = code([[items[0]]])
					if cod != []:
						result.append(cod[0])

			else:	
				for k in items:
					if type(k) is list:
						k = ''
					if(makeSingularAndLower(k) in dead_words):
						if (isPlural(k)):
							string = ' (4'
						else:
							string = ' (3'
						sub = items[0]
						items.remove(k)
						if(isOccupational(sub)):
							string = string+'I'
						elif(gender(sub) == False): 
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else:
							string = string+'F'
						string = string+'K' #o U dipende se il gruppo diviso da and è conosciuto o no o forse S se sono stranieri
						string = string+'A)' #o C in caso Madre e fratello piccolo?
						result.append((' '.join(items)+' '+k+string).replace('NNP',''))
					if(makeSingularAndLower(k) in imaginary_words):
						if (isPlural(k)):
							string = ' (6'
						else:
							string = ' (5'
						sub = items[0]
						items.remove(k)
						
						if(isOccupational(sub)):
							string = string+'I'
						elif(gender(sub) == False): 
							string = string+'M'
						elif(gender(item) == 15):
							string = string+'I'
						else:
							string = string+'F'
						string = string+'K' #o U dipende se il gruppo diviso da and è conosciuto o no o forse S se sono stranieri
						string = string+'A)' #o C in caso Madre e fratello piccolo?
						result.append((' '.join(items)+' '+k+string))
					if(makeSingularAndLower(k) in changed_words):
						items.remove(k)
						form1 = code([[items[0]]])
						if '1' in form1[0]:
							res1 = form1[0].replace('1','7')
						else:
							res1 = form1[0].replace('2','7')
						result.append(res1)
						try:
							form2 = code([[items[1]]])
							if '1' in form1[0]:
								res2 =  form2[0].replace('1','8')
							else:
								res2 = form2[0].replace('2','8')
							result.append(res2)
						except:
							pass
	return result


##########################
# Extraction and encoding of characters within dream reports 
# Parameter:
# 	@sent: sentence.
# Return:
# 	list of clean coded characters.
##########################

def character_code(sent, stanford_parser):
	sent.replace(',','and')
	senteces = tokenize.sent_tokenize(sent)
	result = []
	for s in senteces:

		#stanford_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
		parse = stanford_parser.raw_parse(s)

		tree = list(parse)

		#tree[0].draw()

		subject = []
		subject= getNodes(tree,subject)
		temp = subject[:]
		temp2 = subject[:]


		subj2 = makeGroupAndFindDeadCharacters(temp)


		subj3 = makeGroupAndFindImaginaryCharacters(temp2)

		finalSubj = []
		for k in subj3:
			if len(k)>1:
				for f in k:
					if [f] in subj2:
						subj2.remove([f])
				subj2.append(k)


		subj4 = makeGroupAndFindChangedFormCharacters(subject)

		for k in subj4:
			if len(k)>1:
				for f in k:
					if [f] in subj2:
						subj2.remove([f])
				subj2.append(k)
		

		subj2 = removeDuplicates(subj2)
		result.append(code(subj2))
		if [] in result:
			result.remove([])
		for x in result:
			for z in x:
				if z == 'I D':
					x.remove('I D')
		
	result = removeDuplicatesAndCreateListOfArrayResult(result)
	result = removeCodedCharacter(result)
	result = cleanResults(result)
	return result

##########################
# Extraction and encoding of emotions within dream reports 
# Parameter:
# 	@sent: dream report.
# Return:
# 	list of clean coded emotions with characters.
##########################


def take_emotions(text, stanford_parser, nlp):

	result = []
	final_code = []
	#nlp = spacy.load('en_core_web_sm')
	doc=nlp(text)
	text = text.replace(',','').replace('!','.').replace("I'm","I am").replace("you're","you are").replace("You're","You are").replace("we're","we are").replace("We're","We are").replace("they're","they are").replace("They're","They are").replace('(','').replace(')','').replace("He's","He is").replace("he's","he is").replace("She's","She is").replace("she's","she is").replace("It's","It is").replace("it's","it is").replace("'"," ").replace("\""," ").replace(';','').replace(':','')

	senteces = tokenize.sent_tokenize(text)
	characters = []

	for s in senteces:
		sent_result = []
		result = []

		if 'feel' not in s.split() and 'felt' not in s.split():
			if 'better' in s.split():
				s = s.replace('better','')
			if 'well' in s.split():
				s = s.replace('well','')
			if 'helpless' in s.split():
				s = s.replace('helpless','')
		#tree[0].draw()

		tags = pos_tag(s.split())
		
		l = 2
		new_tags = []
		for x in tags:
			if l == 0 or l == 1:
				l = l+1
			elif x[0] == 'going' or x[0] == 'doing':
				l = 0
			elif x[0] == 'get':
				l= 2
			elif l == 2:
				new_tags.append(x)

		subject = []
		subject= getNodeFromTag(new_tags)

		for t in range(len(subject)):

			subject[t] = subject[t].replace('.','')

		characters = characters + character_code(s, stanford_parser)

		subject2 = subject[:]

		if subject2 != []:
			items = pos_tag(subject2)

			sent_result = makeGroup(subject,items)
			verb = []

			for item in items:
				if item[1] == 'VB' or item[1] == 'VBD' or item[1] == 'VBG' or item[1] == 'VBN' or item[1] == 'VBP' or item[1] == 'VBZ' or item[1] == 'MD':
					verb.append(item[0])

			result.append(sent_result)

			for item in result:
				checkMe(item)

			result = makeGroup2(result[0])

			result = list(filter(('_').__ne__, result))

			result = splitSubjVerb(result,items)

			for f in result:
				if f == []:
					result.remove(f)

			result = checkEachOther(result)
			code = []
			if result != []:
				code = identifyEmotion(result)

			for el in code:
				for i in range(len(el[0])):
					if 'we ' in el[0][i].lower() or 'us ' in el[0][i].lower():
						el[0][i] = takeLastGroup(characters)
					if 'it ' in el[0][i].lower() or 'its ' in el[0][i].lower():
						el[0][i] = takeLast(characters)
					if 'myself ' in el[0][i].lower() or 'me ' in el[0][i].lower() or 'my ' in el[0][i].lower():
						el[0][i] = 'I D'
					if 'they ' in el[0][i].lower() or 'them ' in el[0][i].lower():
						el[0][i] = takeLastGroup(characters)
					if 'she ' in el[0][i].lower() or 'her ' in el[0][i].lower():
						el[0][i] = takeLastFemale(characters)
					if 'he ' in el[0][i].lower() or 'his ' in el[0][i].lower() or 'him ' in el[0][i].lower():
						el[0][i] = takeLastMan(characters)
					if 'myself ' in el[0][i].lower() or 'me ' in el[0][i].lower() or 'my ' in el[0][i].lower():
						el[0][i] = 'I D'
					if 'they ' in el[0][i].lower() or 'them ' in el[0][i].lower() or 'their ' in el[0][i].lower():
						el[0][i] = takeLastGroup(characters)
					if 'she ' in el[0][i].lower() or 'her ' in el[0][i].lower():
						el[0][i] = takeLastFemale(characters)
					if 'he ' in el[0][i].lower() or 'his ' in el[0][i].lower() or 'him ' in el[0][i].lower():
						el[0][i] = takeLastMan(characters)
					
			code = filterCode(code)
			code = filterCode2(code)
			if code != []:
				for item in code:
					if (item.count('CO ')+item.count('SD ')+item.count('HA ')+item.count('AN ')+item.count('AP ')) == 1:
						final_code.append(item)
		
	final_code = remove(final_code)

	return cleanCodes(final_code)

# print(take_emotions("I was angry."))

