import difflib
import xlsxwriter
import random
import characters as ch
import sexuality as sex
import emotions as emot
import aggression as agg
import friendliness as fri 
import getFeatures as getFea
import time
import sys
import spacy
import traceback
from nltk.parse.stanford import StanfordParser

# python3 dream_analyzer.py split_dreams0.txt out_split_dreams0.txt err_split_dreams0.txt

input_file = sys.argv[1]
output_file = sys.argv[2]
error_file = sys.argv[3]
print('%s > %s, %s'%(input_file, output_file, error_file))

dreams_file = open(input_file,'rt',encoding='utf-8')
splitted_file = dreams_file.read().split('\n')
dream_report_list = []
id_dream_list = []
for line in splitted_file:
	#print(line)
	#words = int(line.split('\t')[-2])
	#language = line.split('\t')[-3]
	#if language == 'en':
	dream_report_list.append(line.split('\t')[-1]) #get text
	id_dream_list.append(line.split('\t')[0]) #get id
	#print(line.split('\t')[0])
	#print(line.split('\t')[-1])
dreams_file.close()

charList = []
emoList = []
aggList = []
friendList = []
sexList = []

results_file = open(output_file, 'wt', encoding='utf-8', buffering=1)
err_file = open(error_file, 'wt', encoding='utf-8', buffering=1)
results_file.write('id_dream\ttext_dream\tcharacters_code\temotions_code\taggression_code\tfriendliness_code\tsexuality_code\tmale_female_%\tanimal_%\tfamiliarity_%\tfriends_%\tfamily_%\tdead_imaginary_%\taggression_friendliness_%\tbe_friender_%\taggression_%\tvictimization_%\taggression_characters_index%\tfrienliness_characters_index\tsexuality_characters_index\tnegative_emotions_%\tconfusion_%\n')
stanford_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
nlp = spacy.load('en_core_web_sm')

for i in range(len(dream_report_list)):
	dream = dream_report_list[i]
	id_dream = id_dream_list[i]
	print('%s) id=%s' % (i, id_dream))
	try:
		start_time = time.time()
		characters_result = ch.character_code(dream, stanford_parser)
		for char in characters_result:
			try:
				charList.append(char.split(' (')[1].replace('(','').replace(')',''))
			except:
				charList.append(char)
		#print("--- %s seconds ---" % (time.time() - start_time))
		for emo in emot.take_emotions(dream, stanford_parser, nlp):
			if '(' in emo:
				emo = emo.split(' ')[0]+emo.split('(')[1].split(')')[0]
				emoList.append(emo)
			else:
				emoList.append(emo)
		#print("--- %s seconds ---" % (time.time() - start_time))
		for aggressive in agg.aggression_code(dream, stanford_parser, nlp):
			aggList.append(aggressive.replace('\t',' ').split(' ')[0]+'\t'+aggressive.split(' ')[1]+'\t'+aggressive.split(' ')[2])
		#print("--- %s seconds ---" % (time.time() - start_time))
		for friend in fri.friendliness_code(dream, stanford_parser, nlp):
			friendList.append(friend.split(' ')[0]+'\t'+friend.split(' ')[1]+'\t'+friend.split(' ')[2])
		#print("--- %s seconds ---" % (time.time() - start_time))
		for sexy in sex.sexuality_code(dream, stanford_parser, nlp):
			sexList.append(sexy.split(' ')[0]+'\t'+sexy.split(' ')[1]+'\t'+sexy.split(' ')[2])
		print("%s (%s seconds)" % (i, time.time() - start_time))

		try:
			Male_Female = getFea.maleFemalePercent(charList)
		except:
			Male_Female = 0
		try:
			Animal = getFea.animalPercent(charList)
		except:
			Animal = 0
		try:
			Familiarity = getFea.familiarityPercent(charList)
		except:
			Familiarity = 0
		try:
			Friends = getFea.friendsPercent(charList)
		except:
			Friends = 0
		try:
			Family = getFea.familyPercent(charList)
		except:
			Family = 0
		# file.write(groupPercent(charList))
		try:
			Dead_Imaginary = getFea.deadAndImaginaryPercent(charList)
		except:
			Dead_Imaginary = 0
		try:
			Aggression_Friendliness = getFea.aggressionFriendlinessPercent(aggList,friendList)
		except:
			Aggression_Friendliness = 0
		try:
			BeFriender = getFea.beFrienderPercent(friendList)
		except:
			BeFriender = 0
		try:
			Aggression = getFea.aggressionPercent(aggList)
		except:
			Aggression = 0
		try:
			Victimization = getFea.victimizationPercent(aggList)
		except:
			Victimization = 0
		try:
			A_CIndex = getFea.ACIndex(aggList,charList)
		except:
			A_CIndex = 0
		try:
			F_CIndex = getFea.FCIndex(friendList,charList)
		except:
			F_CIndex = 0
		try:
			S_CIndex = getFea.SCIndex(sexList,charList)
		except:
			S_CIndex = 0
		try:
			NegativeEmotions = getFea.negativeEmotionsPercent(emoList)
		except:
			NegativeEmotions = 0
		try:
			Confusion = getFea.confusionPercent(emoList)
		except:
			Confusion = 0

		results_file.write(id_dream+'\t')
		results_file.write(dream+'\t')
		results_file.write(str(charList).replace('[','').replace(']','').replace('\'',''))
		results_file.write('\t')
		results_file.write(str(emoList).replace('[','').replace(']','').replace('\'',''))
		results_file.write('\t')
		results_file.write(str(aggList).replace('[','').replace(']','').replace('\'','').replace('\\t',' '))
		results_file.write('\t')
		results_file.write(str(friendList).replace('[','').replace(']','').replace('\'','').replace('\\t',' '))
		results_file.write('\t')
		results_file.write(str(sexList).replace('[','').replace(']','').replace('\'','').replace('\\t',' '))
		results_file.write('\t')
		results_file.write(str(Male_Female)+'\t')
		results_file.write(str(Animal)+'\t')
		results_file.write(str(Familiarity)+'\t')
		results_file.write(str(Friends)+'\t')
		results_file.write(str(Family)+'\t')
		results_file.write(str(Dead_Imaginary)+'\t')
		results_file.write(str(Aggression_Friendliness)+'\t')
		results_file.write(str(BeFriender)+'\t')
		results_file.write(str(Aggression)+'\t')
		results_file.write(str(Victimization)+'\t')
		results_file.write(str(A_CIndex)+'\t')
		results_file.write(str(F_CIndex)+'\t')
		results_file.write(str(S_CIndex)+'\t')
		results_file.write(str(NegativeEmotions)+'\t')
		results_file.write(str(Confusion)+'\n')
		charList = []
		emoList = []
		aggList = []
		friendList = []
		sexList = []
	except Exception as e:
		print(e)
		print('error dream id %s'%id_dream)
		err_file.write('%s\n'%id_dream)
		traceback.print_exc()

err_file.close()
results_file.close()


