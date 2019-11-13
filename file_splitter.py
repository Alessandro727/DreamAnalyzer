import sys
import random

#python file_splitter.py all_dreams.tsv ALL_OUT.txt

input_file = sys.argv[1]
existing_file = sys.argv[2]

print('splitting %s with existing ids in %s'%(input_file,existing_file))

existing_ids = set()

existing_entries_file = open(existing_file,'rt',encoding='utf-8')
for line in existing_entries_file:
	existing_ids.add(line.split('\t')[0])
existing_entries_file.close()
print('Already crawled %s entries'%(len(existing_ids)))

dreams_english_50lines = []
dreams_file = open(input_file,'rt',encoding='utf-8')
for line in dreams_file:
	tokens = line.split('\t')
	dream_id = tokens[0]
	words = len(tokens[-1].split(' '))
	language = tokens[-3]
	if len(line) > 10 and words >= 50 and language == 'en' and dream_id not in existing_ids:
		dreams_english_50lines.append(line)

print(len(dreams_english_50lines))
random.shuffle(dreams_english_50lines)

outfiles = []
for i in range(0,10):
	outfiles.append(open('split_dreams%s.txt'%i,'wt',encoding='utf-8'))
i=0
for line in dreams_english_50lines:
	outfiles[i%10].write(line+'\n')
	i += 1

for f in outfiles:
	f.close()


