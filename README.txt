Before starting dream_analyzer.py execute the following commands inside the folder:

1)	export CLASSPATH=~/stanford-postagger-full-2015-04-20/stanford-postagger.jar:~/stanford-ner-2015-04-20/stanford-ner.jar:~/stanford-parser-full-2015-04-20/stanford-parser.jar:~/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar

2)	export STANFORD_MODELS=~/stanford-postagger-full-2015-04-20/models:~/stanford-ner-2015-04-20/classifiers

3) insert dreams to be analyzed within the file 'dreams_to_analyze.txt' with format: id_dream	dream_report (one dream for line)

4) run dream_analyzer.py