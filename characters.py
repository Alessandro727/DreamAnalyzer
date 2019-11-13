import re
import spacy
from nltk.corpus import wordnet as wn
import inflect
import json
import nltk
import random
from nltk.corpus import names
from nltk import tokenize
from nltk.parse.stanford import StanfordParser
from nltk import pos_tag

"""
Script created for the extraction and encoding of characters within dream reports. 
To understand the rules of characters identification visit the following link: 
https://dreams.ucsc.edu/Coding/characters.html 
"""

p = inflect.engine()

dictionary = {'father': 'F', 'mother': 'M', 'brother': 'B', 'sister': 'S', 'husband': 'H', 'wife': 'W', 'son': 'A', 'daughter': 'D', 'child': 'C', 'infant': 'I', 'parent': 'K', 'grandmother': 'R', 'grandfather': 'R', 'aunt': 'R', 'uncle': 'R', 'nephew': 'R', 'niece': 'R', 'cousin': 'R', 'brother-in-law': 'R', 'sister-in-law': 'R', 'stepmother': 'R', 'stepfather': 'R', 'ex-husband': 'R', 'ex-wife': 'R', 'half-brother': 'R', 'half-sister': 'R', 'roommate': 'K', 'postman': 'K', 'man': 'K', 'policeman': 'K', 'boss': 'K', 'classmate': 'K', 'friend': 'K', 'buddy': 'K', 'dad': 'F', 'predecessor': 'K', 'ancestor': 'R', 'origin': 'K', 'mom': 'M', 'creator': 'M', 'relative': 'K', 'relation': 'K', 'kin': 'S', 'blood brother': 'K', 'kinsperson': 'S', 'partner': 'K', 'bride': 'W', 'spouse': 'K', 'companion': 'H', 'consort': 'H', 'groom': 'H', 'boy': 'A', 'heir': 'A', 'offspring': 'K', 'descendant': 'A', 'dependent': 'A', 'woman': 'D', 'female offspring': 'D', 'male offspring': 'A', 'toddler': 'I', 'adolescent': 'C', 'juvenile': 'C', 'youth': 'C', 'newborn': 'I', 'kid': 'I', 'suckling': 'I', 'originator': 'X', 'prototype': 'X', 'matriarch': 'R', 'dowager': 'R', 'granny': 'R', 'grandma': 'R', 'patriarch': 'R', 'forefather': 'R', 'elder': 'R', 'sibling': 'R', 'fellow': 'K'}
undefinedList = ['character', 'leader', 'official', 'star', 'VIP', 'bigwig', 'celebrity', 'eminence', 'lion', 'luminary', 'nabob', 'notability', 'notable', 'personage', 'VIP', 'ace', 'big cheese', 'big deal', 'big gun', 'big name', 'big shot', 'big stuff', 'bigwig', 'celeb', 'cynosure', 'famous person', 'figure', 'heavyweight', 'hero', 'hotshot', 'immortal', 'lion', 'luminary', 'magnate', 'mahatma', 'major leaguer', 'name', 'notable', 'personage', 'personality', 'somebody', 'someone', 'star', 'superstar', 'the cheese', 'worthy', 'VIP', 'ace', 'big cheese', 'big deal', 'big gun', 'big name', 'big shot', 'big stuff', 'bigwig', 'celeb', 'cynosure', 'famous person', 'figure', 'heavyweight', 'hero', 'hotshot', 'immortal', 'lion', 'luminary', 'magnate', 'mahatma', 'major leaguer', 'name', 'notable', 'personage', 'personality', 'somebody', 'someone', 'star', 'superstar', 'the cheese', 'worthy', 'being', 'body', 'character', 'creature', 'human', 'man', 'mortal', 'part', 'person', 'personage', 'somebody', 'someone', 'woman', 'being', 'body', 'character', 'creature', 'human', 'man', 'mortal', 'part', 'person', 'personage', 'somebody', 'someone', 'woman', 'VIP', 'celebrity', 'dignitary', 'heavyweight', 'household name', 'luminary', 'name', 'notable', 'person of note', 'personage', 'public figure', 'so-and-so', 'some person', 'someone', 'star', 'superstar', 'whoever', 'celebrity', 'draw', 'favorite', 'headliner', 'hero', 'idol', 'lead', 'leading role', 'luminary', 'name', 'somebody', 'someone', 'starlet', 'superstar', 'topliner']
dead_words = ['body', 'cadaver', 'remains', 'carcass', 'bones', 'carrion', 'deceased', 'departed', 'stiff', 'mort']
imaginary_words = ['devil','monster','whimsical', 'fantastic', 'abstract', 'fictional', 'unreal', 'fanciful', 'hypothetical', 'imagined', 'theoretical', 'ideal', 'visionary', 'apocryphal', 'assumed', 'chimerical', 'deceptive', 'delusive', 'dreamy', 'fabulous', 'hallucinatory', 'illusive', 'illusory', 'imaginative', 'legendary', 'made-up', 'mythological', 'nonexistent', 'notional', 'phantasmal', 'phantasmic', 'quixotic', 'shadowy', 'spectral', 'supposed', 'supposititious', 'trumped up', 'unsubstantial', 'apparitional', 'dreamed-up', 'dreamlike', 'fancied', 'figmental', 'imaginary']
changed_words = ['returned', 'reversed', 'transferred', 'replaced', 'restored', 'bartered', 'transposed', 'reciprocated', 'swapped', 'alternated', 'switched', 'rotated', 'substituted', 'commutated', 'interchanged', 'traded', 'shuffled', 'changed', 'turned', 'convert', 'switch', 'trasform']
animalList = ['dinosaur', 'aardwolf', 'admiral', 'adouri', 'african black crake', 'african buffalo', 'african bush squirrel', 'african clawless otter', 'african darter', 'african elephant', 'african fish eagle', 'african ground squirrel', 'african jacana', 'african lion', 'african lynx', 'african pied wagtail', 'african polecat', 'african porcupine', 'african red-eyed bulbul', 'african skink', 'african snake', 'african wild cat', 'african wild dog', 'agama lizard', 'agile wallaby', 'agouti', 'albatross', 'albatross', 'alligator', 'alligator', 'alpaca', 'amazon parrot', 'american virginia opossum', 'american alligator', 'american badger', 'american beaver', 'american bighorn sheep', 'american bison', 'american black bear', 'american buffalo', 'american crow', 'american marten', 'american racer', 'american woodcock', 'anaconda', 'andean goose', 'ant', 'anteater', 'anteater', 'antechinus', 'antelope ground squirrel', 'antelope', 'antelope', 'antelope', 'arboral spiny rat', 'arctic fox', 'arctic ground squirrel', 'arctic hare', 'arctic lemming', 'arctic tern', 'argalis', 'armadillo', 'armadillo', 'armadillo', 'armadillo', 'asian elephant', 'asian false vampire bat', 'asian foreset tortoise', 'asian lion', 'asian openbill', 'asian red fox', 'asian water buffalo', 'asian water dragon', 'asiatic jackal', 'asiatic wild ass', 'ass', 'australian brush turkey', 'australian magpie', 'australian masked owl', 'australian pelican', 'australian sea lion', 'australian spiny anteater', 'avocet', "azara's zorro", 'baboon', 'baboon', 'baboon', 'baboon', 'baboon', 'badger', 'badger', 'badger', 'badger', 'bahama pintail', 'bald eagle', 'baleen whale', 'banded mongoose', 'bandicoot', 'bandicoot', 'bandicoot', 'barasingha deer', 'barbet', 'barbet', 'barbet', 'bare-faced go away bird', 'barking gecko', 'barrows goldeneye', 'bat', 'bat', 'bat', 'bat-eared fox', 'bateleur eagle', 'bear', 'bear', 'bear', 'bear', 'bear', 'beaver', 'beaver', 'beaver', 'beaver', 'bee-eater', 'bee-eater', 'bee-eater', 'bee-eater', 'beisa oryx', 'bengal vulture', "bennett's wallaby", 'bent-toed gecko', 'bettong', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bison', 'black and white colobus', 'black bear', 'black curlew', 'black kite', 'black rhinoceros', 'black spider monkey', 'black swan', 'black vulture', 'black-backed jackal', 'black-backed magpie', 'black-capped capuchin', 'black-capped chickadee', 'black-cheeked waxbill', 'black-collared barbet', 'black-crowned crane', 'black-crowned night heron', 'black-eyed bulbul', 'black-faced kangaroo', 'black-footed ferret', 'black-fronted bulbul', 'black-necked stork', 'black-tailed deer', 'black-tailed prairie dog', 'black-tailed tree creeper', 'black-throated butcher bird', 'black-throated cardinal', 'black-winged stilt', 'blackbird', 'blackbuck', 'blackish oystercatcher', 'blacksmith plover', 'bleeding heart monkey', 'blesbok', 'bleu', 'bleu', 'blue and gold macaw', 'blue and yellow macaw', 'blue catfish', 'blue crane', 'blue duck', 'blue fox', 'blue peacock', 'blue racer', 'blue shark', 'blue waxbill', 'blue wildebeest', 'blue-breasted cordon bleu', 'blue-faced booby', 'blue-footed booby', 'blue-tongued lizard', 'blue-tongued skink', 'boa', 'boa', 'boa', 'boa', 'boa', 'boar', 'boat-billed heron', 'bobcat', 'bohor reedbuck', 'bonnet macaque', 'bontebok', 'booby', 'booby', 'booby', 'bottle-nose dolphin', 'boubou', 'brazilian otter', 'brazilian tapir', 'brindled gnu', 'brocket', 'brocket', 'brolga crane', 'brown and yellow marshbird', 'brown antechinus', 'brown brocket', 'brown capuchin', 'brown hyena', 'brown lemur', 'brown pelican', 'brush-tailed bettong', 'brush-tailed phascogale', 'brush-tailed rat kangaroo', 'buffalo', 'buffalo', 'buffalo', 'buffalo', 'bulbul', 'bulbul', 'bulbul', 'bunting', "burchell's gonolek", 'burmese black mountain tortoise', 'burmese brown mountain tortoise', 'burrowing owl', 'bush dog', 'bushbaby', 'bushbuck', 'bushpig', 'bustard', 'bustard', 'bustard', 'butterfly', 'butterfly', 'butterfly', 'buttermilk snake', 'caiman', 'california sea lion', 'camel', 'campo flicker', 'canada goose', 'canadian river otter', 'canadian tiger swallowtail butterfly', 'cape barren goose', 'cape clawless otter', 'cape cobra', 'cape fox', 'cape raven', 'cape starling', 'cape white-eye', 'cape wild cat', 'capuchin', 'capuchin', 'capuchin', 'capuchin', 'capybara', 'caracal', 'caracara', 'caracara', 'cardinal', 'cardinal', 'caribou', 'carmine bee-eater', 'carpet python', 'carpet snake', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'catfish', 'cattle egret', 'cereopsis goose', 'chacma baboon', 'chameleon', 'cheetah', 'chestnut weaver', 'chickadee', 'chilean flamingo', 'chimpanzee', 'chipmunk', 'chital', 'chuckwalla', 'civet', 'civet cat', 'civet', 'civet', "clark's nutcracker", 'cliffchat', 'coatimundi', 'coatimundi', 'cobra', 'cobra', 'cobra', 'cockatoo', 'cockatoo', 'cockatoo', 'cockatoo', 'cockatoo', 'cockatoo', "coke's hartebeest", 'collared lemming', 'collared lizard', 'collared peccary', 'colobus', 'colobus', 'colobus', 'columbian rainbow boa', 'comb duck', 'common boubou shrike', 'common brushtail possum', 'common dolphin', 'common duiker', 'common eland', 'common genet', 'common goldeneye', 'common green iguana', 'common grenadier', 'common langur', 'common long-nosed armadillo', 'common melba finch', 'common mynah', 'common nighthawk', 'common palm civet', 'common pheasant', 'common raccoon', 'common rhea', 'common ringtail', 'common seal', 'common shelduck', 'common turkey', 'common wallaroo', 'common waterbuck', 'common wolf', 'common wombat', 'common zebra', 'common zorro', 'constrictor', "cook's tree boa", 'coot', 'coqui francolin', 'coqui partridge', 'corella', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cormorant', 'cottonmouth', 'cougar', 'cow', 'coyote', 'crab', 'crab', 'crab', 'crab-eating fox', 'crab-eating raccoon', 'crake', 'crane', 'crane', 'crane', 'crane', 'crane', 'crane', 'crane', 'creeper', 'crested barbet', 'crested bunting', 'crested porcupine', 'crested screamer', 'crimson-breasted shrike', 'crocodile', 'crow', 'crow', 'crow', 'crown of thorns starfish', 'crowned eagle', 'crowned hawk-eagle', 'cuis', 'curlew', 'currasow', 'curve-billed thrasher', 'dabchick', 'dama wallaby', 'dark-winged trumpeter', 'darter', 'darwin ground finch', 'dassie', 'deer', 'deer', 'deer', 'deer', 'deer', 'deer', 'deer', 'deer', 'deer', 'defassa waterbuck', "denham's bustard", 'desert kangaroo rat', 'desert spiny lizard', 'desert tortoise', 'dik', 'dingo', 'dog', 'dog', 'dog', 'dog', 'dolphin', 'dolphin', 'dolphin', 'dove', 'dove', 'dove', 'dove', 'dove', 'dove', 'dove', 'dove', 'downy woodpecker', 'dragon', 'dragon', 'dragon', 'dragon', 'dragon', 'dragon', 'dragonfly', 'dromedary camel', 'drongo', 'duck', 'duck', 'duck', 'duck', 'duiker', 'duiker', 'dunnart', 'dusky gull', 'dusky rattlesnake', 'eagle owl', 'eagle', 'eagle', 'eagle', 'eagle', 'eagle', 'eagle', 'eagle', 'eagle', 'eagle', 'eastern boa constrictor', 'eastern box turtle', 'eastern cottontail rabbit', 'eastern diamondback rattlesnake', 'eastern dwarf mongoose', 'eastern fox squirrel', 'eastern grey kangaroo', 'eastern indigo snake', 'eastern quoll', 'eastern white pelican', 'echidna', 'egret', 'egret', 'egret', 'egyptian cobra', 'egyptian goose', 'egyptian viper', 'egyptian vulture', 'eland', 'elegant crested tinamou', 'elephant', 'elephant', 'eleven-banded armadillo', 'elk', 'emerald green tree boa', 'emerald-spotted wood dove', 'emu', 'eurasian badger', 'eurasian beaver', 'eurasian hoopoe', 'eurasian red squirrel', 'euro wallaby', 'european badger', 'european beaver', 'european red squirrel', 'european shelduck', 'european spoonbill', 'european stork', 'european wild cat', 'fairy penguin', 'falcon', 'falcon', 'fat-tailed dunnart', 'feathertail glider', 'feral rock pigeon', 'ferret', 'ferruginous hawk', 'field flicker', 'finch', 'fisher', 'flamingo', 'flamingo', 'flamingo', 'flamingo', 'flicker', 'flicker', 'flightless cormorant', 'flycatcher', 'flying fox', 'fork-tailed drongo', 'four-horned antelope', 'four-spotted skimmer', 'four-striped grass mouse', 'fowl', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'fox', 'francolin', 'francolin', 'frilled dragon', 'frilled lizard', 'fringe-eared oryx', 'frog', 'frogmouth', 'galah', 'galapagos albatross', 'galapagos dove', 'galapagos hawk', 'galapagos mockingbird', 'galapagos penguin', 'galapagos sea lion', 'galapagos tortoise', "gambel's quail", 'gaur', 'gazelle', 'gazelle', 'gazer', 'gecko', 'gecko', 'gecko', 'gecko', 'gecko', 'gelada baboon', 'gemsbok', 'genet', 'genet', 'genoveva', 'gerbil', 'gerenuk', 'giant anteater', 'giant armadillo', 'giant girdled lizard', 'giant heron', 'giant otter', 'gila monster', 'giraffe', 'glider', 'glider', 'glider', 'glossy ibis', 'glossy starling', 'gnu', 'goanna lizard', 'goat', 'godwit', 'golden brush-tailed possum', 'golden eagle', 'golden jackal', 'golden-mantled ground squirrel', 'goldeneye', 'goldeneye', 'goliath heron', 'gonolek', 'goose', 'goose', 'goose', 'goose', 'goose', 'goose', 'goose', 'goose', 'goose', 'gorilla', "grant's gazelle", 'gray duiker', 'gray heron', 'gray langur', 'gray rhea', 'great cormorant', 'great egret', 'great horned owl', 'great kiskadee', 'great skua', 'great white pelican', 'greater adjutant stork', 'greater blue-eared starling', 'greater flamingo', 'greater kudu', 'greater rhea', 'greater roadrunner', 'greater sage grouse', 'grebe', 'green heron', 'green vine snake', 'green-backed heron', 'green-winged macaw', 'green-winged trumpeter', 'grenadier', 'grenadier', 'grey fox', 'grey heron', 'grey lourie', 'grey mouse lemur', 'grey phalarope', 'grey-footed squirrel', 'greylag goose', 'griffon vulture', 'grison', 'grizzly bear', 'ground legaan', 'ground monitor', 'groundhog', 'grouse', 'grouse', 'guanaco', 'guerza', 'gull', 'gull', 'gull', 'gull', 'gull', 'gull', 'gull', 'gull', 'gulls', 'hanuman langur', 'harbor seal', 'hare', 'hartebeest', 'hartebeest', 'hawk', 'hawk', 'hawk', 'hawk-eagle', 'hawk-headed parrot', 'hedgehog', 'helmeted guinea fowl', 'hen', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'heron', 'herring gull', 'hippopotamus', 'hoary marmot', "hoffman's sloth", 'honey badger', 'hoopoe', 'hornbill', 'hornbill', 'hornbill', 'hornbill', 'horned lark', 'horned puffin', 'horned rattlesnake', 'hottentot teal', 'house crow', 'house sparrow', 'hudsonian godwit', 'hummingbird', 'huron', 'hyena', 'hyena', 'hyena', 'hyrax', 'ibex', 'ibis', 'ibis', 'ibis', 'iguana', 'iguana', 'iguana', 'impala', 'indian giant squirrel', 'indian jackal', 'indian leopard', 'indian mynah', 'indian peacock', 'indian porcupine', 'indian red admiral', 'indian star tortoise', 'indian tree pie', 'insect', 'jabiru stork', 'jacana', 'jackal', 'jackal', 'jackal', 'jackal', 'jackal', 'jackrabbit', 'jaeger', 'jaguar', 'jaguarundi', 'japanese macaque', 'javan gold-spotted mongoose', 'javanese cormorant', 'jungle cat', 'jungle kangaroo', 'kaffir cat', 'kafue flats lechwe', 'kalahari scrub robin', 'kangaroo', 'kangaroo', 'kangaroo', 'kangaroo', 'kangaroo', 'kangaroo', 'kelp gull', 'killer whale', 'king cormorant', 'king vulture', 'kingfisher', 'kingfisher', 'kingfisher', 'kinkajou', "kirk's dik dik", 'kiskadee', 'kite', 'klipspringer', 'knob-nosed goose', 'koala', 'komodo dragon', 'kongoni', 'kookaburra', 'kori bustard', 'kudu', 'land iguana', 'langur', 'langur', 'langur', 'lappet-faced vulture', 'lapwing', 'lapwing', 'large cormorant', 'large-eared bushbaby', 'lark', 'laughing dove', 'laughing kookaburra', 'lava gull', "leadbeateri's ground hornbill", 'least chipmunk', 'lechwe', 'legaan', 'legaan', 'legaan', 'lemming', 'lemming', 'lemur', 'lemur', 'lemur', 'lemur', 'lemur', 'leopard', 'leopard', 'lesser double-collared sunbird', 'lesser flamingo', 'lesser masked weaver', 'lesser mouse lemur', "levaillant's barbet", 'lilac-breasted roller', 'lily trotter', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'lion', 'little blue penguin', 'little brown bat', 'little brown dove', 'little cormorant', 'little grebe', 'little heron', 'lizard', 'lizard', 'lizard', 'lizard', 'lizard', 'lizard', 'lizard', 'lizard', 'llama', 'long-billed cockatoo', 'long-billed corella', 'long-crested hawk eagle', 'long-finned pilot whale', 'long-necked turtle', 'long-nosed bandicoot', 'long-tailed jaeger', 'long-tailed skua', 'long-tailed spotted cat', 'lorikeet', 'loris', 'lory', 'lourie', 'lynx', 'macaque', 'macaque', 'macaque', 'macaque', 'macaw', 'macaw', 'macaw', 'macaw', 'macaw', 'madagascar fruit bat', 'madagascar hawk owl', 'magellanic penguin', 'magistrate black colobus', 'magnificent frigate bird', 'magpie', 'magpie', 'malabar squirrel', 'malachite kingfisher', 'malagasy ground boa', 'malay squirrel', 'mallard', 'malleefowl', 'manatee', 'mandras tree shrew', 'mara', 'marabou stork', 'margay', 'marine iguana', 'marmot', 'marmot', 'marshbird', 'marten', 'masked booby', 'meerkat', 'meerkat', 'mexican beaded lizard', 'mexican boa', 'mexican wolf', "miner's cat", 'mississippi alligator', 'moccasin', 'mocking cliffchat', 'mockingbird', 'mongoose', 'mongoose', 'mongoose', 'mongoose', 'mongoose', 'monitor lizard', 'monitor', 'monitor', 'monitor', 'monkey', 'monkey', 'monkey', 'monkey', 'monkey', 'monster', 'moorhen', 'moose', 'mouflon', 'mountain duck', 'mountain goat', 'mountain lion', 'mourning collared dove', 'mouse', 'mudskipper', 'mule deer', 'musk ox', 'mynah', 'mynah', 'native cat', 'nelson ground squirrel', 'neotropic cormorant', 'netted rock dragon', 'nighthawk', 'nile crocodile', 'nilgai', 'nine-banded armadillo', 'north american beaver', 'north american porcupine', 'north american red fox', 'north american river otter', 'northern elephant seal', 'northern fur seal', 'northern phalarope', 'nubian bee-eater', 'numbat', 'nutcracker', 'nuthatch', 'nyala', 'ocelot', 'old world fruit bat', 'olive baboon', 'onager', 'openbill stork', 'openbill', 'opossum', 'orca', 'oribi', 'oriental short-clawed otter', 'oriental white-backed vulture', 'ornate rock dragon', 'oryx', 'oryx', 'osprey', 'ostrich', 'otter', 'otter', 'otter', 'otter', 'otter', 'otter', 'otter', 'otter', 'ovenbird', 'owl', 'owl', 'owl', 'owl', 'owl', 'owl', 'ox', 'oystercatcher', 'paca', 'pacific gull', 'paddy heron', 'pademelon', 'painted stork', 'pale white-eye', 'pale-throated three-toed sloth', "pallas's fish eagle", 'palm squirrel', 'pampa gray fox', 'paradoxure', 'parakeet', 'parrot', 'partridge', 'peacock', 'peacock', 'peccary', 'peccary', 'pelican', 'pelican', 'pelican', 'pelican', 'penguin', 'peregrine falcon', 'phalarope', 'phalarope', 'phalarope', 'phalarope', 'phascogale', 'phascogale', 'pheasant', 'pheasant', 'pie', 'pie', 'pied avocet', 'pied butcher bird', 'pied cormorant', 'pied crow', 'pied kingfisher', 'pig-tailed macaque', 'pigeon', 'pigeon', 'pine siskin', 'pine snake', 'pine squirrel', 'pintail', 'pintail', 'plains zebra', 'platypus', 'plover', 'plover', 'pocket gopher', 'polar bear', 'polecat', 'porcupine', 'possum', 'potoroo', 'prairie falcon', 'praying mantis', 'prehensile-tailed porcupine', 'pronghorn', 'puffin', 'puku', 'puma', 'puna ibis', 'purple grenadier', 'purple moorhen', 'pygmy possum', 'python', 'python', 'quail', 'quoll', 'rabbit', 'raccoon dog', 'raccoon', 'raccoon', 'racer snake', 'racer', 'racer', 'radiated tortoise', 'rainbow lory', 'rat', 'rat', 'rat', 'rattlesnake', 'raven', 'raven', 'red and blue macaw', 'red brocket', 'red deer', 'red hartebeest', 'red howler monkey', 'red kangaroo', 'red lava crab', 'red meerkat', 'red phalarope', 'red sheep', 'red squirrel', 'red-billed buffalo weaver', 'red-billed hornbill', 'red-billed toucan', 'red-billed tropic bird', 'red-breasted cockatoo', 'red-breasted nuthatch', 'red-capped cardinal', 'red-cheeked cordon bleu', 'red-headed woodpecker', 'red-knobbed coot', 'red-legged pademelon', 'red-necked phalarope', 'red-necked wallaby', 'red-shouldered glossy starling', 'red-tailed cockatoo', 'red-tailed hawk', 'red-tailed phascogale', 'red-tailed wambenger', 'red-winged blackbird', 'red-winged hawk', 'reedbuck', 'reindeer', 'rhea', 'rhea', 'rhea', 'rhesus macaque', 'rhesus monkey', 'rhinoceros', 'rhinoceros', 'rhinoceros', "richardson's ground squirrel", 'ring dove', 'ring-necked pheasant', 'ring-tailed coatimundi', 'ring-tailed gecko', 'ring-tailed lemur', 'ring-tailed possum', 'ringtail', 'ringtail cat', 'ringtail', 'river wallaby', 'roadrunner', 'roan antelope', 'robin', 'robin', 'rock dove', 'roe deer', 'roller', 'rose-ringed parakeet', 'roseat flamingo', 'roseate cockatoo', 'royal tern', 'rufous tree pie', 'rufous-collared sparrow', 'russian dragonfly', 'sable antelope', 'sacred ibis', 'saddle-billed stork', 'sage grouse', 'sage hen', 'sally lightfoot crab', 'salmon pink bird eater tarantula', 'salmon', 'sambar', 'sandgrouse', 'sandhill crane', 'sandpiper', 'sarus crane', 'savanna baboon', 'savanna fox', 'savannah deer', 'savannah deer', 'scaly-breasted lorikeet', 'scarlet macaw', 'scottish highland cow', 'screamer', 'screamer', 'sea birds', 'seal', 'secretary bird', 'serval', 'seven-banded armadillo', 'shark', 'sheathbill', 'sheep', 'sheep', 'sheep', 'shelduck', 'shelduck', 'short-beaked echidna', 'short-nosed bandicoot', 'shrew', 'shrike', 'sidewinder', 'sifaka', 'silver gull', 'silver-backed fox', 'silver-backed jackal', 'siskin', 'siskin', 'skimmer', 'skink', 'skua', 'skua', 'skunk', 'skunk', 'slender loris', 'slender-billed cockatoo', 'sloth bear', 'sloth', 'small indian mongoose', 'small-clawed otter', 'small-spotted genet', 'small-toothed palm civet', "smith's bush squirrel", 'snake', 'snake-necked turtle', 'snow goose', 'snowy egret', 'snowy owl', 'snowy sheathbill', 'sociable weaver', 'sockeye salmon', 'south african hedgehog', 'south american meadowlark', 'south american puma', 'south american sea lion', 'southern black-backed gull', 'southern boubou', 'southern brown bandicoot', 'southern elephant seal', 'southern ground hornbill', 'southern hairy-nosed wombat', 'southern lapwing', 'southern right whale', 'southern screamer', 'southern sea lion', 'southern tamandua', 'southern white-crowned shrike', 'sparrow', 'sparrow', 'spectacled caiman', 'spider', 'spoonbill', 'spoonbill', 'sportive lemur', 'spotted deer', 'spotted hyena', 'spotted wood sandpiper', 'spotted-tailed quoll', 'springbok', 'springbuck', 'springhare', 'spur-winged goose', 'spurfowl', 'square-lipped rhinoceros', 'squirrel glider', 'squirrel', 'stanley bustard', 'stanley crane', 'starfish', 'starling', 'steenbok', 'steenbuck', 'steller sea lion', "steller's sea lion", 'stick insect', 'stilt', 'stone sheep', 'stork', 'striated heron', 'striped dolphin', 'striped hyena', 'striped skunk', 'sugar glider', 'sulfur-crested cockatoo', 'sun gazer', 'sunbird', 'sungazer', 'superb starling', 'suricate', "swainson's francolin", 'swallow', 'swallow-tail gull', 'swamp deer', 'swan', 'tailless tenrec', 'tamandua', 'tammar wallaby', 'tapir', 'tarantula', 'tarantula', 'tasmanian devil', 'tawny eagle', 'tawny frogmouth', 'tayra', 'teal', 'tenrec', 'tern', 'thirteen-lined squirrel', "thomson's gazelle", 'thrasher', 'three-banded plover', 'tiger', 'tiger cat', 'tiger snake', 'timber wolf', 'tinamou', 'toddy cat', 'tokay gecko', 'topi', 'tortoise', 'toucan', 'toucan', 'tree porcupine', 'tropical buckeye butterfly', 'trotter', 'trumpeter swan', 'trumpeter', 'trumpeter', 'tsessebe', 'turaco', 'turkey vulture', 'turkey', 'turtle', 'two-banded monitor', 'two-toed sloth', 'two-toed tree sloth', 'tyrant flycatcher', 'uinta ground squirrel', 'urial', "verreaux's sifaka", 'vervet monkey', 'vicuna', 'vine snake', 'violet-crested turaco', 'violet-eared waxbill', 'viper', 'vulture', 'wagtail', 'wallaby', 'wallaroo', 'wambenger', 'wapiti', 'warthog', 'water legaan', 'water moccasin', 'water monitor', 'waterbuck', 'waterbuck', 'wattled crane', 'waved albatross', 'waxbill', 'waxbill', 'waxbill', 'weaver', 'weeper capuchin', 'western bearded dragon', 'western grey kangaroo', 'western lowland gorilla', 'western palm tanager', 'western patch-nosed snake', 'western pygmy possum', 'western spotted skunk', 'whale', 'whip-tailed wallaby', 'white rhinoceros', 'white spoonbill', 'white stork', 'white-bellied sea eagle', 'white-browed owl', 'white-browed sparrow weaver', 'white-cheeked pintail', 'white-eye', 'white-faced tree rat', 'white-faced whistling duck', 'white-fronted bee-eater', 'white-fronted capuchin', 'white-headed vulture', 'white-lipped peccary', 'white-mantled colobus', 'white-necked raven', 'white-necked stork', 'white-nosed coatimundi', 'white-rumped vulture', 'white-tailed deer', 'white-tailed jackrabbit', 'white-throated kingfisher', 'white-throated monitor', 'white-throated robin', 'white-throated toucan', 'white-winged black tern', 'white-winged dove', 'white-winged tern', 'wild boar', 'wild turkey', 'wild water buffalo', 'wildebeest', 'wolf spider', 'wolf', 'wombat', 'wood pigeon', 'woodchuck', 'woodcock', 'woodpecker', 'woodrat', 'woolly-necked stork', 'worm snake', 'woylie', 'yak', 'yellow baboon', 'yellow mongoose', 'yellow-bellied marmot', 'yellow-billed hornbill', 'yellow-billed stork', 'yellow-brown sungazer', 'yellow-crowned night heron', 'yellow-headed caracara', 'yellow-necked spurfowl', 'yellow-rumped siskin', 'yellow-throated sandgrouse', 'zebra', 'zebra', 'zorilla']
groupList = ['anybody', 'everybody', 'people', 'everyman', 'masses', 'populace', 'voters', 'everyone']


##########################
# Checks if a word is plural or not.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is plural;
#	False otherwise.
##########################

def isPlural(word):
	if word == []:
		return False
	if p.singular_noun(word) is False:
		return False
	return True

##########################
# Removes duplicates in a list.
# Parameter:
# 	@lista: list of results.
# Return:
# 	List of array results;
##########################

def removeDuplicatesAndCreateListOfArrayResult(lista):
	newlist = []
	for x in lista:
		for y in x:
			if y not in newlist:
				newlist.append(y)
	return newlist


##########################
# Checks if word is in the instance variable list 'undefinedList'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is in 'undefinedList';
#	False otherwise.
##########################

def isOccupational(word):
	if word == []:
		return False
	data = json.load(open('Wikidata/occupazione (Q12737077).json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'].lower() == word.lower()):
			return True
		else:
			for w in distro['altLabel_list'].replace(' ','').split(','):
				if w.lower() == word.lower():
					return True
	return False

##########################
# Checks if word is in the list 'etnia (Q41710).json'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is in 'etnia (Q41710).json';
#	False otherwise.
##########################

def isEthnic(word):
	if isPlural(word):
		word = p.singular_noun(word)
	if word == []:
		return False
	data = json.load(open('Wikidata/etnia (Q41710).json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'].lower() == word.lower()):
			return True
		else:
			for w in distro['altLabel_list'].replace(' ','').split(','):
				if w.lower() == word.lower():
					return True
	return False

##########################
# Checks if word is in the instance variable list 'imaginary_words'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is in 'imaginary_words';
#	False otherwise.
##########################

def isImaginary(word):
	if word == []:
		return False
	
	if word.lower() in imaginary_words:
			return True
	return False

##########################
# Checks if word is in the instance variable list 'dead_words'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is in 'dead_words';
#	False otherwise.
##########################

def isDead(word):
	if word == []:
		return False
	
	if word.lower() in dead_words:
			return True
	return False


##########################
# Defines the sex of a character through the use of knowledge bases created through wikidata.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is a female word;
#	False if the word is a male word;
#	15 if is undefined.
##########################

def gender(word):
	word = word.replace('NNP','').replace('nnp','')
	if isPlural(word):
		word = p.singular_noun(word)
	file = open('femaleList.txt','r',encoding='utf-8')
	list_one = file.read().split('\n')
	list_one = [k.lower() for k in list_one ]
	for x in list_one:
		if(word.lower() == x.lower()):
			return True
	if isFemaleName(word):
		return True
	return genderMale(word)

##########################
# Defines the sex of a character through the use of knowledge bases created through wikidata.
# Parameter:
# 	@word: string word.
# Return:
#	False if the word is a male word;
#	15 if is undefined.
##########################

def genderMale(word):
	file = open('maleList.txt','r',encoding='utf-8')
	list_one = file.read().split('\n')
	list_one = [k.lower() for k in list_one ]
	for x in list_one:
		if(word.lower() == x.lower()):
			return False
	if(isMaleName(word)):
		return False
	return 15

##########################
# Checks if one word of the group is in the instance variable list 'dead_words'.
# Parameter:
# 	@items: list of string word.
# Return:
# 	True if the one word is in 'dead_words';
#	False otherwise.
##########################

def deadGroup(items):
	for k in items:
		if makeSingularAndLower(k) in dead_words:
			return True
	return False

##########################
# Checks if one word of the group is in the instance variable list 'imaginary_words'.
# Parameter:
# 	@items: list of string word.
# Return:
# 	True if the one word is in 'imaginary_words';
#	False otherwise.
##########################

def imaginaryGroup(items):
	for k in items:
		if makeSingularAndLower(k) in imaginary_words:
			return True
	return False


##########################
# Defines the sex of a group of characters through the use of knowledge bases created through wikidata.
# Parameter:
# 	@group: list of characters.
# Return:
# 	'M' if there are only males;
#	'F' if there are only females;
#	'J' if there are males and females;
#	'I' if they are undefined.
##########################

def defineGenderGroup(group):
	sex = []
	for item in group:
		if (isOccupational(item) or item == 'someone'):
			sex.append('I')
		elif gender(item) == False:
			sex.append('M')
		elif gender(item) == 15:
			sex.append('I')
		else:
			sex.append('F')
		if 'M' in sex:
			if 'F' in sex:
				return 'J'
	if 'M' not in sex:
		if 'F' not in sex:
			return 'I'
	if 'M' not in sex:
		return 'F'
	else:
		return 'M'

##########################
# Clean results to take only the code for each character.
# Parameter:
# 	@result: list of elements.
# Return:
# 	list of clean results.
##########################

def cleanResults(result):
	newRes = []
	for x in result:
		newRes.append(x.split(' ')[1].replace('(','').replace(')',''))
		#newRes.append(x.split('(')[1].split(')')[0])
	return newRes


##########################
# Removes the pronouns to avoid coding twice the same character.
# Parameter:
# 	@lista: list of elements.
# Return:
# 	list without pronouns.
##########################

def removePRP(lista):
	new = []
	for x in lista:
		if ' PRP' not in x:
			new.append(x)
	return new

##########################
# Removes surname/adjectives of characters to avoid to encoded him twice. 
# Ex: 'Italian guy' (We only code 'italian' and not 'guy')
# Parameter:
# 	@lista: list of elements.
#	@array: list of string creating splitting the report.
# Return:
# 	list without surnames.
##########################

def removeSurnameAndDoubleCoding(lista,array):
	new_lista = []
	if lista == []:
		return lista
	for x in range(len(lista)):
		for y in range(len(array)):
			if lista[x].split(' ')[0] == array[y]:
				if x<len(lista)-1 and y<len(array)-1:
					if lista[x+1].split(' ')[0] == array[y+1]:
						new_lista.append(lista[x+1])
	for k in new_lista:
		if k in lista:
			lista.remove(k)
	return lista

##########################
# Removes characters already encoded within a group.
# Parameter:
# 	@lista: list of elements.
# Return:
# 	list without already encoded characters.
##########################

def removeSingleItems(lista):
	for x in lista:
		x = x.split(' ')[0]
		if isPlural(x):
			for y in lista:
				if p.singular_noun(x) == y.split(' ')[0]:	
					lista.remove(y)
	return lista

##########################
# Remove already encoded characters.
# Parameter:
# 	@lista: list of elements.
# Return:
# 	list without already encoded characters.
##########################

def removeCodedCharacter(lista):
	for x in lista:
		x1 = x
		x1 = x1.split(' (')[0]
		if ' ' in x1:
			for y in x1.split(' '):
				for k in lista:
					if y in k and k != x and y!='' and y!=' ':
						lista.remove(k)
	return lista

##########################
# Removes duplicates within a list.
# Parameter:
# 	@lista: list of elements.
# Return:
# 	list without duplicates.
##########################

def removeDuplicates(lista):
	for i in lista:
			if lista.count(i) > 1:
				lista.remove(i)
				removeDuplicates(lista)
	return lista

##########################
# Checks if word is in the instance variable list 'undefinedList'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is in 'undefinedList';
#	False otherwise.
##########################

def isUndefined(word):
	if word == []:
		return False
	if (word.lower() in undefinedList):
			return True
	return False
		

##########################
# Checks whether the word represents a group of characters by using wordnets and 
# if word is in the instance variable list 'groupList'.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word is a word representing a group;
#	False otherwise. 
##########################

def isAGroup(word):
	if word == []:
		return False
	if word.lower() in groupList:
		return True
	if isPlural(word):
		if isAPerson(p.singular_noun(word)):
			return True
	if(word=='they'):
		return True
	if word == []:
		return False
	if(len(wn.synsets(word)) > 0):
		typology = wn.synsets(word)[0].lexname()
		for i in wn.synsets(word):
			if i.lexname() == 'noun.group':
				for k in wn.synsets(word):
					if k.lexname() == 'noun.body':
						return False
				for j in wn.synsets(word):
					if j.lexname() == 'noun.person':
						return True
	return False

##########################
# Search for the word in the wordnet dictionary and return a list of the taxonomy of each result. 
# Sort the list obtained by the value of the frequency of use present in the wordnet of each result of the list.
# Parameter:
# 	@word: string word.
# Return:
# 	List of taxonomies representing the parameter ordered by the frequency of use. 
##########################

def getFrequency(word):
	lista = []
	final_list = []
	if(len(wn.synsets(word)) > 0):
		typology = wn.synsets(word)
		for s in typology:
			for l in s.lemmas():
				lista.append(s.lexname())
				lista.append(l.count())
				flag = True
				for x in final_list:
					if lista[0] in x[0]:
						flag = False
						x[1] = x[1]+lista[1]
						break
				if flag:
					final_list.append(lista)
				lista = []
	lista = sorted(final_list, key = lambda l: (l[1]), reverse = True)

	return lista 

##########################
# Search for the word in the wordnet dictionary and check if the word identifies a character in most cases.
# Parameter:
# 	@word: string word.
# Return:
# 	True if the word identifies a character in most cases;
#	False if the word not identifies a character in most cases;
##########################

def CharacterFrequency(word):

	lista = getFrequency(word)

	if lista != []:
		if 'verb' in lista[0][0]:
			return False
		if 'artifact' in lista[0][0]:
			return False
		if 'adj' in lista[0][0]:
			return False
		if 'adv' in lista[0][0]:
			return False	
		if 'noun'in lista[0][0]:
			if 'person' in lista[0][0] or 'animal' in lista[0][0] or 'group' in lista[0][0] or 'Tops' in lista[0][0]:
				return True
			else:
				return False
	return True

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter or
# check the description of wordnets of the parameter is identified as noun.animal and not noun.body or noun.person
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in an animal;
#	False if word is not an animal.
##########################

def isAnimal(word):
	if word == []:
		return False
	if word.lower() in animalList:
			return True
	if(len(wn.synsets(word)) > 0):
		typology = wn.synsets(word)[0].lexname()
		for i in wn.synsets(word):
			if i.lexname() == 'noun.body':
				return False
			if i.lexname() == 'noun.person':
				return False
		typology = typology.split('.')[1]
		if(typology == 'animal'):
			return True
	return False

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter.
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in isObjectList.json;
#	False if word is not in isObjectList.json.
##########################

def isObject(word):
	if word == []:
		return False
	file = open('isObjectList.txt','r',encoding='utf-8')
	list_one = file.read().split('\n')
	list_one = [k.lower() for k in list_one ]
	for x in list_one:
		if(word.lower() == x):
			return True
	return False

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter.
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in relationFamily.json;
#	False if word is not in relationFamily.json.
##########################

def isRelationFamily(word):
	if word == []:
		return False
	data = json.load(open('Wikidata/relationFamily.json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'].lower() == word.lower()):
			return True
		else:
			for w in distro['altLabel_list'].replace(' ','').split(','):
				if w.lower() == word.lower():
					return True
	return False

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter.
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in femaleName.json;
#	False if word is not in femaleName.json.
##########################

def isFemaleName(word):
	if word == []:
		return False
	data = json.load(open('Wikidata/femaleName.json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'] == word):
			return True
	return False

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter.
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in maleName.json;
#	False if word is not in maleName.json.
##########################

def isMaleName(word):
	if word == []:
		return False
	data = json.load(open('Wikidata/maleName.json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'] == word):
			return True
	return False

##########################
# Check if inside the knowledge base created using wikipedia is present the term passed as an parameter.
# Parameter:
# 	@word: string word.
# Return:
# 	True if word is in isPerson.json;
#	False if word is not in isPerson.json.
##########################

def isPersonWiki(word):
	if word == []:
		return False
	data = json.load(open('Wikidata/isPerson.json',encoding='utf-8'))

	for distro in data:
		if (distro['workLabel'].lower() == word):
			return True
		else:
			for w in distro['altLabel_list'].replace(' ','').split(','):
				if w.lower() == word.lower() and word != 'am':
					return True
	return False

##########################
# Identifies whether the term represents a person or not. 
# Controls whether the term passed as a parameter is present in lists of categories of people or whether 
# the description of wordnets with the highest frequency of use of the parameter is identified as noun.person.
# Parameter:
# 	@word: string word.
# Return:
# 	True if is a general person;
#	False if is not a general person.
##########################

def isAPerson(word):
	if word == []:
		return False
	word = word.replace('NNP','')
	if word == [] or word == '':
		return False
	if (isPersonWiki(word) or isMaleName(word) or isFemaleName(word) or isRelationFamily(word)) and not isObject(word) and not isAnimal(word):
			return True
	if(len(wn.synsets(word)) > 0):
		if CharacterFrequency(word) and not isObject(word) and not isAnimal(word):
			typology = wn.synsets(word)[0].lexname()
			for i in wn.synsets(word):
				if i.lexname() == 'noun.person':
					for k in wn.synsets(word):
						if k.lexname() == 'noun.body' or k.lexname() == 'noun.group':
							return False
					return True
	return False

##########################
# Method for extracting names, adjectives, adverbs, verbs and conjunctions inside the morphological tree generated by the dream report.
# Parameters:
# 	@parent: root of the morphological tree representing the dream report;
# 	@subj: list in which to insert the extracted elements.
# Return:
# 	List of extracted elements
##########################

def getNodes(parent,subj):
	for node in parent:
		if type(node) is nltk.Tree:
			if node.label() == 'NN' or node.label() == 'CC'  or node.label() == 'POS' or node.label() == 'NNS' or node.label() == 'ADJP' or node.label() == 'JJ' or node.label() == 'VBN' or node.label() == 'VB' or node.label() == 'VBZ' or node.label() == 'NNPS' or node.label() == 'CD': ## MAYBE ALSO PRP
				for i in node.leaves():
					subj.append(i)
			elif  node.label() == 'NNP':
				for i in node.leaves():
					subj.append(i+'NNP')
			elif node.label() == 'PRP':
				for i in node.leaves():
					if i == 'I':
						subj.append(i)
			getNodes(node,subj)
	return subj

##########################
# Transform words into singular and lower.
# Parameter:
# 	@item: string word.
# Return:
# 	singolar e lower string word.
##########################

def makeSingularAndLower(item):
	p = inflect.engine()
	lowerword=item.lower()
	if(p.singular_noun(item)):
		lowerword = p.singular_noun(item).lower()
	return lowerword


##########################
# Method for identifying groups of characters and dead characters.
# Parameter:
# 	@subj: list of words extracted from the morphological tree.
# Return:
# 	list of lists of extracted elements
##########################

def makeGroupAndFindDeadCharacters(subj):
	result = [[]]
	last = []
	for i in range(len(subj)):
		if(subj[i] != []):
			if subj[i] != 'and' and makeSingularAndLower(subj[i]) not in dead_words and subj[i] !='\'s':
				result.append([subj[i]])
				last = [subj[i]]
			
			# Identify groups of characters within the list by using the conjunction "and". An example of a group is "Max and Sarah".
			
			elif(subj[i] == 'and'):
				if (isAPerson(subj[i-1]) or isAGroup(subj[i-1]) or isAnimal(subj[i-1]) or isUndefined(subj[i-1]) or 'NNP' in subj[-1]):
					if (i<len(subj)-1):
						if (isAPerson(subj[i+1]) or isAGroup(subj[i+1]) or isAnimal(subj[i+1]) or isUndefined(subj[i+1]) or 'NNP' in subj[-1]):
							group = [subj[i-1]]+[subj[i]]+[subj[i+1]]
							if [subj[i-1]] in result:
								result.remove([subj[i-1]])
							subj[i+1] = []
							result.append(group)
							last=group
			
			# checks if the word is present inside the 'dead_words' instance variable 
			
			elif(makeSingularAndLower(subj[i]) in dead_words):
				if(subj[i-1]=='\'s'):
					result.append(last+[subj[i]])
					if last in result:
						result.remove(last)
				elif(i<len(subj)-1):
					group = [subj[i]]+[subj[i+1]]
					subj[i+1] = []
					result.append(group)
				else:
					group = [subj[i-1]]+[subj[i]]
					if subj[i-1] in result:
						result.remove(subj[i-1])
					result.append(group)
	if [] in result:
		result.remove([])
	if [[]] in result:
		result.remove([[]])

	return result


##########################
# Method for identifying groups of characters and imaginary characters.
# Parameter:
# 	@subj: list of words extracted from the morphological tree.
# Return:
# 	list of lists of extracted elements
##########################

def makeGroupAndFindImaginaryCharacters(subj):

	result = [[]]
	last = []
	for i in range(len(subj)):
		if(subj[i] != []):
			if subj[i] != 'and' and makeSingularAndLower(subj[i]) not in imaginary_words:
				result.append([subj[i]])
				last = [subj[i]]

			# Identify groups of characters within the list by using the conjunction "and". An example of a group is "Max and Sarah".
			
			elif(subj[i] == 'and'):
				if (isAPerson(subj[i-1]) or isAGroup(subj[i-1]) or isAnimal(subj[i-1]) or isUndefined(subj[i-1]) or 'NNP' in subj[i-1]):
					if (i<len(subj)-1):
						if (isAPerson(subj[i+1]) or isAGroup(subj[i+1]) or isAnimal(subj[i+1]) or isUndefined(subj[i+1]) or 'NNP' in subj[i+1]):
							group = [subj[i-1]]+[subj[i]]+[subj[i+1]]
							if [subj[i-1]] in result:
								result.remove([subj[i-1]])
							subj[i+1] = []
							result.append(group)
							last=group

			# checks if the word is present inside the 'imaginary_words' instance variable 
			
			elif(makeSingularAndLower(subj[i]) in imaginary_words):
				if (isAPerson(subj[i-1]) or isAGroup(subj[i-1]) or isAnimal(subj[i-1]) or isUndefined(subj[i-1]) or 'NNP' in subj[-1]):
					result.append([subj[i-1]]+[subj[i]])
					if last in result:
						result.remove(last)
				elif(i<len(subj)-1):
					if (isAPerson(subj[i+1]) or isAGroup(subj[i+1]) or isAnimal(subj[i+1]) or isUndefined(subj[i+1])or 'NNP' in subj[-1]):
						result.append([subj[i]]+[subj[i+1]])
						subj[i+1] = []
	if [] in result:
		result.remove([])
	if [[]] in result:
		result.remove([[]])

	return result


##########################
# Method for identifying groups of characters and imaginary characters.
# Parameter:
# 	@subj: list of words extracted from the morphological tree.
# Return:
# 	list of lists of extracted elements
##########################

def makeGroupAndFindChangedFormCharacters(subj):
	result = [[]]
	last = []
	for i in range(len(subj)):
		if(subj[i] != []):
			if subj[i] != 'and' and makeSingularAndLower(subj[i]) not in changed_words:
				result.append([subj[i]])
				last = [subj[i]]
			elif(subj[i] == 'and'):
				if (isAPerson(subj[i-1]) or isAGroup(subj[i-1]) or isAnimal(subj[i-1]) or isUndefined(subj[i-1]) or 'NNP' in subj[i-1]):
					if (i<len(subj)-1):
						if (isAPerson(subj[i+1]) or isAGroup(subj[i+1]) or isAnimal(subj[i+1]) or isUndefined(subj[i+1])or 'NNP' in subj[i+1]):
							group = [subj[i-1]]+[subj[i]]+[subj[i+1]]
							if [subj[i-1]] in result:
								result.remove([subj[i-1]])
							subj[i+1] = []
							result.append(group)
							last=group
			elif(makeSingularAndLower(subj[i]) in changed_words):
				if (isAPerson(subj[i-1]) or isAGroup(subj[i-1]) or isAnimal(subj[i-1]) or isUndefined(subj[i-1])or 'NNP' in subj[i-1]):
					if(i<len(subj)-1):
						if (isAPerson(subj[i+1]) or isAGroup(subj[i+1]) or isAnimal(subj[i+1]) or isUndefined(subj[i+1])or 'NNP' in subj[i+1]):
							result.append(last+[subj[i]]+[subj[i+1]])
							subj[i+1] = []
	if [] in result:
		result.remove([])
	if [[]] in result:
		result.remove([[]])

	return result

##########################
# Codes characters following rules in https://dreams.ucsc.edu/Coding/characters.html 
# Parameter:
# 	@characters: list of possible characters. Ex. [['guy'],[Paolo]]
# Return:
# 	list of coded characters.
##########################

def code(characters):
	result = []
	last = ''
	p = inflect.engine()
	for items in characters:
		if items == [''] or items == []:
			continue
		if (len(items)==1):
			if CharacterFrequency(items[0].replace('NNP','').replace('nnp','')):
				if items[0] == 'I':
					result.append(items[0]+' D')
				else:
					item = items[0].replace('nnp','').replace('NNP','')
					p = inflect.engine()
					lowerword=item.lower()
					if(p.singular_noun(item)):
						lowerword = p.singular_noun(item).lower()
					if(isAnimal(item) and not isAPerson(item)):#'''and not isAPerson(item)'''
						if(isPlural(item)):
							result.append(item+' (2ANI)')
						else:
							result.append(item+' (1ANI)')
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
							string = string+'S'
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
							string = string+'S'
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
							string = string+'S'
						if(lowerword == 'child' or lowerword == 'kid'):
							string = string+'C)'
							result.append(item+string)
						elif(lowerword == 'baby'):
							string = string+'B)'
							result.append(item+string)
						else:
							string = string+'A)'
							result.append(item+string)
					elif((isAPerson(item) and not p.singular_noun(item) and not isAnimal(item)) or isEthnic(item)):
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
							string = string+'S'
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
						string = ' (2'
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
							string = string+'S'
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
			for l in items:
				if l == []:
					items.remove(l)
			for item in items:
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
						string = string+'S' 
						string = string+'A)' 
						result.append((' '.join(items)+' '+'dead'+string).replace('NNP',''))
					elif imaginaryGroup(items):
						string = ' (6'
						string = string+defineGenderGroup(items)
						string = string+'S' 
						string = string+'A)' 
						result.append((' '.join(items)+' '+'imaginary'+string).replace('NNP',''))
					else:
						items.remove('and')
						flag = True
						for item in items:
							if code([[item]]) == []:
								flag = False
						if flag:
							string = string+defineGenderGroup(items)
							string = string+'S' 
							string = string+'A)' 
							result.append((' '.join(items)+' '+string).replace('NNP',''))
						else:
							for x in items:
								cod = code([[x]])
								if cod != []:
									result.append(cod[0])
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
						string = string+'K' 
						string = string+'A)' 
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
						string = string+'K' 
						string = string+'A)'
						result.append((' '.join(items)+' '+k+string))
					if(makeSingularAndLower(k) in changed_words):
						#print('===')
						#print(items)
						#print(k, makeSingularAndLower(k))
						items.remove(k)
						#print(items)
						#print('---')
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
# 	@sent: dream report.
# Return:
# 	list of clean coded characters.
##########################

def character_code(text, stanford_parser):
	text = text.replace(',',' and').replace("!",".").replace('?','').replace('(','').replace(')','').replace('\'',' \' ').replace('\"',' \" ').replace('&', 'and')
	
	senteces = tokenize.sent_tokenize(text)
	result = []
	another_list = []
	for s in senteces:
		try:
			#stanford_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
			parse = stanford_parser.raw_parse(s)
			array = s.split(' ')
			tree = list(parse)
			# tree[0].draw() #print the tree
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
		except:

			# Report too long. Divided into portions and creating the tree for each part.

			array = s.split()
			lung = len(array)

			part1 = ' '.join(array[:int(lung/2)])
			part2 = ' '.join(array[int(lung/2):])

			parse1 = stanford_parser.raw_parse(part1)
			parse2 = stanford_parser.raw_parse(part2)

			array = s.split(' ')

			tree1 = list(parse1)
			tree2 = list(parse2)

			subject = []
			subject= getNodes(tree1,subject)
			subject= getNodes(tree2,subject)

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
	result = removeSingleItems(result)
	result = removeSurnameAndDoubleCoding(result,array)
	result = removePRP(result)
	for i in another_list:

		result = result+i
	result = cleanResults(result)	### Comment on this line if you want to see the coded word.
	
	return result

# print(character_code('That guy was eating an ice cream.'))










