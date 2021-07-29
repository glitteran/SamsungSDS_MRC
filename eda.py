import random
import pickle
import re
from konlpy.tag import Komoran

wordnet = {}
with open("data/wordnet.pickle", "rb") as f:
	wordnet = pickle.load(f)
	"""
	ex) 
	'기분': ['기분', '열심'], 
	'습관성': ['습관성'], 
	'에로스': ['에로스'], 
	'친애': ['친애', '애착', '러브', '연정', '애정', '사랑', '애호', '자애', '연애', '귀염', '모정', '총애', '성애'], 
	'희열': ['희열', '기쁨', '즐거움', '신', '환희'], '흥': ['흥', '재미', '신', '즐거움'], 
	'재미': ['재미', '향락', '기쁨', '환락', '즐거움', '환희', '쾌락'], '위안': ['위안', '안심'], 
	'불쾌': ['불쾌'], 
	'사랑': ['사랑', '애호'], 
	'혐오': ['혐오', '혐의', '멀미', '반감', '싫증'],
	"""


# 문자, 숫자, ?만 남도록 정제. 
def get_only_hangul(line):
	parseText= re.sub('[^ A-Za-z0-9가-힣?]', '', line)

	return parseText



########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
# "내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?" -> '내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?'

"""
위 예시를 보면, 제대로 augmentation 되지 않는 것을 확인할 수 있음. 
그 이유는 문장에 대해서 띄어쓰기 단위를 기준으로 토큰화하기 떄문에 
토큰을 유의어로 바꾸는 과정에서 '도시는?'로 일치하는 단어가 wordnet에 존재하지 않기 때문임. 
따라서, '도시는?'에서 다시 '도시'로 추출할 수 있도록 코드를 수정해야 함. 
"""
########################################################################
def synonym_replacement(words, n):
	new_words = words.copy()
	#토큰에서 중복되는 토큰을 제외하고 list에 저장. 
	random_word_list = list(set([word for word in words]))
	#토큰이 저장된 list를 랜덤하게 섞기.
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		#한 단어 토큰에 대한 유의어 list 생성 
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			#한 단어 토큰에 대한 유의어에서 랜덤하게 하나 골라 단어 교체. 
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		#바꿀 수 있는 횟수를 넘어서면 멈춤. 
		if num_replaced >= n:
			break

	if len(new_words) != 0:
		sentence = ' '.join(new_words)
		new_words = sentence.split(" ")

	else:
		new_words = ""

	return new_words

#들어온 단어에 대하여 wordnet에 들어있는 유의어 찾기 
def get_synonyms(word):
	synomyms = []

	try:
		for syn in wordnet[word]:
			for s in syn:
				synomyms.append(s)
	except:
		pass

	return synomyms

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
# "내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?" -> '내년까지 87만kw급 2기가 추가로 도시는?'
########################################################################
def random_deletion(words, p):
	if len(words) == 1:
		return words

	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p: #0~1사이로 random하게 확률값을 생성했을 때, p보다 높은 r값에 대하여 그때의 단어 토큰 포함. 
			new_words.append(word)

	if len(new_words) == 0: #토큰이 다 없어지는 것을 우려하여 한 개만 임의로 삭제함. 
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
# "내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?" -> '내년까지 87만kw급 2기가 화력발전소 추가로 들어서는 도시는?'
########################################################################
def random_swap(words, n):
	new_words = words.copy()
	#n번 반복하여 swap을 진행 함. 
	for _ in range(n):
		new_words = swap_word(new_words)

	return new_words

def swap_word(new_words):
	#토큰들 중 하나의 index를 랜덤하게 골라 random_idx_1, random_idx_2에 할당 
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1

	if len(new_words) < 1: #두 개의 토큰을 고를 수 없는 상황이면 토큰들을 그대로 반환함. 
		return new_words
	elif len(new_words) == 2 : #두 개의 토큰만 존재할 경우 바로 swap 
		new_words[0], new_words[1] = new_words[1], new_words[0]
		return new_words
	else:
		while random_idx_2 == random_idx_1: #random_idx_2가 random_idx_1와 다른 index가 나올때까지 index를 랜덤하게 선택 
			random_idx_2 = random.randint(0, len(new_words)-1)

	#2개의 토큰 swapping
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
# "내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?" -> '내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?'
"""
위 예시를 보면, 제대로 augmentation 되지 않는 것을 확인할 수 있음. 
그 이유는 문장에 대해서 띄어쓰기 단위를 기준으로 토큰화하기 떄문에 
토큰을 유의어로 바꾸는 과정에서 '도시는?'로 일치하는 단어가 wordnet에 존재하지 않기 때문임. 
따라서, '도시는?'에서 다시 '도시'로 추출할 수 있도록 코드를 수정해야 함. 
"""
########################################################################
def random_insertion(words, n):
	new_words = words.copy()
	#n번 반복하여 insertion을 진행 함. 
	for _ in range(n):
		add_word(new_words)
	
	return new_words


def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) <= 0: #유의어가 존재할 때까지 반복
		if len(new_words) > 0 :
			#원본 토큰들 중에서 랜덤하게 한 개의 토큰을 선택 
			random_word = new_words[random.randint(0, len(new_words)-1)]
			#선택한 토큰에 대하여 유의어 list를 구함. 
			synonyms = get_synonyms(random_word)
			counter += 1

		if counter >= 10: #10번의 기회동안 유의어를 찾지 못하면 넘어감. 
			return
	
	#유의어를 랜덤하게 골라 원본 토큰들에서 랜덤한 위치에 삽입함. 
	random_synonym = synonyms[random.randint(0, len(synonyms)-1)]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)


########################################################################
# Easy Data Augmentation

"""
random seed 고정
최종적으로 증강되는 데이터들 중 우연히 겹치는 데이터는 삭제하는 작업이 추가되어야 함. 
"""
########################################################################
komoran = Komoran()
def EDA(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	#문장에서 한글만 남김. 
	#내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는? -> 내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는?
	sentence = get_only_hangul(sentence)

	#띄어쓰기 단위로 분리 
	#내년까지 87만kw급 화력발전소 2기가 추가로 들어서는 도시는? -> '내년까지', '87만kw급', '화력발전소', '2기가', '추가로', '들어서는', '도시는?'
	# words = sentence.split(' ')
	words = komoran.morphs(sentence)
	print("words1",words)

	# 분리된 토큰 중에 "" 토큰은 없애기 
	#'위례', '아이파크', "1차'에서", '이용할', '수', '있는', '8호선', '역은?' -> '위례', '아이파크', "1차'에서", '이용할', '수', '있는', '8호선', '역은?'
	words = [word for word in words if word != ""]
	#print("words2",words)

	#띄어쓰기 단위로 나눠진 토큰 개수 
	num_words = len(words)

	augmented_sentences = []
	# SR, RI, RS, RD 4가지 기법에 대하여 공평하게 augmentation이 되도록 증강 횟수를 정함.  
	# num_aug=9이면 각 3개씩 증강함. 
	num_new_per_technique = int(num_aug/4) + 1
	#print(f"num_aug: {num_aug}, num_new_per_technique:{num_new_per_technique}")

	# 전체 토큰 개수에서 alpha_X의 비율만큼 가져왔을 때, 1과 비교하여 max 값 추출 
	# n_sr:1, n_ri:1, n_rs:1
	n_sr = max(1, int(alpha_sr*num_words))
	n_ri = max(1, int(alpha_ri*num_words))
	n_rs = max(1, int(alpha_rs*num_words))
	#print(f"n_sr:{n_sr}, n_ri:{n_ri}, n_rs:{n_rs}")

	# SR
	for _ in range(num_new_per_technique):
		a_words = synonym_replacement(words, n_sr)
		augmented_sentences.append(' '.join(a_words))

	# RI
	for _ in range(num_new_per_technique):
		a_words = random_insertion(words, n_ri)
		augmented_sentences.append(' '.join(a_words))

	# RS
	for _ in range(num_new_per_technique):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(" ".join(a_words))

	# RD
	for _ in range(num_new_per_technique):
		a_words = random_deletion(words, p_rd)
		augmented_sentences.append(" ".join(a_words))

	#증강된 문장에서 한글만 남기기 
	augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	#증강된 데이터데 대하여 무작위로 순서 섞기 
	#random.shuffle(augmented_sentences)

	# if num_aug >= 1:
	# 	augmented_sentences = augmented_sentences[:num_aug]
	# else: 
	# 	keep_prob = num_aug / len(augmented_sentences)
	# 	augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#원본 문장도 포함함
	#augmented_sentences.append(sentence)

	return augmented_sentences
