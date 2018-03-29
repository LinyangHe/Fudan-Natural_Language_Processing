import numpy as np
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import gutenberg
import nltk
import math
import string
import language_model as lm
import noisy_channel as nc
import copy
import difflib
import sys

epsilon = sys.float_info.epsilon

num_gram = 2
step = num_gram - 1

print("Loading Data...")
# Corpus_words = [i.lower() for i in brown.words(categories='news')]
# Corpus_words_lower = [i.lower() for i in reuters.words()]
# Corpus_words = Corpus_words_lower[:50000]
Corpus_words = [i.lower() for i in reuters.words()[:100000]]

fdist = nltk.FreqDist(Corpus_words)
Corpus_words_len = len(Corpus_words)
Corpus_vocab_len = len(fdist)
nc.run()

nc_Index = nc.Index
nc_Sub_prob = nc.Sub_prob[:28, :28]
nc_Del_prob = nc.Del_prob[:28, :28]
nc_Ins_prob = nc.Ins_prob[:28, :28]
nc_Rev_prob = nc.Rev_prob[:28, :28]
suffixes = {"'s": 2, "n't": 3, "'ll": 3,
			',': 1, '.': 1, "'ve": 3, "'re": 3, "'d": 2}

LM = lm.languageModel(corpus=Corpus_words, ngram=num_gram)
# LM = lm.languageModel(corpus=brown.words(categories = "news"), ngram=num_gram)
LM_ngram_fdist = LM.ngram_fdist

Vocab = {}
with open('vocab.txt', 'r') as vocab_file:
	for line in vocab_file:
		Vocab[line.strip().lower()] = 0
Vocab_len = len(Vocab)

test_data = []
with open('testdata.txt', 'r') as test_file:
	for line in test_file:
		test_data.append(line.strip().split('\t'))

def candidate_generator(word, sub_flag):
	word_length = len(word) + 1
	candidate_words = {}
	subcandidate_words = {}
	if word in Vocab:
		candidate_words[word] = word_prob(word)
	for i in nc_Index:
		for j in range(word_length):
			# Delete
			new_word1 = word[0:j] + i + word[j:]
			if new_word1 in Vocab and new_word1 not in candidate_words:
				P_w = word_prob(new_word1)
				new_word1_ = '@' + new_word1
				index_X = nc_Index[new_word1_[j]]
				index_Y = nc_Index[i]
				prob = product(nc_Del_prob[index_X, index_Y], P_w)
				candidate_words[new_word1] = float(prob)

			# Insert
			new_word2 = word[0:j] + word[j + 1:]
			if new_word2 in Vocab and new_word2 not in candidate_words:
				P_w = word_prob(new_word2)
				new_word2_ = '@' + new_word2
				index_X = nc_Index[new_word2_[j]]
				index_Y = nc_Index[word[j]]
				prob = product(nc_Ins_prob[index_X, index_Y], P_w)
				candidate_words[new_word2] = float(prob)

			# Substite
			new_word3 = word[0:j] + i + word[j + 1:]
			if new_word3 in Vocab and new_word3 not in candidate_words:
				P_w = word_prob(new_word3)

				index_X = nc_Index[word[j]]
				index_Y = nc_Index[i]
				prob = product(nc_Sub_prob[index_X, index_Y], P_w)
				candidate_words[new_word3] = float(prob)

			# Reversal
			new_word4 = word[0:j] + word[j + 1:j + 2] + word[j:j + 1] + word[j + 2:]
			if new_word4 in Vocab and new_word4 not in candidate_words:
				P_w = word_prob(new_word4)

				index_X = nc_Index[new_word4[j]]
				index_Y = nc_Index[new_word4[j + 1]]
				prob = product(nc_Rev_prob[index_X, index_Y], P_w)
				candidate_words[new_word4] = float(prob)

			if sub_flag:
				subcandidate_words[new_word1] = 0
				subcandidate_words[new_word2] = 0
				subcandidate_words[new_word3] = 0
				subcandidate_words[new_word4] = 0

	return candidate_words, subcandidate_words


def word_prob(word):
	return (fdist[word] + 0.1) / (Corpus_words_len + 0.1 * Vocab_len)


def product(a, b):
	return math.exp(math.log(a) + math.log(b))


def get_best_candidate(candidates, test_word_index, test_words, method):
	test_word_index_cp = copy.deepcopy(test_word_index)
	test_word_index_cp += step

	for candidate in candidates:
		word_gram_prob = 1
		test_words[test_word_index_cp] = candidate
		for i in range(test_word_index_cp - step, test_word_index_cp + 1):
			word_gram = [test_words[j] for j in range(i, i + num_gram)]
			if '<s>' in word_gram:
				continue
			word_gram_prob = LM.ngram_prob(word_gram, smoothing_method=method, parameter = 0.1)
			candidates[candidate] *= word_gram_prob
	best_candidate = sorted(candidates, key=lambda x: candidates[x])[-1]
	return best_candidate


def upper_lower(word, naive_word):
	if word.islower():
		return naive_word
	if word[0].isupper() and word[-1].isupper():
		return naive_word.upper()
	return naive_word.capitalize()


def include_number(word):
	for i in range(10):
		if str(i) in word:
			return True
	return False


def find_wrong_word(test_words):
	word_gram_probs = {}
	for i in range(len(test_words) - step):
		right_flag = 0
		word_gram = [test_words[j] for j in range(i, i + num_gram)]
		for k in word_gram:
			if include_number(k) or k in suffixes or '.' in k:
				right_flag = 1
		if right_flag:
			continue
		word_gram_probs[str(i)] = LM.ngram_prob(
			word_gram, smoothing_method='naive',  parameter = 0.1)
		# print(word_gram, word_gram_probs[str(i)])

	best_index = sorted(word_gram_probs, key=lambda x: word_gram_probs[x])[0]
	return int(best_index) + step


def main():
	# correct one
	result_file = open('result.txt', 'w')
	for data in test_data:

		print(data[0])
		correct_time = 0
		correct_times = int(data[1])
		test_words = nltk.tokenize.word_tokenize(data[2])
		test_words_dict = {i: 0 for i in test_words}
		test_words_copy = copy.deepcopy(test_words)
		affix = ['<s>'] * step
		test_words_copy.extend(affix)
		affix.extend(test_words_copy)
		test_words_copy = affix
		candidates = {}
		for test_word_index, test_word in enumerate(test_words):
			if not test_word:
				continue

			if "'" == test_word[0]:
				test_word_prefix = "'"
				test_word_root = test_word[1:].lower()
			else:
				test_word_prefix = ''
				test_word_root = test_word.lower()

			if test_word_root in Vocab:
				continue
			else:
				candidates, subcandidates = candidate_generator(test_word_root, 1)
				if not candidates:
					for j in subcandidates:
						candidates.update(candidate_generator(j, 0)[0])

				if candidates:
					best_candidate = get_best_candidate(
						candidates, test_word_index, test_words_copy, 'naive')
				else:
					best_candidate = test_word_root

				best_candidate = upper_lower(test_word, best_candidate)


				test_words[test_word_index] = test_word_prefix + best_candidate
				correct_time += 1

		while correct_time < correct_times:
			# No non-vocab error
			test_words_lower = [i.lower() for i in test_words]
			wrong_index = find_wrong_word(test_words_lower)
			wrong_word = test_words[wrong_index]

			if "'" == wrong_word[0]:
				wrong_word_prefix = "'"
				wrong_word_root = wrong_word[1:].lower()
			else:
				wrong_word_prefix = ''
				wrong_word_root = wrong_word.lower()

			candidates = candidate_generator(wrong_word_root, 0)[0]

			candidates[wrong_word] = 0
			best_candidate = get_best_candidate(candidates, wrong_index, test_words_copy, 'naive')

			best_candidate = upper_lower(wrong_word, best_candidate)
			test_words[wrong_index] = wrong_word_prefix + best_candidate
			correct_time += 1
			# print(wrong_word, best_candidate)

		for index, i in enumerate(test_words):
			if i in suffixes:
				continue
			test_words[index] = ' ' + i
		result_file.write(data[0] + '\t' + ''.join(test_words).strip() + '\n')


if __name__ == '__main__':
	main()
