from nltk.corpus import brown
import nltk
import sys

class languageModel:

	# smoothing_method = ''
	epsilon = sys.float_info.epsilon
	def __init__(self, corpus, ngram):
		self.corpus = corpus
		self.ngram = ngram

		self.word_analysis()

	def non_zero_divide(self, x, y):
		if x == 0 or y == 0:
			return self.epsilon
		return x/y

	def word_analysis(self):
		self.corpus = [i.lower() for i in self.corpus]

		self.ngram_fdist = []
		self.ngram_fdist.append(nltk.FreqDist(self.corpus))

		self.token_length = len(self.corpus)
		self.type_length = len(self.ngram_fdist[0])

		for i in range(1, self.ngram):
			self.ngram_fdist.append(nltk.FreqDist(
				nltk.ngrams(self.corpus, i+1)))

	def ngram_prob(self, word_gram, smoothing_method, parameter):
		grams_len = len(word_gram)
		# self.smoothing_method = smoothing_method
		if smoothing_method == 'naive':
			c_1 = tuple(word_gram)
			c_2 = tuple(word_gram[0:-1]) if grams_len > 2 else word_gram[0]
			return self.non_zero_divide(self.ngram_fdist[grams_len-1][c_1], self.ngram_fdist[grams_len-2][c_2])
		if smoothing_method == 'laplace':
			return self.laplace_smoothing(word_gram)
		if smoothing_method == 'add-k':
			return self.add_k_smoothing(word_gram,parameter)
		if smoothing_method == 'unigram-prior':
			return self.unigram_prior(word_gram,parameter)
		if smoothing_method == 'absolute-dis':
			return self.absolute_discounting(word_gram)
		if smoothing_method == 'kneser-ney':
			return self.KN_smoothing(word_gram)
	
	def unigram_prob(self,word):
		return (self.ngram_fdist[0][word] + 1)/(self.token_length+self.type_length)

	def add_k_smoothing(self,word_gram,k):
		grams_len = len(word_gram)
		c_1 = tuple(word_gram)
		c_2 = tuple(word_gram[0:-1]) if grams_len > 2 else word_gram[0]
		return (self.ngram_fdist[grams_len-1][c_1] + k)/(self.ngram_fdist[grams_len-2][c_2] + k*self.type_length)

	def absolute_discounting(self,word_gram):
		grams_len = len(word_gram)
		word_cur = word_gram[-1]
		words_before = word_gram[0:-1]

		c_1 = tuple(word_gram)
		c_2 = tuple(words_before) if grams_len > 2 else word_gram[0]
		gram_count = self.ngram_fdist[grams_len-1][c_1]
		words_before_count = self.ngram_fdist[grams_len-2][c_2]

		if gram_count == 0:
			d = 0
		elif gram_count == 1:
			d = 0.5
		else:
			d = 0.75

		word_cur_show_time = 0
		words_before_show_time = 0
		ngrams = self.ngram_fdist[grams_len - 1]

		for i in ngrams:
			if i[0:-1] == words_before:
				words_before_show_time += 1

		lambda_dis = self.non_zero_divide(d, words_before_count) * words_before_show_time
		P_w = self.unigram_prob(word_cur)

		P_asb = self.non_zero_divide(gram_count - d,words_before_count) + lambda_dis*P_w
		return P_asb




	def unigram_prior(self,word_gram,k):
		unigram_p = self.unigram_prob(word_gram[-1])
		grams_len = len(word_gram)
		c_1 = tuple(word_gram)
		c_2 = tuple(word_gram[0:-1]) if grams_len > 2 else word_gram[0]
		return (self.ngram_fdist[grams_len-1][c_1] + k*unigram_p)/(self.ngram_fdist[grams_len-2][c_2] + k)

	def laplace_smoothing(self,word_gram):
		grams_len = len(word_gram)
		c_1 = tuple(word_gram)
		c_2 = tuple(word_gram[0:-1]) if grams_len > 2 else word_gram[0]
		return (self.ngram_fdist[grams_len-1][c_1] + 0.1)/(self.ngram_fdist[grams_len-2][c_2] + 0.1*self.type_length)

	def KN_smoothing(self,word_gram):
		# Using the KN smoothing algorithm to generate the n-gram probability
		# self.prob_conti = {}
		grams_len = len(word_gram)
		word_cur = word_gram[-1]
		words_before = word_gram[0:-1]

		ngrams = self.ngram_fdist[grams_len - 1]
		ngram_num = len(ngrams)
		#Compute the prob of continuation
		word_cur_show_time = 0
		words_before_show_time = 0
		for i in ngrams:
			if i[-1] == word_cur:
				word_cur_show_time += 1
			if i[0:-1] == words_before:
				words_before_show_time += 1
		prob_conti =  word_cur_show_time/ngram_num

		word_gram_position = tuple(word_gram)
		words_before_position = tuple(words_before) if grams_len > 2 else word_gram[0]
		word_gram_count = self.ngram_fdist[grams_len-1][word_gram_position]
		words_before_count = self.ngram_fdist[grams_len-2][words_before_position]

		if words_before_count == 0:
			return self.epsilon

		prob_kn = (max(word_gram_count-0.75, 0) + (0.75*words_before_show_time*prob_conti))/words_before_count
		return prob_kn
