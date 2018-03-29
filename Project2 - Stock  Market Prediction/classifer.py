import numpy
import nltk
import jieba
import math 

class classifer(object):
	"""docstring for classifer"""

	def __init__(self, priors, features):
		# self.text_generator_list = text_generator_list
		self.priors = priors
		self.features = features
		self.token_lengths = [sum(features[0][0].values()), sum(features[1][0].values())]
		self.type_length = len(features[0][0]) + len(features[1][0])

	def classify(self, text_generator_list, method):
		if method == "naive_bayes_bootstrap":
			return self.__naive_bayes_bootstrap(text_generator_list, self.priors, self.features)
		if method == "naive_bayes_nltk":
			return self.__naive_bayes_nltk(self.features)
		pass

	def __naive_bayes_bootstrap(self, text_generator_list, priors, features):
		prior_pos = self.priors[0]
		prior_neg = self.priors[1]
		token_length_pos = self.token_lengths[0]
		token_length_neg = self.token_lengths[1]
		type_length = self.type_length

		# Compute the probability of positive and negative model
		prob_pos, prob_neg = prior_pos, prior_neg
		fdist_pos, fdist_neg = features[0][0], features[1][0]
		prob_deno_pos = token_length_pos + type_length
		prob_deno_neg = token_length_neg + type_length

		for text_generator in text_generator_list:
			for token in text_generator:
				try:
					prob_pos += math.log((fdist_pos[token] + 1)/prob_deno_pos)
				except KeyError:
					pass
				try:
					prob_neg += math.log((fdist_neg[token] + 1)/prob_deno_neg)
				except KeyError:
					pass

		# print(prob_pos, prob_neg)
		return 1 if prob_pos > prob_neg else -1

	def __svm(self):
		pass

	def __logistic_regression(self):
		pass
