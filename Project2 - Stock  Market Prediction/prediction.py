import nltk
import feature_extraction as FE
import classifer as CL
import pickle

news_cut = pickle.load(open('news_cut_dataset.txt','rb'))

#Fearture extraction using words bag for bootstrap
def featureExtraction_bootstrap():
	Pos_id = []
	Neg_id = []
	len_pos, len_neg = 0, 0
	with open('train.txt') as train_file:
		for line in train_file:
			line = line.strip().split('\t')
			if line[0] == '+1':
				len_pos += 1
				Pos_id.extend(int(i) for i in line[1].split(','))
			else:
				len_neg += 1
				Neg_id.extend(int(i) for i in line[1].split(','))
	priors = [len_pos / (len_pos + len_neg), len_neg / (len_pos + len_neg)]
	n1 = 0
	n2 = 100
	fdist_pos, fdist_neg = FE.noise_eliminate(news_cut, Pos_id, Neg_id, remain_num1= n1, remain_num2 = n2, ratio = -1)
	# fdist_pos, fdist_neg = FE.naive_feature(news_cut, Pos_id, Neg_id)
	features = [(fdist_pos, '+1'), (fdist_neg, '-1')]
	return priors, features

# Classification using bootstrap
def Classify_bootstrap(priors, features):
	myClassifer = CL.classifer(priors, features)

	result_file = open('result.txt','w')
	with open('test.txt') as test_file:
		for line in test_file:
			line = line.strip().split('\t')
			text_generator_list = []
			ids = [int(i) for i in line[1].split(',')]
			for id in ids:
				text_generator_list.append(news_cut[id]['title'])
				text_generator_list.append(news_cut[id]['title'])
				text_generator_list.append(news_cut[id]['content'])

			predict = myClassifer.classify(text_generator_list, method = 'naive_bayes_bootstrap')
			if predict > 0:
				result_file.write('+1\n')
			else:
				result_file.write('-1\n')
	result_file.close()

#Fearture extraction using words bag for NLTK
def featureExtraction_nltk():
	features = []
	times = 0
	with open('train.txt') as train_file:
		for line in train_file:
			times += 1
			if times % 1000 == 0:
				print(times)
			Pos_id = []
			Neg_id = []
			line = line.strip().split('\t')
			if line[0] == '+1':
				Pos_id.extend(int(i) for i in line[1].split(','))
			else:
				Neg_id.extend(int(i) for i in line[1].split(','))
			# fdist_pos, fdist_neg = FE.naive_feature(news_cut, Pos_id, Neg_id)
			fdist_pos, fdist_neg = FE.noise_eliminate(news_cut, Pos_id, Neg_id,5,25,-1)

			fdist_test = dict(fdist_pos,**fdist_neg)
			features.append((fdist_test,line[0]))

	return features

# Classification using nltk 
def Classify_nltk(features):
	classifier = nltk.DecisionTreeClassifier.train(features)
	# classifier = nltk.NaiveBayesClassifier.train(features)
	# classifier = nltk.MaxentClassifier.train(features)


	# classifier.show_most_informative_features(50)
	result_file = open('result.txt','w')

	times = 0
	with open('test.txt') as test_file:
		for line in test_file:
			times += 1
			if times % 100 == 0:
				print(times)
			line = line.strip().split('\t')
			text_generator_list = []
			ids = [int(i) for i in line[1].split(',')]
			# classified_class = NLTK_classify(line[0],ids)
			if line[0] == '+1':
				Pos_id_test = ids
				Neg_id_test = []
			else:
				Pos_id_test = []
				Neg_id_test = ids
			fdist_pos_test, fdist_neg_test = FE.naive_feature(news_cut, Pos_id_test, Neg_id_test)
			fdist_test = dict(fdist_pos_test,**fdist_neg_test)
			classified_class = classifier.classify(fdist_test)

			result_file.write(classified_class+'\n')

	result_file.close()



if __name__ == '__main__':
	priors, features = featureExtraction_bootstrap()
	Classify_bootstrap(priors, features)

	# features = featureExtraction_nltk()
	# Classify_nltk(features)
