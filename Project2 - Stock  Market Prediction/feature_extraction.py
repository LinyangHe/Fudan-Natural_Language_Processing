import numpy
import nltk

def naive_feature(dataset, Pos_id, Neg_id):
	return noise_eliminate(dataset, Pos_id, Neg_id, -1, -1, -1)

def noise_eliminate(dataset, Pos_id, Neg_id, remain_num1, remain_num2, ratio):
	positive_fdist = {}
	negative_fdist = {}
	
	for index, id in enumerate(Pos_id):
		if index % 1000 == 0 and index > 999:
			print(index)
		text = dataset[id]

		positive_fdist_title = nltk.FreqDist(text['title'])
		for w in positive_fdist_title:
			positive_fdist[w] = positive_fdist.get(w, 0) + 2*positive_fdist_title[w]

		positive_fdist_content = nltk.FreqDist(text['content'])
		for w in positive_fdist_content:
			positive_fdist[w] = positive_fdist.get(w, 0) + positive_fdist_content[w]


	for index, id in enumerate(Neg_id):
		if index % 1000 == 0 and index > 999:
			print(index)
		text = dataset[id]

		negative_fdist_title = nltk.FreqDist(text['title'])
		for w in negative_fdist_title:
			negative_fdist[w] = negative_fdist.get(w, 0) + 2*negative_fdist_title[w]

		negative_fdist_content = nltk.FreqDist(text['content'])
		for w in negative_fdist_content:
			negative_fdist[w] = negative_fdist.get(w, 0) + negative_fdist_content[w]
	

	if remain_num1 > 0:
		positive_fdist = dict(sorted(positive_fdist.items(), key=lambda e:e[1], reverse=True)[remain_num1:remain_num2])
		if ratio > 0:
			remain_num2 = int(remain_num1 + (remain_num2-remain_num1)*ratio)
		negative_fdist = dict(sorted(negative_fdist.items(), key=lambda e:e[1], reverse=True)[remain_num1:remain_num2])

	# print(len(positive_fdist))
	# print(len(negative_fdist))
	return positive_fdist, negative_fdist

def boolean_naive_feature(dataset, Pos_id, Neg_id):
	positive_fdist = {}
	negative_fdist = {}
	
	for index, id in enumerate(Pos_id):
		if index % 1000 == 0:
			print(index)
		text = dataset[id]
		# positive_fdist.update(nltk.FreqDist(text['content']))
		# positive_fdist.update(nltk.FreqDist(text['title']))
		# positive_fdist.update(nltk.FreqDist(text['title']))

	for index, id in enumerate(Neg_id):
		if index % 1000 == 0:
			print(index)
		text = dataset[id]
		# negative_fdist.update(nltk.FreqDist(text['content']))
		# negative_fdist.update(nltk.FreqDist(text['title']))
		# negative_fdist.update(nltk.FreqDist(text['title']))
	
	if remain_num > 0:
		positive_fdist = dict(sorted(positive_fdist.items(), key=lambda e:e[1], reverse=True)[:remain_num])
		negative_fdist = dict(sorted(negative_fdist.items(), key=lambda e:e[1], reverse=True)[:remain_num])

	return positive_fdist, negative_fdist


def boolean_noise_eliminate(Pos_id, Neg_id):
	pass

