import pickle
import jieba
import codecs
#Load file
f = codecs.open('news.txt','r','utf-8')
news_dataset = [eval(i) for i in f.readlines()]
f.close()

#Chinese words segmentation
news_cut = {}
for item in news_dataset:
	piece = {}
	id = item['id']
	piece['title'] = jieba.lcut(item['title'])
	piece['content'] = jieba.lcut(item['content'])
	news_cut[id] = piece

pickle.dump(news_cut, open('news_cut_dataset.txt', 'wb'))