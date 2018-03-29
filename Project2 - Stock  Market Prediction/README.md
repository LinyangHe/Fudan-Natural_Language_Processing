- 程序文件：

  1.classifer.py：分类器相关文件
  2.feature_extractio.py:提取特征
  3.prediction.py： 主文件
  4.word_segmatation.py： 分词程序

- 数据：

news_cut_dataset.txt：已经分好的词，采用pickle进行导入与读取。

- 使用：

如果没有分好的词文件，则先运行word_segmatation.py,否则直接运行主文件。