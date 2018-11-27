1. Run handle.py first
2. Run main.py
3. There're 3 places needed to change in the main.py if you want to change the smoothing method. Just search "LM.ngram_prob" in the main.py and you will find them.
  There are 6 smoothing methods:
    - "naive" for naive model
    - "laplace" for Laplace model
    - "add-k" for add-k model, and the following parameter is the k, you can change it freely
    - "unigram-prior" for unigram prior model. Also, you can change the following parameter.
    - "absolute-dis" for absolute discount smoothing
    - "kneser-ney" for kneser-ney model
4. You can find more details in the report
