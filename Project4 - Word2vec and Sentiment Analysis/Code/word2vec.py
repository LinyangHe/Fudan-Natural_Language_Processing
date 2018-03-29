import numpy as np
import random

from gradcheck import gradcheck_naive
from gradcheck import sigmoid  # sigmoid function can be used in the following codes


def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    """
    assert len(x.shape) <= 2
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    return np.divide(y, normalization)


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit
    # length

    # YOUR CODE HERE
    rowsnum = x.shape[0]
    x /= np.sqrt(np.sum(x**2, 1)).reshape(rowsnum, 1)
    # raise NotImplementedError
    # END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print()


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # YOUR CODE HERE
    prob_distribution = softmax(np.dot(predicted, outputVectors.T))
    cost = -np.log(prob_distribution[target])

    difference = prob_distribution

    target_vector = np.zeros((prob_distribution.shape))
    target_vector[target] = 1
    difference = difference - target_vector

    tokens_len = difference.shape[0]
    dimension = predicted.shape[0]

    grad = difference.reshape((tokens_len, 1)) * predicted.reshape((1, dimension))
    gradPred = (difference.reshape((1, tokens_len)).dot(outputVectors)).flatten()

    # raise NotImplementedError
    # END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient

    # YOUR CODE HERE
    gradPred = np.zeros(predicted.shape)
    grad = np.zeros(outputVectors.shape)

    new_index = dataset.sampleTokenIdx()
    indices = [target]
    # print(new_index)
    for k in range(K):
        new_index = dataset.sampleTokenIdx()
        while new_index == target:
            new_index = dataset.sampleTokenIdx()
        indices += [new_index]
    # print(indices)
    sampling_labels = np.array([1] + [-1 for k in range(K)])
    # print(sampling_labels)
    vec_out = outputVectors[indices, :]
    # print(vec_out)

    likelihood = sigmoid(vec_out.dot(predicted) * sampling_labels)
    difference = sampling_labels * (likelihood - 1)
    gradPred = np.dot(difference.reshape((1, K+1)),vec_out)
    gradPred = gradPred.flatten()
    tokens_len = predicted.shape[0]
    gradtemp = np.dot(difference.reshape((K+1, 1)),predicted.reshape(
        (1, tokens_len)))
    cost = -np.sum(np.log(likelihood))

    for k in range(K+1):
        grad[indices[k]] += gradtemp[k, :]

    likelihood = sigmoid(predicted.dot(outputVectors[target,:]))
    difference = likelihood - 1
    grad[target, :] += difference * predicted
    gradPred += difference * outputVectors[target, :]
    cost = -np.log(likelihood)


    for k in range(K):
        index = dataset.sampleTokenIdx()
        temp_likelihood = np.dot(-predicted,outputVectors[index,:])
        likelihood = sigmoid(temp_likelihood)

        cost += -np.log(likelihood)
        difference = 1 - likelihood

        grad[index, :] += difference * predicted
        gradPred += difference * outputVectors[index, :]
    # raise NotImplementedError

    # END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors

    # YOUR CODE HERE

    NIn = inputVectors.shape
    NOut = outputVectors.shape
    gradIn = np.zeros(NIn)
    gradOut = np.zeros(NOut)
    # print(NIn,NOut)

    curIndex = tokens[currentWord]
    predicted = inputVectors[curIndex, :]
    # print(predicted)
    cost = 0
    for word in contextWords:
        index = tokens[word]
        temp_cost, temp_gradIn, temp_gradOut = word2vecCostAndGradient(predicted, index, outputVectors, dataset)
        gradIn[currentI, :] += temp_gradIn
        gradOut += temp_gradOut
        cost += temp_cost

    # raise NotImplementedError
    # END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom
        # we use // to let N/2  be int
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    # print("\n==== Gradient check for CBOW      ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    # print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    # print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
