#!/usr/bin/env python

import numpy as np
import random
import sys
import math

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    ex = np.exp(x * -1)
    s = 1/(1+ex)
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

#     print("bhenchooodd")
#     print('centerWordVec',centerWordVec)
#     print('centerWordVec.shape',centerWordVec.shape)
#      
#     print('outsideWordIdx',outsideWordIdx)
#     print('outsideVectors',outsideVectors)
#     print('outsideVectors.shape',outsideVectors.shape)
#     print('dataset',dataset)

    loss = 0.0
    gradCenterVec = np.zeros(centerWordVec.shape)
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    
#     print('gradCenterVec',gradCenterVec)
#     print('gradOutsideVecs',gradOutsideVecs)

    # asserting same size vectors
    assert(centerWordVec.shape[0] == outsideVectors.shape[1])
#     sys.exit()
    ### YOUR CODE HERE
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    
    cos_sim = outsideVectors.dot(centerWordVec)
#     print('cos_sim ',cos_sim )

    # softmax to get the word probablities
    word_prob = softmax(cos_sim)
#     print('word_prob',word_prob)
    # current outside word
    loss = -1 * math.log(word_prob[outsideWordIdx])
#     print('loss ',loss )
    
    prob_dist = outsideVectors * np.reshape(word_prob,(outsideVectors.shape[0],1))
#     print('prob_dist',prob_dist)
    
    gradCenterVec = (-1 * outsideVectors[outsideWordIdx] ) + np.sum(prob_dist,0)
    
    # outside vectors
    # preserve the current outside word vector
    currentOutsideVec = gradOutsideVecs[outsideWordIdx] 
    
    
    word_prob_col = word_prob.reshape(word_prob.shape[0],1)

#     print('gradOutsideVecs ',gradOutsideVecs.shape)
#     print('centerWordVec ',centerWordVec.shape)
#     print('word_prob ',word_prob.shape)
#     print('word_prob_col',word_prob_col.shape)
#     
#     print('centerWordVec ',centerWordVec)
#     print('word_prob_col',word_prob_col)
    
    gradOutsideVecs = centerWordVec * word_prob_col
    gradOutsideVecs[outsideWordIdx] = centerWordVec * (word_prob[outsideWordIdx]-1) 
    
#     sys.exit()
#     sys.exit()
#     gradCenterVec
#     print('gradCenterVec',gradCenterVec)

    ### END YOUR CODE
#     sys.exit()

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    loss = 0.0
    gradCenterVec = np.zeros(centerWordVec.shape)
    gradOutsideVecs = np.zeros(outsideVectors.shape)


    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices
#     print('negSampleWordIndices',negSampleWordIndices)
    unique_ids, count_ids = np.unique(negSampleWordIndices,return_counts= True)
#     print('unique',unique_ids)
#     print('unique',unique_ids.shape)
#     print('count ',count_ids )
#     print('count ',count_ids.shape )

#     print('negSampleWordIndices',negSampleWordIndices)
    ### YOUR CODE HERE

    ### Please use your implementation of sigmoid in here.
    
    uo_dot_vc = outsideVectors[outsideWordIdx].dot(centerWordVec)
   
#     uk_dot_vc = outsideVectors[negSampleWordIndices].dot(centerWordVec)
    uk_dot_vc = outsideVectors[unique_ids].dot(centerWordVec)
         
#     print('u_k_dot_v_c',uk_dot_vc)
    
    uk_dot_vc= uk_dot_vc * -1
#     print('u_k_dot_v_c',uk_dot_vc)
    sig_uk_vc = sigmoid(uk_dot_vc)
#     print('sig_uk_vc',sig_uk_vc)
    
    log_uk = np.log(sig_uk_vc)
#     print('log_uk',log_uk)
    
    log_uk = log_uk * count_ids
#     print('log_uk',log_uk)

    neg_sample = np.sum(log_uk)
#     print('neg_sample',neg_sample)
    
#     sys.exit()    
    
#     print('u_k_ids',u_k_ids.shape)
#     print('u_k_dot_v_c',uk_dot_vc.shape)
    sig_uo_dot_vc = sigmoid(uo_dot_vc)
    for_uo = np.log(sig_uo_dot_vc)
#     print('for_uo',for_uo)
    
    loss -= (for_uo + neg_sample)

    grad_vc_1 = (sig_uo_dot_vc-1) * outsideVectors[outsideWordIdx]
#     print('grad_vc_1',grad_vc_1)
    
    sig_uk_vc = 1- sig_uk_vc
#     print('sig_uk_vc',sig_uk_vc)
    
    sig_uk_vc_bin =   sig_uk_vc * count_ids
#     print('sig_uk_vc_re_bin',sig_uk_vc_bin)

    sig_uk_vc_re = sig_uk_vc_bin.reshape(sig_uk_vc_bin.shape[0],1)
#     print('sig_uk_vc_re',sig_uk_vc_re)
    
    
    grad_vc_2 = np.sum(outsideVectors[unique_ids] * sig_uk_vc_re,axis=0)
#     print('grad_vc_2',grad_vc_2)
       
    gradCenterVec = grad_vc_1 +  grad_vc_2
    
    
    # negativesample indexes has duplicate values. Need to take care of them
    # for grad center vectors we summed it over and hence it worked
    
    
    # outside words gradient
    gradOutsideVecs[unique_ids] = sig_uk_vc_re * centerWordVec
#     print('gradOutsideVecs',gradOutsideVecs)
    
    gradOutsideVecs[outsideWordIdx] = (sig_uo_dot_vc-1) * centerWordVec
#     print('gradOutsideVecs',gradOutsideVecs)
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
#     print("bhenchooodd")
#     print("currentCenterWord",currentCenterWord)
#     print("windowSize",windowSize)
#     print("outsideWords",outsideWords)
#     print("word2Ind",word2Ind)
#        
#     print("centerWordVectors",centerWordVectors)
#     print("outsideVectors",outsideVectors)
#     print("dataset",dataset)

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    gradCenterVec = np.zeros(centerWordVectors.shape[1])
#     print('gradCenterVec.shape', gradCenterVec.shape)

    ### YOUR CODE HERE
#     for outsideword in outsideWords:

#     centerWordVec,
#     outsideWordIdx,
#     outsideVectors,
#     dataset
    
#     print(centerWordVectors)
#     print(centerWordVectors[2,])
#     print(centerWordVectors[2])
#     print(word2Ind[currentCenterWord])

#     sys.exit()
    # the outside word vector gradients are particularly important to take care of
    # here the test wont show our bug. So be careful.

    # what about the outside vectors not in outside words?
    
    
    try:
        centerWordVec = centerWordVectors[word2Ind[currentCenterWord]]
#         print(centerWordVec)
    except:
        #return if center word vector not available
        print("Oops!",sys.exc_info()[0],"occured.")
        return    


    for outWord in outsideWords:

        out_loss,gradCenterVecOne,gradOutsideVecsOne = word2vecLossAndGradient(centerWordVec, word2Ind[outWord],outsideVectors,dataset)
        gradCenterVec += gradCenterVecOne
        gradOutsideVectors += gradOutsideVecsOne
        loss += out_loss
#         print('loss',loss)
#         print('out_loss',out_loss)
    
    ### END YOUR CODE
#     sys.exit()

    gradCenterVecs[word2Ind[currentCenterWord]] = gradCenterVec
    return loss, gradCenterVecs, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################
# first call:
#     gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#         word2vecModel = skipgram
#         word2Ind = dummy_tokens
#         wordVectors =  dummy_vectors
#         dataset = dataset
#         windowSize = 5
#         word2vecLossAndGradient = naiveSoftmaxLossAndGradient
          
        
def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """

    # dummy class declaration/initialization 
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
            
    # adding variables and methods to the dummy class is allowed!!
     
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

 
 
    # Random number generation isn't truly "random". It is deterministic, and the
    # sequence it generates is dictated by the seed value you pass into random.seed.
    # Typically you just invoke random.seed(), and it uses the current time as the
    # seed value, which means whenever you run the script you will get a different
    # sequence of values

    random.seed(31415)
    np.random.seed(9265)
    
    #     print(np.random.randn(10,3))
    # Each vector should have unit length. Divide by the total vector length.
    dummy_vectors = normalizeRows(np.random.randn(10,3))

    # this is our dummy corpus. words = strings do not matter
    
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
