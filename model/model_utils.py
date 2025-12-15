import numpy as np
import numpy.typing as npt

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    # Count frequency of a word in each sentence from list of sentences
    word_freq = {}
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word,0) + 1
    
    # Build vocabulary with words that appear >= 2 and add <UNK> for rare words (<2)
    vocab = ["<UNK>"] + [word for word in word_freq if word_freq[word] >= 2]
    word2idx  = {word: idx for idx,word in enumerate(vocab)} # word to index mapping

    # Initialize bag-of-words matrix
    V = len(vocab) # vocabulary size
    M = len(sentences) # number of sentences
    bow_matrix = np.zeros((V, M), dtype=int)

    # Fill the bag-of-words matrix, if the word is not in vocab, use <UNK>
    for col, sentence in enumerate(sentences):
        words = sentence.lower().split()
        for word in words:
            row = word2idx.get(word, 0)   # if the word is not found in vocab/wordcount less than 2, use index of <UNK>
            bow_matrix[row, col] += 1
    
    return bow_matrix
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    intent_list, unique_intents = data
    K =  len(unique_intents) # number of unique intents
    M = len(intent_list) # number of sentences
    
    # Map each unique intent to an index
    intent2idx = {intent: idx for idx, intent in enumerate(sorted(unique_intents))}

    # Initialize labels matrix
    labels = np.zeros((K, M), dtype=int)

    # Fill the labels matrix
    for col, intent in enumerate(intent_list):
        row = intent2idx[intent]
        labels[row, col] = 1

    return labels
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # Compute row-wise sum and broadcast the result in same dimensions as the original input
    ''' If inputs has large numbers, np.exp(inputs) heads in the direction of infinity.
        To avoid this, subtract the max from the inputs  '''
    exp_inputs = np.exp(z-np.max(z, axis = 0, keepdims= True))
    row_sums =np.sum(exp_inputs, axis = 0, keepdims = True)
    return exp_inputs/ row_sums
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # Compare each element with 0 and return the maximum, relu(x) = max(0,x)
    return np.maximum (0 , z) 
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # Derivative of ReLU is 1 for z > 0 else 0
    # Compare each element with 0, the output is True or False which is converted to 1.0 and 0.0 using astype(float)
    return (z > 0).astype(float)
    #########################################################################