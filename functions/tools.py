import numpy as np

# Returns uniform random distribution weight nxm array
def mlp1def(n, m):
    return 2*np.random.rand(n, m+1)-1

# Returns v state
def getV(x, w):
    v = np.ndarray(shape=(w.shape[0], x.shape[1]))
    for i in range(0, v.shape[1]):
        for j in range(0, v.shape[0]):
            v[j, i] = np.dot(np.transpose(w[j, :]),x[:, i])
    return v

# Returns sigmoid's value (considers as output) 
def sigmo(v):
    y = np.ndarray(shape=(v.shape[0], v.shape[1]) ) 
    for i in range (0, y.shape[1]):
        for j in range (0, y.shape[0]):
            y[j, i] = ( (1 - np.exp(-2 * v[j, i])) / (1 + np.exp(-2 * v[j, i])) )
    return y

# Samples and weight as input, returns sigmoid's value
def mlp1run(x, w):
    # Adding a column of 1 to represent the bias
    tmp = np.ones( (1, x.shape[1]) )
    # Now concatenate the line of 1 with the x matrix
    x = np.concatenate( (tmp, x), axis = 0)
    # Define V values
    v = getV(x, w)
    # Get sigmoid's values
    y = sigmo(v)
    return y

def getScore(y, samples_label):
    score = 0
    total = y.shape[1]
    class_predicted = []
    for i in range(0, y.shape[1]):
        class_predicted.append( np.argmax(y[:,i]))
        if class_predicted[i] == samples_label[i]:
            score = score + 1
    print(class_predicted)
    print(samples_label[0 : y.shape[1]])
    print('Score : ' + str(score) + '/' + str(total))
    return score / total

