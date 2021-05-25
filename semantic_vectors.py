import numpy as np
import pickle
glove_file = './glove.6B/glove.6B.50d.txt'
vectors = {}
for idx, line in enumerate(open(glove_file)):
    row = line.split()
    word = row[0]
    vals = np.array([float(x) for x in row[1:]])
    vectors[word] = vals
f = open("semantic_vectors.pkl","wb")
pickle.dump(vectors,f)
f.close()