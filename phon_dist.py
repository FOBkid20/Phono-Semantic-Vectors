import numpy as np
import panphon.distance
import eng_to_ipa as ipa
import pandas as pd
import itertools

#read in
df = pd.read_csv('glove_no_punct.csv')
dst = panphon.distance.Distance()

#get first column

wordcol=df['V1']
wordlist = list(wordcol)
phonlist = list(map(ipa.convert, wordlist))
#list of unique pairs in phon, orth form

t=list(itertools.combinations(phonList,2))
orth=list(itertools.combinations(wordlist,2))

#phonetic distances
distances = list(itertools.starmap(dst.feature_edit_distance,t))

#semantic distances

dist_frame = df.drop('V1', axis =1 )
sem_mat = dist_frame.values.tolist()
s = list(itertools.combinations(sem_mat,2))
sem_dist = list(itertools.starmap(euc_dist,s))

#gather tuples into dataframes
phon_frame = pd.DataFrame(t,columns=["First","Second"])
ortho_frame = pd.DataFrame(orth,columns=["Ortho_one","Ortho_two"])
phon_frame['Dist']=distances

#concatinate into one data frame
orth_phon = [ortho_frame,phon_frame]
nf = pd.concat(orth_phon, axis=1)
nf['Semantic'] = sem_dist
nf.to_csv("semantic_phon.csv",index=False)

def euc_dist(x,y):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)
