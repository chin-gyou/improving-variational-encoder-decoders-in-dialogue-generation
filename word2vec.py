from gensim.models import Word2Vec
import numpy as np
import pickle

class word2vec:
    def __init__(self, vecfile):
        self.oov=0
        self.pretrained=Word2Vec.load_word2vec_format(vecfile,binary=True)

    def word_to_vec(self,word):
        if word in self.pretrained.vocab:
            return self.pretrained[word]
        else:
            self.oov+=1
            return np.random.normal(0, 1, 300)

    def embedding_matrix(self,dicts):
        embed=np.zeros(shape=(len(dicts)+1,300),dtype=np.float32)
        for w,i,_,_ in dicts:
            print(w)
            embed[i+1]=self.word_to_vec(w.lower())
        return embed

    def save_wordvecs(self,dicts,fout):
        embed=self.embedding_matrix(dicts)
        with open(fout,'wb') as f:
            pickle.dump(embed,f)

if __name__=='__main__':
    model=word2vec('GoogleNews-vectors-negative300.bin')
    with open('Training.dict.pkl','rb') as f:
        dicts=pickle.load(f)
        model.save_wordvecs(dicts,'embedding.mat')
        print(model.oov)

#    binarized_corpus = []
#    for line, dialogue in enumerate(open('./data/t_given_s_dialogue_length3_6.txt', 'r')):
#        dialogue = dialogue.replace('|', ' -1 ').strip().split() + ['-1'] #1 is the index of __eou__
#        binarized_dialogue = []
#        for word in dialogue:
 #           word_id = int(word)
 #           if word_id == 1:
#                word_id = 0
#            elif word_id == -1:
#                word_id = 1
#            else:
#                assert word_id > 1
#            binarized_dialogue.append(word_id)
#        binarized_corpus.append(binarized_dialogue)
   
#   with open('subtle.pkl', 'wb') as f:
#        pickle.dump(binarized_corpus,f)

