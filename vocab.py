from itertools import chain
from collections import Counter
from utils import pad_sents
import torch

class Vocab(object):
    def __init__(self):
        self.token2id = dict()
        self.token2id['<pad>'] = 0
        self.token2id['<s>'] = 1
        self.token2id['</s>'] = 2
        self.token2id['<unk>'] = 3
        self.unk_id = self.token2id['<unk>']
        self.id2token = {v: k for k, v in self.token2id.items()}
        
    def add(self, word):
        """ Add word/token to vocab, if it is previously unseen.
        @param token (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.token2id[word] = len(self)
            self.id2token[wid] = word
            return wid
        else:
            return self[word]   
    
    def __contains__(self, token):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained    
        """
        return token in self.token2id
        
    def __getitem__(self, token):
        """ Retrieve token's index. Return the index for the unk
        token if the token is out of vocabulary.
        @param token (str): token to look up.
        @returns index (int): index of token 
        """
        return self.token2id.get(token, self.unk_id)
        
    def __len__(self):
        """ Compute number of tokens in VocabEntry.
        @returns len (int): number of tokens in VocabEntry
        """
        return len(self.token2id)
        
    def tokens2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]
     
    def to_input_tensor(self, source)-> torch.Tensor:
        # @param source (List[List[str]]): list of source sentence tokens
        # @returns sents_var: tensor of (max_sentence_length, batch_size)
        token_ids = self.tokens2indices(source)
        sents_t = pad_sents(token_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long)
        return torch.t(sents_var)
        
     
    @staticmethod
    def from_corpus(corpus, freq_cutoff=1):
        """ Given a corpus construct a Vocab.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param freq_cutoff (int): if token occurs n < freq_cutoff times, drop the token
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry