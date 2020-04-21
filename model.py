import sys
from collections import namedtuple
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils


Hypothesis = namedtuple('Hypothesis', ['value', 'score','lstm_state'])

class LM(nn.Module):
    """ Language Model entity
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(LM, self).__init__()
        self.embeddings = nn.Embedding(len(vocab),embed_size, padding_idx=vocab['<pad>'])
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.LSTM =  torch.nn.LSTM(embed_size,hidden_size, num_layers=3, bias=True, bidirectional=False)
        self.vocab_projection = torch.nn.Linear(hidden_size, len(self.vocab), bias=False)
        
    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ @param source (List[List[str]]): list of source sentence tokens

        # @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                # log-likelihood of generating the gold-standard target tokens for
                                # each example in the input batch. Here b = batch size.
        # calculate log-likelihood of target given input
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]
        
        # size = (max_sentence_length (+1) , batch_size)
        source_padded = self.vocab.to_input_tensor(source)
        
        # size = (max_sentence_length (+1) , batch_size)
        target_padded = self.vocab.to_input_tensor(target)
        
        # size = (max_sentence_length, batch_size, embed_size)
        X = self.embeddings(source_padded)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, source_lengths)
        transient_vals, (last_hidden, last_cell) = self.LSTM(X)
        
        # src_len * batch_size * hidden_size
        vals = torch.nn.utils.rnn.pad_packed_sequence(transient_vals)[0]
       
        # reshape to b*sent_len*h
        # vals = vals.permute(1,0,2)
        
        # src_len * batch_size * vocab_size
        outs = self.vocab_projection(vals)
        P = F.log_softmax(self.vocab_projection(vals), dim=-1)
        
        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks
        scores = target_gold_words_log_prob.sum(dim=0)
        
        return scores
        
    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_next_steps: int = 5) -> List[Hypothesis]:
        """ Given a single source SMILE part, perform beam search, yielding LM hypotheses.
        @param src_sent (List[str]): a single source SMILE part (tokens)
        @param beam_size (int): beam size
        @param max_next_steps (int): maximum number of time steps to unroll the language model
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields, plus an extra containing other data:
                value: List[str]: the predicted token of maximum size len(src_sent) + max_next_steps
                score: float: the log-likelihood of the target SMILE
                lstm_state: extra data
                
        """
        
        current_hypotheses = self.sentence_to_hypes([src_sent], None,beam_size)
        
        for i in range(max_next_steps-1):
            new_hypes=[]
            for hyp in current_hypotheses:
                full_hyp = hyp[0]
                hyp_score = hyp[1]
                hyp_lstm_state = hyp[2]
                if not self.is_complete(full_hyp):
                    hyp_last = [[full_hyp[-1]]]
                    new_hypes += self.sentence_to_hypes(hyp_last, hyp,beam_size,hyp_lstm_state)
            current_hypotheses = self.select_best(current_hypotheses,new_hypes, beam_size)
        
        return [(src_sent+hyp[0], hyp[1], hyp[2]) for hyp in current_hypotheses]
    
    def select_best(self, current_hypotheses: List[Hypothesis],new_hypes: List[Hypothesis], beam_size)-> List[Hypothesis]:
        """ Select best beam_size hypotheses
        """
        all_hypes_to_compare = new_hypes.copy()
        completed_current = [ hyp for hyp in current_hypotheses if self.is_complete(hyp[0])]
        all_hypes_to_compare.extend(completed_current)
        
        indices = sorted(range(len(all_hypes_to_compare)), key=lambda i: all_hypes_to_compare[i][1])[-beam_size:]
        
        best_list= list()
        for i in reversed(indices):
            best_list.append(all_hypes_to_compare[i])
        return best_list            
    
    def is_complete(self, sent: List[str]):
        return sent[-1] == '</s>'
        
    def sentence_to_hypes(self, src : List[List[str]], hype_base: Hypothesis, beam_size, lstm_state=None):
        """takes in single sentence and basis for the hypothesis and returns the new hypotheses
        """
        
        # size = (max_sentence_length, batch_size, embed_size)
        X = self.embeddings(self.vocab.to_input_tensor(src))
        # src_len * batch_size * hidden_size
        vals = None
        new_lstm_state = None
        if lstm_state == None:
            vals, new_lstm_state = self.LSTM(X)
        else:
            vals, new_lstm_state = self.LSTM(X,lstm_state)
        # src_len * batch_size * vocab_size
        outs = self.vocab_projection(vals)
        P = F.log_softmax(outs, dim=-1)
        # take last layer of this - this will give the next predicted letter(s) in sequence
        # P_last has shape vocab_size
        P_last = P[-1,0,:]
        P_last_sorted, indices = torch.sort(P_last)
        for index in range(P_last_sorted.shape[0]):
            val = P_last_sorted[index].item()
            # print("probability of selecting token {} is {}".format(self.vocab.id2token[indices[index].item()], np.exp(val)))
        
        # now take beam_size most probable next tokens - first predicteds!
        hypotheses = []
        for i in range(beam_size):
            hype_next = None
            hype_score = 0
            if hype_base == None:
                hype_next = [self.vocab.id2token[indices[-(i+1)].item()]]
                hype_score = P_last_sorted[-(i+1)].item()
            else:                
                hype_next = hype_base[0].copy() + [self.vocab.id2token[indices[-(i+1)].item()]] # copy base value
                hype_score = hype_base[1] + P_last_sorted[-(i+1)].item()
            hypotheses.append(Hypothesis(hype_next, hype_score,new_lstm_state))
            
        return hypotheses

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = LM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)    
        
