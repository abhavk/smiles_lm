import numpy as np
import itertools as it
from collections import Counter
from vocab import Vocab
from model import LM
from utils import batch_iter, smi_tokenizer
import torch
import time
import math
import sys

def evaluate_ppl(model, dev_data, batch_size=4):
    """ Evaluate perplexity on dev sentences
    @param model : LM
    @param dev_data (list of sentences): list of tuples containing sentences
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def file_to_sentences(filepath):
    """ Convert file to input sentences (only below maximum size 50) that can be sent to create batches
    @param filepath: path of file to read
    @returns sent_list: list of sentences
    """
    with open(filepath) as file:
        sentences = file.readlines()
    sentence_smiles = [sentence.split()[0] for sentence in sentences]
    sent_list = list()
    for sentence in sentence_smiles:
        tokenized = smi_tokenizer(sentence)
        sentence_as_list = tokenized.split()
        if (len(sentence_as_list) <= 50):
            sent_list.append(sentence_as_list)
    return sent_list
    
# list object can be passed to utils.batch_iter

train_data = file_to_sentences('./001chem.src.txt')
dev_data = file_to_sentences('./dev.txt')

# Create vocab and model
vocab = Vocab.from_corpus(train_data)
model = LM(128, 128, vocab)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# constants
log_every = 100
clip_grad = 5.0
valid_niter = 100
model_save_path = './model.bin'
max_patience = 5
max_trials = 3
lr_decay = 0.5


epoch = train_iter = num_trial = 0

report_loss = cum_loss = 0
report_tgt_words = cum_tgt_words = report_examples = cum_examples = 0
valid_num = 0
hist_valid_scores = []
patience = 0

train_time = begin_time = time.time()
print('begin Maximum Likelihood training')
while (epoch<10):
    epoch += 1
    for src_sents, tgt_sents in batch_iter(train_data, batch_size=16):
        train_iter += 1
        optimizer.zero_grad()

        scores = model(src_sents, tgt_sents)
        batch_size = len(src_sents)
        
        example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
        batch_loss = example_losses.sum()
        loss = batch_loss / batch_size
        
        loss.backward()
        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        
        batch_losses_val = batch_loss.item()
        report_loss += batch_losses_val
        cum_loss += batch_losses_val
        
        tgt_words_num_to_predict = sum(len(s) for s in tgt_sents)  # omitting leading `<s>`
        report_tgt_words += tgt_words_num_to_predict
        cum_tgt_words += tgt_words_num_to_predict
        report_examples += batch_size
        cum_examples += batch_size
        
        if train_iter % log_every == 0:
            print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                  'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                     report_loss / report_examples,
                                                                                     math.exp(
                                                                                         report_loss / report_tgt_words),
                                                                                     cum_examples,
                                                                                     report_tgt_words / (
                                                                                                 time.time() - train_time),
                                                                                     time.time() - begin_time),
                  file=sys.stderr)

            train_time = time.time()
            report_loss = report_tgt_words = report_examples = 0.
            
        if train_iter % valid_niter == 0:
            print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(
                                                                                             cum_loss / cum_tgt_words),
                                                                                         cum_examples),
                  file=sys.stderr)
            
            cum_loss = cum_examples = cum_tgt_words = 0.
            valid_num += 1
            # compute dev. ppl and bleu
            dev_ppl = evaluate_ppl(model, dev_data)
            valid_metric = -dev_ppl

            print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)
                            
            if is_better:
                patience = 0
                print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                model.save(model_save_path)

                # also save the optimizers' state
                torch.save(optimizer.state_dict(), model_save_path + '.optim')
            elif patience < max_patience:
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)

                if patience == max_patience:
                    num_trial += 1
                    print('hit #%d trial' % num_trial, file=sys.stderr)
                    if num_trial == max_trials:
                        print('early stop!', file=sys.stderr)
                        exit(0)

                    # decay lr, and restore from previously best checkpoint
                    lr = optimizer.param_groups[0]['lr'] * lr_decay
                    print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                    # load model
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    model.load_state_dict(params['state_dict'])

                    print('restore parameters of the optimizers', file=sys.stderr)
                    optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # reset patience
                    patience = 0
    
    
