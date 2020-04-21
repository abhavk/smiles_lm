import numpy as np
import re
import math

def pad_sents(sents, pad_token):
    """ Pad list of sentences(SMILES) according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of SMILES, where each sentence
                                    is represented as a list of tokens
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of SMILES where SMILES shorter
        than the max length SMILES are padded out with the pad_token, such that
        each SMILES in the batch now has equal length.
    """
    sents_padded = []

    max_length = max([len(sentence) for sentence in sents])
    sents_padded = [sentence+(max_length-len(sentence))*[pad_token] for sentence in sents]

    return sents_padded
   
def batch_iter(data, batch_size=2, shuffle=True):
    """ Yield batches of source and target SMILES reverse sorted by length (largest to smallest).
    @param data (list of (sentences), i.e. list of SMILES): list of tokenized sentences (List[List[str]])
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e), reverse=True)
        src_sents = [['<s>'] + e for e in examples]
        tgt_sents = [e+ ['</s>'] for e in examples]
        
        yield src_sents, tgt_sents
        
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)
