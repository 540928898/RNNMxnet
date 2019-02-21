# -*- coding: utf-8 -*-

'''
Author: Gupeng

Time: 2019.2.21

Func: RNN Bucketing Realize

'''
import mxnet as mx
import numpy as np
import argparse
'''
Creat model sym_len
'''
def sym_gen(seq_len):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=len(vocab),output_dim=num_embed, name='embed')
    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)
    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')
    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return pred, ('data',), ('softmax_label',)  
    
    
def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    #read file 
    lines = open(fname).readlines()
    #split file by space and flitered them by function filter()
    lines = [filter(None, i.split(' ')) for i in lines]
    #achieve integer list from vocabulary list. Add unknown vocabulary to vocab
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label,
                                               start_label=start_label)
    return sentences, vocab    
    
    
buckets=[10,20,30,40,50,60]    
start_label=1
invalid_label=0
batch_size=50
train_sent,vocab=tokenize_text("./dataPTB/ptb.train.txt",
                               start_label=start_label,
                               invalid_label=invalid_label)

val_sent, _ = tokenize_text("./dataPTB/ptb.test.txt", 
                            vocab=vocab,
                            start_label=start_label,
                            invalid_label=invalid_label)                         
print(type(vocab),len(vocab))                              
print(type(train_sent),train_sent[:5])                               
data_train = mx.rnn.BucketSentenceIter(train_sent, 
                                       batch_size, 
                                       buckets=buckets,
                                       invalid_label=invalid_label)                               
                               
                               
data_val = mx.rnn.BucketSentenceIter(val_sent, 
                                     batch_size,
                                     buckets=buckets,
                                     invalid_label=invalid_label)
                               
                               
for _,i in enumerate(data_train):
    print(i.data[0][:2],i.label[0][:2])
    break                               
num_layers = 2
num_hidden = 200
num_embed =256
stack = mx.rnn.SequentialRNNCell()
for i in range(num_layers):
    stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_'%i))
a,_,_ = sym_gen(1)
#draw picture
mx.viz.plot_network(symbol=a)

if __name__=='__main__':
    import logging
    head='%(asctime)-15s %(message)s'
    logging.basicConfig(level =logging.DEBUG,format=head)
    model = mx.mod.BucketingModule(sym_gen=sym_gen,
                                   default_bucket_key=data_train.default_bucket_key,
                                   context=mx.cpu(0))
    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.Perplexity(invalid_label),
        kvstore             = 'device',
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate':0.01,
                                'momentum': 0.0,
                                'wd': 0.0001 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 2,
        batch_end_callback  = mx.callback.Speedometer(batch_size, 50, auto_reset=False))



















                               
                               