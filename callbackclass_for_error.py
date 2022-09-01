# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:40:39 2020

@author: Admin
"""


from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    "Callback to save model after every epoch"
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
    def on_epoch_end(self, model):
        output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        print("Save model to {}".format(output_path))
        model.save(output_path)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    "Callback to log information about training"
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.nbatch = 0
        #print("on_epoch_begin, similarity_human_man : ",model.wv.similarity('human', 'man') )
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch), " Nbatch #{}".format(self.nbatch))
        #print("on_epoch_end, similarity_human_man : ",model.wv.similarity('human', 'man') )
        self.epoch += 1
    def on_batch_begin(self, model):
        self.nbatch += 1
    


