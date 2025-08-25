import re
import os
import pickle
from ast import literal_eval


import torch


from torch_geometric.data import Data

import numpy as np

import ESIL_DATA_EXTRACT.ESIL_utils as esil_utils
import function_embedder.train_transformer as tfs



def create_dataset(config, word2vec, unk_vec, op_dump_folder, func_dump_folder, binary_dir):



    MAX_LEN = config["MALGNN"]["transformer"]["MAX_LEN"]
    model = tfs.FunctionEncoder(
        model_dim=config["MALGNN"]["transformer"]["embedding_dim"],
        max_len=MAX_LEN,    
        num_layers=config["MALGNN"]["transformer"]["num_layers"],
        nhead=config["MALGNN"]["transformer"]["num_heads"],              
    )

    model.load_state_dict(torch.load(config["MALGNN"]["transformer"]["model_file"] + config["MALGNN"]["transformer"]["final_model_name"]))
    model.eval().cuda()

    label_key_to_family = literal_eval(config["MALGNN"]["label_key_to_family"])
    
    out_folder = config["MALGNN"]["fcg_data_dir"]

    labels = literal_eval(config["MALGNN"]["labels"]


    mirai_samples = [binary_dir + '/' +'mirai/'+i for i in os.listdir(binary_dir + '/' +'mirai/')]
    gafgyt_samples = [binary_dir + '/' +'gafgyt/'+i for i in os.listdir(binary_dir + '/' +gafgyt/')]
    tsunami_samples = [binary_dir + '/' +'tsunami/'+i for i in os.listdir(binary_dir + '/' +'tsunami/')]
    
    
    binaries = gafgyt_samples + mirai_samples + tsunami_samples
    
    for s in binaries:
        out_dir = out_folder + '/' + s.split('/')[-2] + '/' + s.split('/')[-1]   # dir structure: samples/familyname/filename

  
        label = labels[s.split('/')[-2]]
 
        esil_utils.extract_fcg_graph_from_opdum(label_key_to_family, op_dump_folder, func_dump_folder, label, word2vec, model, out_dir)
