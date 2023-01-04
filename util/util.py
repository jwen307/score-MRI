# -*- coding: utf-8 -*-

import argparse
import json
import pickle

def flags():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--load_ckpt_dir',
        type=str,
        default='None',
        help='Directory of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--ckpt_epoch',
        type=int,
        default=9,
        help='Epoch of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--ckpt_step',
        type=int,
        default=9,
        help='Step of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--train_inj',
        action='store_true',
        help='Train the injective portion of the network'
        )
    
    parser.add_argument(
        '--train_bij',
        action='store_true',
        help='Train the injective portion of the network'
        )
    
    return parser.parse_args()
   
def bijflags():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--load_ckpt_dir',
        type=str,
        default='None',
        help='Directory of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--ckpt_epoch',
        type=int,
        default=9,
        help='Epoch of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--ckpt_step',
        type=int,
        default=9,
        help='Step of the checkpoint that you want to load'
        )
    
    parser.add_argument(
        '--train_net',
        action='store_true',
        help='Train the the network'
        )

    return parser.parse_args()

# Read a json file
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
    
def read_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    
    return obj

def write_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    