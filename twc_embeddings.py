import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
#from dcpcse.models import RobertaForCL, BertForCL

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
#print(CURR_DIR)
sys.path.append(CURR_DIR)

from dcpcse.models import RobertaForCL, BertForCL
from scipy.spatial.distance import cosine
import pdb
import json

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]

class DCPCSEModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug  = False
        print("In DCPCSE Constructor")

    def init_model(self,model_name = None):
        # Load transformers' model checkpoint
        model_name = "models/large" if model_name is None else model_name
        args = construct_args()
        config = AutoConfig.from_pretrained(model_name)
    
        #if 'roberta' in args.model_name_or_path:
        self.model = RobertaForCL.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=args.cache_dir,
                revision=args.model_revision,
                use_auth_token=True if args.use_auth_token else None,
                model_args=args                  
            )

    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def compute_embeddings(self,input_file_name,input_data,is_file):
        texts = read_text(input_data) if is_file == True else input_data
        batch = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(self.device)
        
        # Get raw embeddings
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            pooler_output = outputs.pooler_output
            embeddings =  pooler_output.cpu()
        return texts,embeddings

    def output_results(self,output_file,texts,embeddings,main_index = 0):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_dict = {}
        #print("Total sentences",len(texts))
        for i in range(len(texts)):
                cosine_dict[texts[i]] = 1 - cosine(embeddings[main_index], embeddings[i])

        #print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict))
        return sorted_dict


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output.txt",
            help="Output file .")
    parser.add_argument("--input", type=str, default="small_test.txt",
            help="Input test file .")
    parser.add_argument("--model_name_or_path", type=str,default="models/large",
            help="Transformers' model name or path")
    parser.add_argument("--pooler_type", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--temp", type=float, 
            default=0.05, 
            help="Temperature for softmax.")
    parser.add_argument("--hard_negative_weight", type=float, 
            default=0.0, 
            help="The **logit** of weight for hard negatives (only effective if hard negatives are used).")
    parser.add_argument("--do_mlm", action='store_true', 
            help="Whether to use MLM auxiliary objective.")
    parser.add_argument("--mlm_weight", type=float, 
            default=0.1, 
            help="Weight for MLM auxiliary objective (only effective if --do_mlm).")
    parser.add_argument("--mlp_only_train", action='store_true', 
            help="Use MLP only during training")
    parser.add_argument("--pre_seq_len", type=int, 
            default=10, 
            help="The length of prompt")
    parser.add_argument("--prefix_projection", action='store_true', 
            help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", type=int, 
            default=512, 
            help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--cache_dir", type=str, 
            default=None,
            help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--model_revision", type=str, 
            default="main",
            help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", action='store_true', 
            help="Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models).")
    
    
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    
    
    args = parser.parse_args()
    return args
    

def main():
    args = construct_args()
    obj = DCPCSEModel()
    obj.init_model(args.model_name_or_path)
    texts,embeddings = obj.compute_embeddings(args.input,args.input,True)
    results = obj.output_results(args.output,texts,embeddings)
    



if __name__ == "__main__":
    main()
