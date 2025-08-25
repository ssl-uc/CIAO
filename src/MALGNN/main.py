import argparse
import json
import logging
from pathlib import Path
import os
import pickle
from gensim.models import Word2Vec


import ESIL_DATA_EXTRACT.ESIL_utils as esil_utils
import function_embedder.train_word2vec as wvec
import function_embedder.train_transformer as tfs
import create_dataset as cdt
import GNN.train_gnn as trg




# Helper: Detect project root automatically

def get_project_root():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "config").exists():  # config folder in repo root
            return parent
    return current_path.parents[2] 

# Main 

def main(binary_sample_dir, config, step):
    logging.basicConfig(level=logging.INFO)

    malgnn_cfg = config.get("MALGNN", {})
 
    
    
    
    '''Dump esil tokens for each binary for function embedder training. We will dump function and instruction info also for later use'''
    if step == "dump" or step == "all":
        esil_out_dir = Path(__file__).resolve().parent / malgnn_cfg.get("esil_out_dir")
        
        binary_ops_path = Path(__file__).resolve().parent / malgnn_cfg.get("ops_path")
        binary_func_path = Path(__file__).resolve().parent / malgnn_cfg.get("func_path")
        
        try:
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("esil_binary_file"), 'rb') as f:
                binaries = pickle.load(f)
        except:
            print("Please put the list of binaries contained in a pkl file in ESIL_DATA_EXTRACT/ folder")
            
        esil_utils.dump_instruction_level_chunks(binary_ops_path, binary_func_path, binaries=binaries,  out_dir=esil_out_dir)
        
        
     
    '''Train Word2Vec on the dumped esil token data'''   
    if step == "word2vec" or step == "all":
     
        chunk_dir = Path(__file__).resolve().parent / malgnn_cfg.get("esil_out_dir")
        
        vector_size, window, min_count, epochs = malgnn_cfg["word2vec"].get("vector_size"), malgnn_cfg["word2vec"].get("window"), malgnn_cfg["word2vec"].get("min_count"), malgnn_cfg["word2vec"].get("epochs")
        
        save_path = Path(__file__).resolve().parent / malgnn_cfg["word2vec"].get("model_file")
        
        
        wvec.train(chunk_dir, vector_size, windows, min_count, epochs, save_path)
        
        
    '''Dump data for transformer encoder training'''     
    if step == "transformer_data" or step == "all":
        try:
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("transformer_binary_file"), 'rb') as f:
                binaries = pickle.load(f)
        except:
            print("Please put the list of binaries (unstripped) contained in a pkl file in ESIL_DATA_EXTRACT/ folder")
        
        binary_ops_path = Path(__file__).resolve().parent / malgnn_cfg.get("ops_path")
        binary_func_path = Path(__file__).resolve().parent / malgnn_cfg.get("func_path")
        
        out_dir = Path(__file__).resolve().parent / malgnn_cfg.get("transformer_out_dir")
        
        word2vec = Word2Vec.load(Path(__file__).resolve().parent / malgnn_cfg["word2vec"]["model_file"])
        unk_vec = [0.0] * word2vec.vector_size 
        
        data_dir = out_dir + 'arm32' # create index data for one architecture, during training it will be fast to lookup there index, then load the file
        output_index_path = Path(__file__).resolve().parent / malgnn_cfg.get("transformer_index_file") # save output index folder
        
        with open(Path(__file__).resolve().parent / malgnn_cfg.get("func_name_map"),'rb') as f: # this is the file for mapping same functions to a common name 
            func_name_map =  pickle.load(f)
       
    
    
        esil_utils.dump_arch_vector_datasets(binaries, binary_ops_path, binary_func_path, word2vec, unk_vec, out_dir)
        
        
        esil_utils.build_path_metadata_index(data_dir, func_name_map, output_index_path)
        
        
    '''Train transformer encoder. We need labeled similar function pairs across archtectures'''    
    if step == "transformer" or step == "all":
    
        input_dim, model_dim, MAX_LEN, num_layers, nhead = malgnn_cfg["transformer"]["input_dim"], malgnn_cfg["transformer"]["embedding_dim"], malgnn_cfg["transformer"]["MAX_LEN"], malgnn_cfg["transformer"]["num_layers"], malgnn_cfg["transformer"]["num_heads"]
        arch1_index_path = Path(__file__).resolve().parent / malgnn_cfg.get("transformer_index_file")
        arch2_dir = Path(__file__).resolve().parent / malgnn_cfg.get("transformer_out_dir") + "mips32"
        
        
        with open(Path(__file__).resolve().parent / malgnn_cfg.get("func_name_map"),'rb') as f: # this is the file for mapping same functions to a common name 
            func_name_map =  pickle.load(f)
        
        max_per_label, max_step, device, epochs, lr = malgnn_cfg["transformer"]["max_per_label"], malgnn_cfg["transformer"]["max_step"], malgnn_cfg["transformer"]["device"], malgnn_cfg["transformer"]["epochs"], malgnn_cfg["transformer"]["learning_rate"]
        model_out = malgnn_cfg["transformer"]["model_file"]
        
        tfs.train(input_dim, model_dim, MAX_LEN, num_layers, nhead, arch1_index_path, arch2_dir, func_name_map, max_per_label, max_step, device, epochs, lr, model_out)
        
        
    '''Dump FCGs with node features for GNN training'''     
    if step == "dataset" or step == "all":
        
        
        word2vec = Word2Vec.load(Path(__file__).resolve().parent / malgnn_cfg["word2vec"]["model_file"])
        unk_vec = [0.0] * word2vec.vector_size 
        
        binary_ops_path = Path(__file__).resolve().parent / malgnn_cfg.get("ops_path")
        binary_func_path = Path(__file__).resolve().parent / malgnn_cfg.get("func_path")
        
        
        
        
        
        cdt.create_dataset(config, word2vec, unk_vec, binary_ops_path, binary_func_path, binary_sample_dir)

    '''Train and Test GNN+Classification Pipeline''' 
    if step == "gnn_train" or step == "all":
    
        data_dir = Path(__file__).resolve().parent / malgnn_cfg.get("fcg_data_dir")
        
        try:
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_arm_train"), 'rb') as f:
                train_arm_label = pickle.load(f)
                
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_mips_train"), 'rb') as f:
                train_mips_label = pickle.load(f)
                
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_arm_val"), 'rb') as f:
                val_arm_label = pickle.load(f)
                
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_mips_val"), 'rb') as f:
                val_mips_label = pickle.load(f)
                
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_arm_test"), 'rb') as f:
                test_arm_label = pickle.load(f)
                
            with open(Path(__file__).resolve().parent / malgnn_cfg.get("label_mips_test"), 'rb') as f:
                test_mips_label = pickle.load(f)
                
        except:
            print(f"Need pickle file in label directory containing train, test, validation labels for ARM and MIPS, i.e., FCG_DATASET")
            
        
        
        dataset, train_ds, val_ds, test_ds = trg.prepare_dataset(data_dir, train_arm_label, train_mips_label, val_arm_label, val_mips_label, test_arm_label, test_mips_label)
        
        model_dir = Path(__file__).resolve().parent / malgnn_cfg["gnn"]["model_dir"]
        batch_size, epochs = Path(__file__).resolve().parent / malgnn_cfg["gnn"]["batch_size"], model_dir = Path(__file__).resolve().parent / malgnn_cfg["gnn"]["epochs"]
        
        
        trg.run_training(dataset, train_ds, val_ds, test_ds, device_config, model_dir, batch_size, epochs)
        

if __name__ == "__main__":

    repo_root = get_project_root()
   
    config_dir = repo_root / "config"
    
    binary_sample_dir = repo_root / "samples" # todo add where needed. edit preevious code and modify the paths accordingly

    config_path = config_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
        
        
    parser = argparse.ArgumentParser(description="CIAO Pipeline")
    parser.add_argument("--step", type=str, help="all will execute everything, other steps are dump-dump esil tokens for selected binarie", required=True)
    args = parser.parse_args()
    main(binary_sample_dir, config, args.step)

