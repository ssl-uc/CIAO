import re
import r2pipe
import pickle
import gzip
from collections import defaultdict

import torch
from torch_geometric.data import Data
import r2pipe
import numpy as np

'''Normalize ESIL Tokens'''
def normalize_esil_token(token, op_type = ""):


    # Check if it's a call-like instruction
    is_call = op_type.lower() == "call"

    # Temporary variables
    if token.startswith("$"):
        return "<TMP>"

    # 1. FUNCTION POINTERS (called addresses)
    if is_call and (token.startswith("0x") or token.lstrip("-").isdigit()):
        return "<FUNC>"

    if re.fullmatch(r"-?\d+", token):
        return "<DATA>"
    
    # 2. IMMEDIATE VALUES
    if re.match(r"^0x[fF]{5,}", token):  # 0xffff... cases
        return "<IMM>"
    if re.match(r"^0x[0-9a-fA-F]{1,3}$", token):  # short hex constants
        return "<IMM>"

    # 3. MEMORY ADDRESSES
    if re.match(r"^0x[0-9a-fA-F]{4,}$", token):  # larger hex
        return "<MEM>"

    # 4. REGISTERS
    if re.match(r"^(r|w|v|a|s|t)\d+$", token):
        return "<REG>"
     
     

    # 5. leave as-is
    return token
   
   
'''Extract ESIl Tokens'''
def extract_esil_functions_normalized(binary_path, ops_path, func_path):

    r2 = r2pipe.open(binary_path, flags=["-2"])
    r2.cmd("aaa")
    functions = json.loads(r2.cmd("aflj"))
    esil_functions = []
    
    with open(func_path+binary_path.split('/')[-1], wb) as f:  # dump the function info for later use in the pipeline
        pickle.dump(functions, f)

    for func in functions:
        r2.cmd(f"s {func['offset']}")
        try:
            instructions = json.loads(r2.cmd("pdfj"))["ops"]
            
            with open(ops_path+binary_path.split('/')[-1], wb) as f: # dump all instrucions info for later use in the pipeline
                pickle.dump(instructions, f)
        except:
            continue
        instr_tokens = []

        for op in instructions:
            if "esil" not in op:
                continue
            op_type = op.get("type", "")  
            esil_expr = op["esil"]
            tokens = esil_expr.split(",")
            norm_tokens = [normalize_esil_token(t, op_type) for t in tokens]
            norm_tokens = [t for t in norm_tokens if t.strip()]

            if norm_tokens:  # only add non-empty token lists
                instr_tokens.append(norm_tokens)
        esil_functions.append(instr_tokens)

    r2.quit()
    return esil_functions
    

'''Dump ESIL Tokens for Word2vec Training. Binaries is a list of samples for which ESIL tokens will be extracted'''
def dump_instruction_level_chunks(binary_ops_path, binary_func_path, binaries=None, out_dir="esil_chunks_instr"):

    if binaries == None:
        print("Please give binaries list for extracting ESIL token.")

    all_functions = []


    for binary in binaries:
       
        funcs = extract_esil_functions_normalized(binary, binary_ops_path, binary_func_path)
        for func in funcs:
            if func:  # List of tokenized instructions
                all_functions.append(func)
       
        path = os.path.join(out_dir, f"{binary.split('/')[-1]}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"data": all_functions}, f)
        print(f"Saved chunk {chunk_id} with {len(all_functions)} functions")
        all_functions = []
        chunk_id += 1


'''Concatenate ESIL Token Embeddings for each function for transformer training'''
def extract_esil_vectors(instructions, word2vec, unk_vec):

    all_vecs = []

    for op in instructions:
        if "esil" not in op:
            continue
        op_type = op.get("type", "")
        esil_expr = op["esil"]
        raw_tokens = esil_expr.split(",")

        norm_tokens = [normalize_esil_token(t, op_type) for t in raw_tokens]
        norm_tokens = [t for t in norm_tokens if t.strip()]

        for token in norm_tokens:
            if token in word2vec.wv:
                vec = word2vec.wv[token].tolist()
            else:
                vec = unk_vec
            all_vecs.append(vec)

    return all_vecs if all_vecs else None
    

def normalize_arch(r2):
    info = r2.cmdj("ij")
    arch = info["bin"]["arch"]
    bits = info["bin"]["bits"]
    return f"{arch}{bits}"
    
'''Dump arch specific data for transformer encoder training'''    
def dump_arch_vector_datasets(binaries, binary_ops_path, binary_func_path, word2vec, unk_vec, out_dir): #todo add and modify pickle file for non stripped samples
    os.makedirs(out_dir, exist_ok=True)
    

    for binary in binaries:

        r2 = r2pipe.open(binary, flags=["-2"])
        r2.cmd("aaa")
        arch = normalize_arch(r2)
        r2.quit()
        
        with open(binary_func_path+binary.split('/')[-1], 'rb') as f:
            funcs = pickle.load(f)
        
        with open(binary_ops_path+binary.split('/')[-1], 'rb') as f:
            instructions = pickle.load(f)


        filtered = [f for f in funcs if f.get("name", "").startswith("sym.")]
        function_dataset = []

        for func in filtered:
            vecs = extract_esil_vectors(instructions, word2vec, unk_vec)
            if vecs:
                function_dataset.append({
                    "label": func["name"].replace("sym.", ""),
                    "arch": arch,
                    "tokens": vecs
                })

        # Save each arch-specific dataset
        os.makedirs(out_dir+'/'+arch, exist_ok=True)
        with gzip.open(out_dir+'/'+arch+'/'+binary.split('/')[-1]+".pkl.gzip", "wb") as f:
            pickle.dump(function_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(function_dataset)} functions")

'''Transformer data will be evry large and loading them during training will be slow. Build index of different functions to load durinng training quickly'''
def build_path_metadata_index(data_dir, func_name_map, output_index_path): 
    label_map = defaultdict(list)

    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        try:
            with gzip.open(full_path, "rb") as f:
                funcs = pickle.load(f)
                for i, func in enumerate(funcs):
                    label = func_name_map[func.get("label")]
                    
                    if label:
                        label_map[label].append({
                            "path": full_path,
                            "index": i  # function index within file
                        })
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    with gzip.open(output_index_path, "wb") as f:
        pickle.dump(dict(label_map), f)

    print(f"Saved index to {output_index_path}")

'''Function Embedding for Graph Node'''
def extract_function_vectors_from_opdum(funcs, word2vec, model, ops_all):

    function_vectors = {}

    for func in funcs:
        addr = func['offset']
        ops =  ops_all[addr]

        all_vecs = []
       
        for op in ops:
            if "esil" not in op:
                continue
            op_type = op.get("type", "")
            esil_expr = op["esil"]
            raw_tokens = esil_expr.split(",")

            norm_tokens = [esil_utils.normalize_esil_token(t, op_type) for t in raw_tokens]
            norm_tokens = [t for t in norm_tokens if t.strip()]

            for token in norm_tokens:
                if token in word2vec.wv:
                    vec = word2vec.wv[token].tolist()
                else:
                    vec = unk_vec
                all_vecs.append(vec)
       
        vec = esil_utils.get_function_embedding(all_vecs, model)
        function_vectors[hex(addr)] = vec

    return function_vectors





def get_function_embedding(instruction_token_lists, model):
    """
    Compute encoded function embedding safely.
    Handles empty functions, short functions, and variable lengths.
    """
   
    with torch.no_grad():
        # Handle empty or None
        if not instruction_token_lists:  
            func_vec = torch.zeros((1, 1, 100), dtype=torch.float32, device='cuda')  # in_dim = 100
        else:
            func_tensor = torch.tensor(instruction_token_lists[:MAX_LEN], dtype=torch.float32)

            # Ensure correct shape (seq_len, embed_dim)
            if func_tensor.ndim == 1:
                # Single token, expand to (1, embed_dim)
                func_tensor = func_tensor.unsqueeze(0)

            # Add batch dimension → [1, seq_len, embed_dim]
            func_vec = func_tensor.unsqueeze(0).to('cuda')

        emb = model(func_vec)  # [1, model_dim]
   
    return emb.squeeze(0).cpu()  # Return [model_dim] tensor


def extract_function_vectors_dataset(funcs):

    function_vectors = {}

    for func in funcs:
        addr = func["offset"]
        r2.cmd(f"s {addr}")
        try:
            instructions = json.loads(r2.cmd("pdfj"))["ops"]
        except:
            continue
        instr_tokens = []

        for op in instructions:
            if "esil" not in op:
                continue
            op_type = op.get("type", "")  # like "call"
            esil_expr = op["esil"]
            tokens = esil_expr.split(",")
            norm_tokens = [normalize_esil_token(t, op_type) for t in tokens]
            norm_tokens = [t for t in norm_tokens if t.strip()]

            if norm_tokens:  # only add non-empty token lists
                instr_tokens.append(norm_tokens)
        
        vec = get_function_embedding(instr_tokens, model)
        function_vectors[hex(addr)] = vec

    return function_vectors    

'''Extract FCG and create dataset for GNN'''
def extract_fcg_graph(labels, out_folder):


    for s in binaries:
        out_dir = out_folder + s.split('/')[-3] + '/' + s.split('/')[-1]
        r2 = r2pipe.open(s, flags=["-2"])
        r2.cmd('aaa')

        funcs = r2.cmdj('aflj')

        label = labels[s.split('/')[-3].split('_')[-1]]
        

        func_addr_to_idx = {f['offset']: idx for idx, f in enumerate(funcs)}
        node_features = []
        edges = []

        # get function features
        function_vectors = extract_function_vectors_dataset(funcs)
        for f in funcs:   
            node_features.append(np.array(function_vectors[hex(f["offset"])]))

            r2.cmd(f"s {f['offset']}")
            disasm = r2.cmdj('pdfj') or {}
            ops = disasm.get("ops", [])
            for op in ops:
                # for ARM
                if op.get("type") == "call" and "jump" in op:
                    dst = op["jump"]
                    if dst in func_addr_to_idx:
                        edges.append([func_addr_to_idx[f['offset']], func_addr_to_idx[dst]])
                # for MIPS
                elif op.get("type") == "rcall":
                # Check if references exist for the indirect jump target
                    refs = op.get("refs", [])
                    for ref in refs:
                        if ref["type"] == "CALL":
                            dst = ref["addr"]
                            if dst in func_addr_to_idx:
                                edges.append([func_addr_to_idx[f['offset']], func_addr_to_idx[dst]])

        if not edges:
            print(f'Error: {s}') #debug for binaries without edges
            return None

        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        torch.save(data, save_dir)

'''Extract FCG and create dataset for GNN'''
def extract_fcg_graph_from_opdum(label_key_to_family, op_dump_folder, func_dump_folder, label, word2vec, model, new_save):



    with open(func_dump_folder+label_key_to_family[label]+'/'+new_save.split('/')[-1], 'rb') as f:
        funcs = pickle.load(f)

    with open(op_dump_folder+label_key_to_family[label]+'/'+new_save.split('/')[-1], 'rb') as f:
        ops = pickle.load(f)
        
    func_addr_to_idx = {f['offset']: idx for idx, f in enumerate(funcs)}

    node_features = []
    edges = []
    
    # get function features
    function_vectors = extract_function_vectors_from_opdum(funcs, word2vec, model, ops)
    for f in funcs:  
        node_features.append(np.array(function_vectors[hex(f["offset"])]))
        
        addr = f['offset']
        ops =  ops_all[addr]
        for op in ops:
            # for ARM
            if op.get("type") == "call" and "jump" in op:
                dst = op["jump"]
                if dst in func_addr_to_idx:
                    edges.append([func_addr_to_idx[f['offset']], func_addr_to_idx[dst]])
            # for MIPS
            elif op.get("type") == "rcall":
            # Check if references exist for the indirect jump target
                refs = op.get("refs", [])
                for ref in refs:
                    if ref["type"] == "CALL":
                        dst = ref["addr"]
                        if dst in func_addr_to_idx:
                            edges.append([func_addr_to_idx[f['offset']], func_addr_to_idx[dst]])
                            
    if not edges:
        return None

                           

        
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    x = x.squeeze(1)  

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    torch.save(d, new_save)
        

