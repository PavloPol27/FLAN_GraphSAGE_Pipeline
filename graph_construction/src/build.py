from tree_builder import buildGraphs
from utils import *
from dgl.data.utils import save_graphs
import time

import os
import pandas as pd

if __name__ == '__main__':
    # args
    args = parse_args()
    claims_list_transformed = data_prep(args.json_path)
    start_time = time.time()
    graphs, claim_level_infos, node_phrases = buildGraphs(claims_list_transformed, root2child=args.root2child)
    claim_info_dicts = {
            "app_num": torch.stack([a[0] for a in claim_level_infos]),\
            "app_cat": torch.stack([a[1] for a in claim_level_infos]),\
            "original_claim_idx": torch.stack([a[2] for a in claim_level_infos])
            } 
    save_graphs(args.graph_path, graphs, labels=claim_info_dicts)
    create_json(node_phrases, args.phrases_path)
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Execution Time: {execution_time} seconds")
    
    # Command to run (do not use relative paths!!!)
    # python build.py --json_path /bigstorage/pavlo/Qatent/FLAT_FLAN_Paragraphs/data/preranking/preranking_content_100.json --graph_path /bigstorage/pavlo/Qatent/graph_embeddings/graphs/preranking_100.dgl --phrases_path /bigstorage/pavlo/Qatent/FLAT_FLAN_Paragraphs/data/new_data/preranking_100.json --root2child > /bigstorage/pavlo/Qatent/FLAT_FLAN_Paragraphs/data/outputs/output_preranking_100.txt