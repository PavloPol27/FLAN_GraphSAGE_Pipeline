import dgl
# import dgl.function as fn
import torch
import torch.nn as nn
import re
import os
import json
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    RandomSampler,
    SequentialSampler,
)
import argparse
import yaml
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_json(data, file_path):
    print("Saving phrases to {}".format(file_path))
    with open(file_path, 'w') as file:
        # Converting from list of dictionaries to JSON + indent=4 for readability
        json.dump(data, file, indent=4) 
        print("Phrases saved successfully to {}".format(file_path))

def replace_said_to_the(sentence):
    sentence = sentence.replace("said", "the")
    return sentence

def convert_app_category(app_category):
    # Extract the letter part and the numeric part
    letter_part = ''.join(filter(str.isalpha, app_category))
    number_part = ''.join(filter(str.isdigit, app_category))
    # Convert the letter part to a numerical position
    letter_num = ord(letter_part) - ord('A') + 1
    # Convert the number part to an integer
    number_num = int(number_part)
    return (letter_num, number_num)

def convert_category_to_string(numerical_representation):
    letter_num, number_num = numerical_representation
    letter_part = chr(letter_num + ord('A') - 1)
    number_part = str(int(number_num))
    return letter_part + number_part

def remove_claim_references(claim):
    pattern = r'\s(in accordance with|according to|as recited in|as claimed|as? set|as in|as defined).+[Cc]laims?[mtoandr\d\s\-]*,?'
    cleaned_claim = re.sub(pattern, custom_replacement, claim)
    pattern_with_of = r'\s(of any of the preceding|of any one of the preceding|of anyone of the preceding|of any one of|of any one of the preceding|of any one of the|of one of the|of any of|of)\s[Cc]laims?[mtoandr\d\s\-]*,?'
    if re.search(pattern_with_of, cleaned_claim):
        cleaned_claim = re.sub(pattern_with_of, custom_replacement, cleaned_claim)
    return cleaned_claim

def custom_replacement(match):
    # Check if the match ends with a comma
    if match.group(0).endswith(','):
        return ','
    elif match.group(0).endswith(' wherein') or match.group(0).endswith(' further'):
        return ', '
    else:
        return ''

def remove_references_to_features(line):
    pattern = re.compile(r'\s\(\s*[abivxlcdmA-Z]+\s*\)[\.,]?|\s\(\s*\d+\.?\d?[A-Za-z]*(?:[,;]\s*\d+\.?\d?[A-Za-z]*\s*)*\)[\.,]?|\(\s*[A-Z]\s*\)[\.,]?|\d+[A-Za-z]?\)[\.,]?|\s*\(\d+[A-Za-z]?')
    sentence = re.sub(pattern, '', line)
    sentence = re.sub(r'\s*\(\)', '', sentence)
    number_pattern = re.compile(r'(\sand|\sor)?\s?(\d+[\dA-Za-z]?|\[[A-Za-z]\])|\s?[A-Z]*\d+[A-Z]*|-->\s')
    sentence = re.sub(number_pattern, '', sentence)
    sentence = re.sub(r',{2,};?|,;', '', sentence)
    sentence = re.sub(r'^[^a-zA-Z]+', '', sentence)
    return sentence

def remove_figure_references(line):
    pattern = r'\b([Ff]igures?|[Ff][Ii][Gg][Ss]?\.?)\s*(\d+[A-Z]?(-\d+[A-Z]?)?(,\s*\d+[A-Z]?(-\d+[A-Z]?)?)*( and \d+[A-Z]?(-\d+[A-Z]?)?)?|[IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s*and\s*[IVXLCDM]+)?)(,\s*and\s*\d+)?'
    def replacement_function(match):
        if match.group(1).lower().startswith('figs')\
            or match.group(1).lower().startswith('figures'):
            return 'figures'
        return 'figure'
    return re.sub(pattern, replacement_function, line)

def clean_claim(claim):
    claim = remove_claim_references(claim)
    claim = remove_figure_references(claim)
    claim = remove_references_to_features(claim)
    claim = re.sub(r'\s+', ' ', claim).strip()
    claim = claim.lstrip(' ')
    return claim

def process_lines(data):
    all_sentences = []
    for entry in data:
        app_num = entry['Application_Number']
        app_category = entry['Application_Category']
        lines = entry.get('Content')
        lines = {k:v for k,v in lines.items() if k.startswith('c-en') if v and v != "-->"}  # Filter out only claims and non empty paragraphs; Removed: or re.match(r"p\d+", k)
        for line_id, line_text in lines.items():
            match = re.search(r'(\d+)$', line_id)  # Regex to find numbers at the end of the string
            if match:
                line_number = int(match.group())  # Convert number string to integer
                line_text = replace_said_to_the(line_text)
                # if not line_id.startswith('c-en'):
                #     line_text = remove_claim_references(line_text)
                all_sentences.append([line_text, app_num, app_category, line_number, 0 if line_id.startswith('c-en') else 1])
    return all_sentences

def data_prep(file_path):
    print("================== Running Data Preparation ==================")
    data = load_json(file_path)
    processed_lines = process_lines(data)
    df = pd.DataFrame(processed_lines, columns=['line_input', 'applicationNumber', "application_category", 'line_id', "line_type"])
    print(len(set(df.applicationNumber + df.application_category)))
    app_cats = [convert_app_category(app_cat) for app_cat in df.application_category]
    
    
    print("==> Finished reading json")
    print("==> size (#claims) : {}".format(len(df[df["line_type"] == 0])))
    print("==> size (#paragraphs) : {}".format(len(df[df["line_type"] == 1])))
    
    app_nums = torch.tensor([int(app_num) for app_num in df.applicationNumber], dtype=torch.int64)
    app_categories = torch.tensor(app_cats, dtype=torch.int64)
    line_idx = torch.tensor(df.line_id.values, dtype=torch.int64)
    line_types = torch.tensor(df.line_type.values, dtype=torch.int64)
    return group_by_app_num(
            zip(
                df["line_input"],
                app_nums,
                app_categories,
                line_idx,
                line_types
            )
        )

def group_by_app_num(list_of_tuples):
    lastAppNum = 0
    ans = []
    cur = []
    for line_text, app_num, app_cat, line_idx, line_types in list_of_tuples:
        if lastAppNum != 0 and lastAppNum != app_num:
            ans.append(cur)
            cur = []
        lastAppNum = app_num
        cur.append([line_text, app_num, app_cat, line_idx, line_types])
    ans.append(cur)
    return ans

def group_edge_type(edge_type):
    """
    Group edge types into 8 categories, based on the edge_type.
    """
    d = {
        1:0,
        2:1,
        3:2,

        4:8,
        5:9,

        10:3,
        11:3,
        12:3,
        13:3,
        14:3,
        15:3,
        16:3,
        17:3,
        18:3,

        20:4,
        21:4,
        22:4,

        30:5,

        40:6,
        41:6,

        50:7
    }
    return d[edge_type]
    # return 0

def convert_dgl_to_networkx(dgl_graphs, phrases_list, output_dir="../../graph_storage"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, dgl_graph in enumerate(dgl_graphs):
        # Convert DGL graph to NetworkX graph
        nx_graph = dgl_graph.to_networkx(node_attrs=["encoding", "depth"])

        # Visualize the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(nx_graph)  # Layout for the nodes

        # Extracting labels from the corresponding phrases
        phrases = phrases_list[i]
        node_labels = {}
        
        for n in nx_graph.nodes():
            label = phrases.get(n, "")
            # Wrap the text for better readability
            wrapped_label = "\n".join(textwrap.wrap(label, width=40))
            node_labels[n] = wrapped_label

        # Draw the network with labels
        nx.draw(nx_graph, pos, labels=node_labels, with_labels=True, node_color='skyblue', edge_color='#FF5733', font_size=8, node_size=500)

        plt.title(f"Graph Visualization with Node Phrases (Graph {i+1})")
        plt.savefig(f"{output_dir}/graph_{i+1}.png")
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--graph_path', type=str)
    parser.add_argument('--phrases_path', type=str)
    parser.add_argument('--root2child', dest='root2child', action='store_true')
    parser.add_argument('--no-root2child', dest='root2child', action='store_false')
    parser.set_defaults(root2child=True)
    #ROOT2CHILD
    #It can be toggled when calling buildGraphs, depending on whether the user wants
    #the edges in the graph to point from root nodes to child nodes (True) or the
    #opposite direction (False)
    return parser.parse_args()