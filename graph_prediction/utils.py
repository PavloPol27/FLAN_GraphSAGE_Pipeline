import re
import torch
import json
import yaml
import random
from model import *
import numpy as np
import copy
from pytorch_metric_learning.losses import NTXentLoss


def open_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def load_config(config_file_path : str):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_batches(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches

def convert_category_to_string(numerical_representation):
    letter_num, number_num = numerical_representation
    letter_part = chr(letter_num + ord('A') - 1)
    number_part = str(int(number_num))
    return letter_part + number_part


def group_graphs_by_application_number(graphs, labels, citing_cited_mapping=None):
    app_nums = [str(int(number)) for number in labels['app_num']]
    app_cats = [convert_category_to_string(category) for category in labels['app_cat']]
    original_claim_numbers = [int(number) for number in labels['original_claim_idx']]
    grouped_graphs = {}

    # Create a dictionary to map application numbers to citing claims
    citing_claims_dict = {}
    if citing_cited_mapping:
        for mapping in citing_cited_mapping:
            citing_id = mapping[0]
            citing_claims = mapping[1]
            citing_claims_dict.setdefault(citing_id, set()).update(citing_claims)
        for citing_id in citing_claims_dict:
            citing_claims_dict[citing_id] = sorted(citing_claims_dict[citing_id])
    
    
    for graph, app_num, app_cat, claim_num in zip(graphs, app_nums, app_cats, original_claim_numbers):
        claim_key = f"c-en-{claim_num:04d}"
        app_num_cat_key = f"{app_num}{app_cat}"

        add_graph = False
        if citing_cited_mapping:
            if app_num_cat_key in citing_claims_dict and claim_key in citing_claims_dict[app_num_cat_key]:
                add_graph = True
        else:
            add_graph = True

        if add_graph:
            if app_num_cat_key not in grouped_graphs:
                grouped_graphs[app_num_cat_key] = {}
            if claim_key not in grouped_graphs[app_num_cat_key]:
                grouped_graphs[app_num_cat_key][claim_key] = []
            grouped_graphs[app_num_cat_key][claim_key].append(graph)

    return [{"Application_Number": app_num_cat_key, "Graphs": claims} for app_num_cat_key, claims in grouped_graphs.items()]


def adjust_train_test_split(citing_grouped, cited_grouped, mapping, test_size=0.2, val_size=0.1, train_size=0.7, random_state=42):
    citing_apps = {entry['Application_Number'] for entry in citing_grouped}
    cited_apps = {entry['Application_Number'] for entry in cited_grouped}

    random.seed(random_state)
    
    citing_docs = list(mapping.keys())
    random.shuffle(citing_docs)
    
    test_split_index = int(len(citing_docs) * (1 - test_size))
    val_split_index = int(test_split_index * (1 - val_size))
    train_split_index = int(len(citing_docs) * train_size)
    
    train_citing_docs = citing_docs[:train_split_index]
    val_citing_docs = citing_docs[train_split_index:test_split_index]
    test_citing_docs = citing_docs[test_split_index:]

    train_pairs = []
    val_pairs = []
    test_pairs = []
    train_pairs_set = set()
    val_pairs_set = set()
    test_pairs_set = set()
    test_citing = set()

    def create_pairs(citing_docs, pairs_list, pairs_set, test=False):
        for citing_num in citing_docs:
            if citing_num in mapping and citing_num in citing_apps:
                for cited_num in mapping[citing_num]:
                    if cited_num in cited_apps:
                        pair = (citing_num, cited_num)
                        if pair not in pairs_set:
                            if test:
                                test_citing.add(citing_num)
                            pairs_list.append(pair)
                            pairs_set.add(pair)

    create_pairs(train_citing_docs, train_pairs, train_pairs_set)
    create_pairs(val_citing_docs, val_pairs, val_pairs_set)
    create_pairs(test_citing_docs, test_pairs, test_pairs_set, test=True)

    return train_pairs, val_pairs, test_pairs


def make_pairs(citing_claim, citing_graphs, cited_claims_set, cited_graphs, other_cited_claims, hardest_graphs, hardest_claims, hardest_app_nums, cited_app):
    pos_pairs = []
    neg_pairs = []
    citing_graph = citing_graphs.get(citing_claim)

    if citing_graph:
        # Create positive pairs
        for cited_claim in cited_claims_set:
            if cited_claim in cited_graphs:
                cited_graph = cited_graphs[cited_claim]
                pos_pairs.append((cited_app, cited_claim, cited_graph[0]))
        
        # Create negative pairs
        for other_cited_claim in other_cited_claims:
            if other_cited_claim in cited_graphs:
                cited_graph = cited_graphs[other_cited_claim]
                neg_pairs.append((cited_app, other_cited_claim, cited_graph[0]))

        # Include negative pairs from hardest patents
        for hardest_graph, hardest_claim, hardest_app_num in zip(hardest_graphs, hardest_claims, hardest_app_nums):
            neg_pairs.append((hardest_app_num, hardest_claim, hardest_graph[0]))

    return citing_graph, pos_pairs, neg_pairs


def create_pairs_for_contrastive_learning(
        train,
        citing_grouped,
        cited_grouped,
        hardest_grouped,
        citing_cited_mapping,
        cited_paragraphs_claims_mapping,
        prerankings_pairs,
        seed=42,
        gold=False
    ):
    """
    Generates positive or negative pairs of graphs between citing and cited patents based on their application numbers, claims, and references. Also includes negative pairs from the most similar (hardest) patents that are not citations.

    Parameters:
    train (list): A list of tuples, each containing:
        - citing_application_number (str): The application number of the citing patent.
        - cited_application_number (str): The application number of the cited patent.
    citing_cited_mapping (list): A list of lists, where each sublist contains:
        - citing_id (str): The application number of the citing patent as a string (e.g., "3736084A1").
        - citing_claims (list): A list of citing claim identifiers (e.g., ["c-en-0001", "c-en-0002"]).
        - cited_id (str): The application number of the cited patent as a string (e.g., "2735402A1").
        - cited_refs (list): A list of cited reference identifiers (e.g., ["pa01", "p0041"]).
        - category (str): A category identifier (not used in the function, can be any string).
    cited_paragraphs_claims_mapping (dict): A dictionary mapping cited reference identifiers to their corresponding claims, structured as:
        { 'cited_patent_id': { 'cited_patent_id-ref_id': 'cited_patent_id-claim_id', ... }, ... }
    pair_type (str): The type of pairs to generate. Can be 'positive' or 'negative'.
    prerankings_pairs (dict, optional): A dictionary where each key is a citing application number and the value is the corresponding hardest (most similar) application number that is not a citation. Used to generate additional negative pairs.
    hardest_grouped (list, optional): A list of dictionaries, where each dictionary contains:
        - Application_Number (str): The application number.
        - Graphs (dict): A dictionary mapping claim identifiers to graphs for the hardest (most similar) applications.

    Returns:
    list: A list of tuples, where each tuple contains:
        - citing_graph: A graph from the citing patent.
        - cited_graph: A graph from the cited or hardest patent.
    """
    pairs = []
    citing_app_map = {entry['Application_Number']: entry['Graphs'] for entry in citing_grouped}
    cited_app_map = {entry['Application_Number']: entry['Graphs'] for entry in cited_grouped}
    hardest_app_map = {entry['Application_Number']: entry['Graphs'] for entry in hardest_grouped} if hardest_grouped else {}
    random.seed(seed)

    # Group claims by (citing_application_number, citing_claim)
    count = 0
    claim_pairs = {}
    gold_dict = {}
    for citing_application_number, cited_application_number in train:
        citing_graphs = citing_app_map[citing_application_number]
        cited_graphs = cited_app_map[cited_application_number]
        cited_claims_set = set()
        for citing_id, c_claims, cited_id, cited_refs, _ in citing_cited_mapping:
            if citing_id == citing_application_number and cited_id == cited_application_number:
                citing_claims = c_claims
                for ref in cited_refs:
                    if ref.startswith('c-en'):
                        cited_claims_set.add(ref)
                    ref_key = f"{cited_id}-{ref}"
                    if ref_key in cited_paragraphs_claims_mapping[cited_id]:
                        cited_claim = cited_paragraphs_claims_mapping[cited_id][ref_key].split('-')[-1]
                        cited_claims_set.add(f"c-en-{int(cited_claim):04d}")
                break

        cited_claims_set = sorted(cited_claims_set)
        other_cited_claims = sorted(set(cited_graphs.keys()) - set(cited_claims_set))
        hardest_graphs = []
        hardest_claims = []
        hardest_app_nums = []
        for citing_app, hardest_apps in prerankings_pairs.items():
            if citing_app == citing_application_number:
                for hardest_app in hardest_apps:
                    hardest_graph_dict = hardest_app_map.get(hardest_app, {})
                    if hardest_graph_dict:
                        hardest_claim = random.choice(list(hardest_graph_dict.keys()))
                        hardest_graph = hardest_graph_dict[hardest_claim]
                        hardest_graphs.append(hardest_graph)
                        hardest_claims.append(hardest_claim)
                        hardest_app_nums.append(hardest_app)
                break

        if not hardest_graphs:
            continue

        for citing_claim in citing_claims:
            key = (citing_application_number, citing_claim)
            if key not in claim_pairs:
                citing_graph, pos_pairs, neg_pairs = make_pairs(
                    citing_claim, citing_graphs, cited_claims_set, cited_graphs, other_cited_claims, hardest_graphs, hardest_claims, hardest_app_nums, cited_application_number
                )
                try:
                    claim_pairs[key] = ((citing_application_number, citing_claim, citing_graph[0]), pos_pairs, neg_pairs)
                except:
                    continue
            else:
                _, pos_pairs, neg_pairs = make_pairs(
                    citing_claim, citing_graphs, cited_claims_set, cited_graphs, other_cited_claims, hardest_graphs, hardest_claims, hardest_app_nums, cited_application_number
                )
                claim_pairs[key][1].extend(pos_pairs)
                claim_pairs[key][2].extend(neg_pairs)
            if gold:
                # Update gold dictionary
                gold_key = citing_application_number
                distinct_app_nums = set(app_num for app_num, claim, _ in pos_pairs)
                
                if gold_key not in gold_dict:
                    gold_dict[gold_key] = list(distinct_app_nums)
                else:
                    # Update the set to include only distinct app_num values
                    existing_app_nums = set(gold_dict[gold_key])
                    updated_app_nums = existing_app_nums.union(distinct_app_nums)
                    gold_dict[gold_key] = list(updated_app_nums)
    if gold:
        save_json(gold_dict, "/bigstorage/pavlo/Qatent/graph_embeddings/other_files/gold_1000.json")
    return list(claim_pairs.values())

def send_graph_to_device(g, device):
    g = g.to(device)
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g

def nt_xent_loss(embeddings, labels, temperature=0.5):
    loss_func = NTXentLoss(temperature=temperature)
    loss = loss_func(embeddings, labels)
    return loss

def create_labels_and_embeddings(model, pairs, device):
    embeddings = []
    labels = []
    label_counter = 0

    for graph, pos_pairs, neg_pairs in pairs:
        graph = send_graph_to_device(graph[2], device)
        node_features = graph.ndata['encoding']
        graph_embedding = model(graph, node_features).mean(dim=0).to(device)
        embeddings.append(graph_embedding)
        labels.append(label_counter)
        
        for pos_graph in pos_pairs:
            pos_graph = send_graph_to_device(pos_graph[2], device)
            pos_embedding = model(pos_graph, pos_graph.ndata['encoding']).mean(dim=0).to(device)
            embeddings.append(pos_embedding)
            labels.append(label_counter)
        
        label_counter += 1

        for neg_graph in neg_pairs:
            neg_graph = send_graph_to_device(neg_graph[2], device)
            neg_embedding = model(neg_graph, neg_graph.ndata['encoding']).mean(dim=0).to(device)
            embeddings.append(neg_embedding)
            labels.append(label_counter)
            label_counter += 1

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, device=device)
    return embeddings, labels

def train_epoch(model, optimizer, pairs, accumulation_steps, temperature, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(create_batches(pairs, accumulation_steps)):
        embeddings, labels = create_labels_and_embeddings(model, batch, device)
        loss = nt_xent_loss(embeddings, labels, temperature=temperature)
        
        if loss.requires_grad:
            loss.backward()
        else:
            print("Loss does not require grad")
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(pairs)
    return avg_loss

def evaluate_model(model, pairs, temperature, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in create_batches(pairs, 1):  # Assuming evaluation is done one batch at a time
            embeddings, labels = create_labels_and_embeddings(model, batch, device)
            loss = nt_xent_loss(embeddings, labels, temperature=temperature)
            total_loss += loss.item()
    avg_loss = total_loss / len(pairs)
    return avg_loss

def generate_embedding(model, graph, device='cuda'):
    model.eval()
    with torch.no_grad():
        graph = send_graph_to_device(graph, device)
        node_features = graph.ndata['encoding']
        embedding = model(graph, node_features).mean(dim=0)
    return embedding.cpu().numpy()


def compare_model_params(model, old_params):
    new_params = list(model.parameters())
    for old_param, new_param in zip(old_params, new_params):
        if not torch.equal(old_param, new_param):
            return True
    return False


# Function to rerank preranked documents based on claim embeddings
def rerank_documents(citing_graph, preranked_docs, model, index, preranking_claim_embeddings, device, top_k=100):
    preranking_embeddings = []
    keys = []
    for doc_id in preranked_docs:
        for key in preranking_claim_embeddings:
            if key[0] == doc_id:
                preranking_embeddings.append(preranking_claim_embeddings[key])
                keys.append(key)
    preranking_embeddings = np.array(preranking_embeddings)
    citing_graph = send_graph_to_device(citing_graph[2], device)
    query_embedding = generate_embedding(model, citing_graph)
    _, I = index.search(np.array([query_embedding]), len(preranking_embeddings))
    
    distinct_similar_document_keys = []
    seen = set()
    for idx in I[0]:
        formatted_key = keys[idx][0]  # Only use the document identifier
        if formatted_key not in seen:  # Ensure the document is not the same as the citing claim
            distinct_similar_document_keys.append(formatted_key)
            seen.add(formatted_key)
        if len(distinct_similar_document_keys) == top_k:
            break
            
    return distinct_similar_document_keys