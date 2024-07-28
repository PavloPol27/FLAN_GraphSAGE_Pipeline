import argparse
import torch
import dgl
import numpy as np
import faiss
from utils import load_config, open_json, group_graphs_by_application_number, adjust_train_test_split, create_pairs_for_contrastive_learning, train_epoch, evaluate_model, generate_embedding, compare_model_params, rerank_documents
from model import SAGE
from metrics import mean_recall_at_k, mean_ranking, mean_inv_ranking, mean_average_precision
import wandb
import copy

class GraphEmbeddingPipeline:
    def __init__(self, config_path, mapping_path, mapping_testing_path, claim_paragraph_mapping_path, prerankings_pairs_path,
                 citing_graphs_path, cited_graphs_path, hardest_graphs_path, preranking_graphs_path, preranking_test_path, gold_dict_path,
                 best_model_dir, existing_model_path=None, train=True, test=False):

        self.mapping_path = mapping_path
        self.mapping_testing_path = mapping_testing_path
        self.claim_paragraph_mapping_path = claim_paragraph_mapping_path
        self.prerankings_pairs_path = prerankings_pairs_path
        self.citing_graphs_path = citing_graphs_path
        self.cited_graphs_path = cited_graphs_path
        self.hardest_graphs_path = hardest_graphs_path
        self.preranking_graphs_path = preranking_graphs_path
        self.preranking_test_path = preranking_test_path
        self.gold_dict_path = gold_dict_path

        self.train = train
        self.test = test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_config(config_path)
        self.model = SAGE(
            self.config['x_size'],
            self.config['hidden_size'],
            self.config['gnn_layers'],
            self.config['aggregator_type'],
            self.config['dropout']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        self.initial_params = copy.deepcopy(list(self.model.parameters()))

        self.best_model_dir = best_model_dir
        self.best_model_path = self._get_best_model_path(existing_model_path)
        self.setup_wandb()

    def _get_best_model_path(self, existing_model_path):
        if existing_model_path:
            return existing_model_path
        else:
            identifier = time.strftime("%Y%m%d-%H%M%S")
            model_filename = f"best_model_{identifier}.pth"
            return os.path.join(self.best_model_dir, model_filename)
    
    def setup_wandb(self):
        wandb.init(
            project="GraphSageEmbeddings",
            config=self.config
        )

    def load_data(self):
        # Load all necessary data
        mapping = open_json(self.mapping_path)
        mapping_testing = open_json(self.mapping_testing_path)
        claim_paragraph_mapping = open_json(self.claim_paragraph_mapping_path)
        prerankings_pairs = open_json(self.prerankings_pairs_path)

        citing_graphs, citing_labels = dgl.load_graphs(self.citing_graphs_path)
        self.citing_grouped = group_graphs_by_application_number(citing_graphs, citing_labels, citing_cited_mapping=mapping)

        cited_graphs, cited_labels = dgl.load_graphs(self.cited_graphs_path)
        self.cited_grouped = group_graphs_by_application_number(cited_graphs, cited_labels)

        hardest_graphs, hardest_labels = dgl.load_graphs(self.hardest_graphs_path)
        self.hardest_grouped = group_graphs_by_application_number(hardest_graphs, hardest_labels)

        preranking_graphs, preranking_labels = dgl.load_graphs(self.preranking_graphs_path)
        self.preranking_grouped = group_graphs_by_application_number(preranking_graphs, preranking_labels)

        self.adjusted_train, self.adjusted_val, self.adjusted_test = adjust_train_test_split(
            self.citing_grouped, self.cited_grouped, mapping_testing, test_size=0.2, val_size=0.1, random_state=42
        )

        self.pairs_for_claims_training = create_pairs_for_contrastive_learning(self.adjusted_train, self.citing_grouped, self.cited_grouped, self.hardest_grouped, mapping, claim_paragraph_mapping, prerankings_pairs)
        self.pairs_for_claims_test = create_pairs_for_contrastive_learning(self.adjusted_test, self.citing_grouped, self.cited_grouped, self.hardest_grouped, mapping, claim_paragraph_mapping, prerankings_pairs)
        self.pairs_for_claims_validation = create_pairs_for_contrastive_learning(self.adjusted_val, self.citing_grouped, self.cited_grouped, self.hardest_grouped, mapping, claim_paragraph_mapping, prerankings_pairs)

    def train_model(self):
        if not self.train:
            return
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        early_stopping_patience = 5

        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch + 1}")
            avg_train_loss = train_epoch(self.model, self.optimizer, self.pairs_for_claims_training, 10, 1.0, self.device)
            avg_val_loss = evaluate_model(self.model, self.pairs_for_claims_validation, 1.0, self.device)
            print(f"Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
            wandb.log({"Training Loss": avg_train_loss, "Validation Loss": avg_val_loss})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Epoch {epoch + 1}: Validation loss improved. Model saved!")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print("Early stopping due to no improvement in validation loss.")
                    break

            self.initial_params = copy.deepcopy(list(self.model.parameters()))

    def test_model(self):
        if not self.test:
            return

        # Load the best model state
        self.model.load_state_dict(torch.load(self.best_model_path))
        preranking_claim_embeddings = {}

        # Generate embeddings for all preranking claims
        for app_dict in self.preranking_grouped:
            for claim_key, citing_graph in app_dict["Graphs"].items():
                citing_embedding = generate_embedding(self.model, citing_graph[0], device=self.device)
                preranking_claim_embeddings[(app_dict["Application_Number"], claim_key, citing_graph[0])] = citing_embedding

        # Load preranking test data and gold dictionary
        preranking_test = open_json(self.preranking_test_path)
        gold_dict = open_json(self.gold_dict_path)

        top_k = 100
        results = {}

        for citing_graph, _, _ in self.pairs_for_claims_test:
            preranked_docs = preranking_test.get(citing_graph[0], [])
            if preranked_docs:
                # Create a subset of embeddings for the current citing graph's preranked documents
                subset_embeddings = []
                subset_keys = []
                for doc_id in preranked_docs:
                    for key, embedding in preranking_claim_embeddings.items():
                        if key[0] == doc_id:
                            subset_embeddings.append(embedding)
                            subset_keys.append(key)

                if not subset_embeddings:
                    print(f"No embeddings found for the preranked documents of {citing_graph[0]}")
                    continue

                # Convert to numpy array
                subset_embeddings = np.array(subset_embeddings)
                dimension = subset_embeddings.shape[1]

                # Create FAISS index for the current subset
                index = faiss.IndexFlatL2(dimension)
                index.add(subset_embeddings)

                # Rerank documents using the newly created FAISS index
                similar_document_keys = rerank_documents(
                    citing_graph,
                    preranked_docs,
                    self.model,
                    index,
                    preranking_claim_embeddings,
                    self.device,
                    top_k=top_k
                )
                results[citing_graph[0]] = similar_document_keys

        self.evaluate_results(results, gold_dict)

    def evaluate_results(self, results, gold_dict):
        true_labels = []
        predicted_labels = []
        for app_num, similar_document_keys in results.items():
            if app_num in gold_dict:
                true_labels.append(gold_dict[app_num])
                predicted_labels.append(similar_document_keys)

        k_values = [1, 5, 10, 20, 50, 100]
        for k in k_values:
            recall_at_k = mean_recall_at_k(true_labels, predicted_labels, k)
            print(f"Recall at {k}: {recall_at_k:.4f}")

        mean_rank = mean_ranking(true_labels, predicted_labels)
        mean_inv_rank = mean_inv_ranking(true_labels, predicted_labels)
        map_score = mean_average_precision(true_labels, predicted_labels)

        print(f"Mean ranking: {mean_rank:.4f}")
        print(f"Mean inverse ranking: {mean_inv_rank:.4f}")
        print(f"Mean Average Precision (MAP): {map_score:.4f}")

    def run(self):
        self.load_data()
        self.train_model()
        self.test_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Embedding Pipeline")
    parser.add_argument('--config', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/config/config_sage.yaml", help='Path to the configuration file.')
    parser.add_argument('--mapping', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/construction/citation_train/Citation_Train.json", help='Path to the mapping JSON file.')
    parser.add_argument('--mapping_testing', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/mapping_citing_cited/Mappings_Citing_To_Cited_100.json", help='Path to the testing mapping JSON file.')
    parser.add_argument('--claim_paragraph_mapping', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/cited_claim_paragraph_mapping/cited_paragraph_claim_mapping_124.json", help='Path to the claim paragraph mapping JSON file.')
    parser.add_argument('--prerankings_pairs', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/easy_negatives_preranking/easy_negatives_preranking_pairs_100_10.json", help='Path to the prerankings pairs JSON file.')
    parser.add_argument('--citing_graphs', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/graphs/100_samples/citing_graphs_100.dgl", help='Path to the citing graphs DGL file.')
    parser.add_argument('--cited_graphs', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/graphs/100_samples/cited_graphs_124.dgl", help='Path to the cited graphs DGL file.')
    parser.add_argument('--hardest_graphs', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/graphs/100_samples/easy_negatives_graphs_100_top10.dgl", help='Path to the hardest graphs DGL file.')
    parser.add_argument('--preranking_graphs', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/graphs/100_samples/preranking_100.dgl", help='Path to the preranking graphs DGL file.')
    parser.add_argument('--preranking_test', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/preranking_for_test_set/preranking_test.json", help='Path to the preranking test JSON file.')
    parser.add_argument('--gold_dict', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/gold/gold.json", help='Path to the gold dictionary JSON file.')
    parser.add_argument('--best_model_dir', type=str, default="/bigstorage/pavlo/Qatent/FLAN/data/prediction/best_models", help='Directory to save the best model checkpoints.')
    parser.add_argument('--existing_model_path', type=str, help='Path to an existing model checkpoint for testing.')
    parser.add_argument('--train', dest='train', action='store_true', help='Flag to indicate if the model should be trained.')
    parser.add_argument('--no-train', dest='train', action='store_false', help='Flag to indicate if the model should not be trained.')
    parser.add_argument('--test', dest='test', action='store_true', help='Flag to indicate if the model should be tested.')
    parser.set_defaults(train=False, test=False)

    args = parser.parse_args()

    pipeline = GraphEmbeddingPipeline(
        args.config,
        args.mapping,
        args.mapping_testing,
        args.claim_paragraph_mapping,
        args.prerankings_pairs,
        args.citing_graphs,
        args.cited_graphs,
        args.hardest_graphs,
        args.preranking_graphs,
        args.preranking_test,
        args.gold_dict,
        args.best_model_dir,
        existing_model_path=args.existing_model_path,
        train=args.train,
        test=args.test
    )
    pipeline.run()