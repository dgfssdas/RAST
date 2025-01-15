import numpy as np
import networkx as nx
from collections import defaultdict
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, args, logging):
        self.args = args
        self.dataset = args.dataset
        self.dataset_dir = str(os.path.join(args.data_dir, args.dataset))
        self.seed = args.seed
        self.windows_size = args.windows_size
        self.test_batch_size = args.test_batch_size

        self.pt_file = os.path.join(self.dataset_dir, "physical_topology_final.csv")
        self.pk_file = os.path.join(self.dataset_dir, "priori_knowledge_final.csv")
        self.temporal_data_path = os.path.join(self.dataset_dir, "temporal_data")

        self.n_physical_nodes = self.load_node_count(self.pt_file)
        self.n_priori_nodes = self.load_node_count(self.pk_file)
        self.n_nodes = self.n_physical_nodes + self.n_priori_nodes

        self.node_temporal_data = self.load_temporal_data(self.n_physical_nodes, self.temporal_data_path)
        self.train_data, self.valid_data, self.test_data = self.train_valid_test(self.node_temporal_data)

        self.kg_data = self.merge_graph(self.pt_file, self.pk_file)
        self.kg_graph, self.kg_relation_dict = self.construct_kg(self.kg_data, logging)

        self.print_info(logging)

    def load_node_count(self, pt_file):

        df = pd.read_csv(pt_file)

        nodes = set(df["head"]).union(set(df["tail"]))
        return len(nodes)

    def load_temporal_data(self, node_num, data_dir):

        node_temporal_data = {}
        for node_id in range(node_num):
            file_path = f"{data_dir}/node_{node_id}.csv"
            node_temporal_data[node_id] = pd.read_csv(file_path)

        return node_temporal_data

    def split_data(self, data, window_size, val_size=0.1, test_size=0.2):

        sequences = []
        labels = []
        for i in range(len(data) - window_size):
            sequences.append(data.iloc[i:i + window_size, :-1].values)
            labels.append(data.iloc[i + window_size - 1, -1])

        sequences = np.array(sequences)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=self.seed, shuffle=False)

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=self.seed, shuffle=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_valid_test(self, node_temporal_data):

        train_data: dict = {"X": [], "y": []}
        val_data: dict = {"X": [], "y": []}
        test_data: dict = {"X": [], "y": []}

        for node_id, data in node_temporal_data.items():
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(data, self.windows_size)

            train_data["X"].append(X_train)
            train_data["y"].append(y_train)

            val_data["X"].append(X_val)
            val_data["y"].append(y_val)

            test_data["X"].append(X_test)
            test_data["y"].append(y_test)

        return train_data, val_data, test_data

    def merge_graph(self, pt, pk):

        pt_df = pd.read_csv(pt)
        pt_nodes = set(pt_df["head"]).union(set(pt_df["tail"]))
        # pt_relations = set(pt_df["relation"])

        pk_df = pd.read_csv(pk)
        # pk_nodes = set(pk_df["head"]).union(set(pk_df["tail"]))
        # pk_relations = set(pk_df["relation"])

        max_pt_node_index = max(pt_nodes)
        # max_pt_relation_index = max(pt_relations)

        pk_df["head"] = pk_df["head"] + max_pt_node_index + 1
        pk_df["tail"] = pk_df["tail"] + max_pt_node_index + 1

        merged_triplets = pt_df[["head", "relation", "tail"]].values.tolist() + pk_df[["head", "relation", "tail"]].values.tolist()
        merged_triplets = merged_triplets.drop_duplicates()

        return merged_triplets

    def construct_kg(self, kg_data, logging):

        kg_n_relations = max(kg_data['relation']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'head': 'tail', 'tail': 'head'}, axis='columns')
        inverse_kg_data['relation'] += kg_n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        self.kg_train_data = kg_data
        self.kg_n_train = len(self.kg_train_data)
        self.kg_n_relations = max(kg_data['relation']) + 1

        kg_graph = nx.MultiDiGraph()
        logging.info("begin load communication kg triples ...")
        rd = defaultdict(list)
        for row in self.kg_train_data.iterrows():
            head, relation, tail = row[1]
            kg_graph.add_edge(head, tail, key=relation)
            rd[relation].append([head, tail])
        return kg_graph, rd

    def print_info(self, logging):
        logging.info('n_nodes:              %d' % self.n_nodes)
        logging.info('n_physical_nodes:     %d' % self.n_physical_nodes)
        logging.info('n_priori_nodes:       %d' % self.n_priori_nodes)
        logging.info('n_relations:          %d' % self.kg_n_relations)
        logging.info('windows_size:         %d' % self.windows_size)
        logging.info('n_train_windows:      %d' % len(self.train_data['X'][0]))
        logging.info('n_valid_windows:      %d' % len(self.valid_data['X'][0]))
        logging.info('n_test_windows:       %d' % len(self.test_data['X'][0]))
