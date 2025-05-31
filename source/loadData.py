import gzip
import ijson
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
from torch_geometric.utils import degree
import networkx as nx
from scipy.sparse.linalg import eigsh
import numpy as np
from collections import Counter
from decimal import Decimal # Importa Decimal per gestire il tipo

# Aggiusta la funzione per calcolare gli autovalori del Laplaciano
def calculate_laplacian_eigenfeatures(graph_nx, k=5):
    L = nx.laplacian_matrix(graph_nx).astype(float)
    num_nodes = graph_nx.number_of_nodes()

    if num_nodes == 0:
        return torch.empty((0, k), dtype=torch.float)
    
    if num_nodes == 1:
        return torch.zeros((1, k), dtype=torch.float)
    
    k_actual = min(k, num_nodes - 1 if num_nodes > 1 else 0)
    
    if k_actual == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float)

    try:
        # 'SM' for smallest magnitude (closest to zero)
        eigenvalues, eigenvectors = eigsh(L, k=k_actual, which='SM', sigma=1e-8)
        
        # Ensure eigenvectors are of shape (num_nodes, k)
        eigenfeatures = np.zeros((num_nodes, k))
        eigenfeatures[:, :k_actual] = eigenvectors
        
        return torch.tensor(eigenfeatures, dtype=torch.float)
    except Exception as e:
        # Se si verifica un errore (es. grafo non connesso e num_nodes-1 < k_actual, o altri problemi di convergenza)
        # restituisci un tensore di zeri per quel grafo
        print(f"Warning: Failed to calculate Laplacian eigenfeatures for a graph: {e}. Returning zeros.")
        return torch.zeros((num_nodes, k), dtype=torch.float)


def dictToGraphObject(graph_dict, graph_id, is_test_set=False):
    """
    Converte un dizionario che rappresenta un grafo (dal formato JSON) in un oggetto Data di PyTorch Geometric.
    Estrae le features dei nodi, gli indici degli archi, gli attributi degli archi e le etichette.
    """
    
    num_nodes = graph_dict.get('num_nodes')
    if num_nodes is None or num_nodes == 0:
        # print(f"Skipping graph {graph_id} due to 0 or missing nodes.") # Già gestito nel loader
        return None, "NO_NODES"

    # Etichette del grafo (per la classificazione)
    y = graph_dict.get('y')
    if y is not None and len(y) > 0:
        y = torch.tensor(y[0], dtype=torch.long) # Assumiamo che y sia una lista di 1 elemento
    elif not is_test_set: # Se non è un test set e la label manca/è invalida
        # print(f"Skipping graph {graph_id} due to missing or invalid label.") # Già gestito nel loader
        return None, "INVALID_LABEL"
    else: # Per il test set, y potrebbe essere None
        y = None 

    # Edge Index (matrice di adiacenza in formato COO)
    edge_index_raw = graph_dict.get('edge_index')
    if edge_index_raw is not None and len(edge_index_raw) == 2:
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long) # Grafo senza archi

    # Crea un oggetto NetworkX per calcolare le features dei nodi
    graph_nx = nx.Graph()
    if num_nodes > 0: # Aggiungi nodi se num_nodes > 0
        graph_nx.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        # edge_index è [2, num_edges], quindi trasponiamo per ottenere coppie di nodi
        graph_nx.add_edges_from(edge_index.t().tolist())

    x_features_list = []

    # Features basate sulla topologia (come da esempio)
    # 1. Degree
    if num_nodes > 0:
        node_degrees = torch.tensor([d for n, d in graph_nx.degree()], dtype=torch.float).unsqueeze(1)
        x_features_list.append(node_degrees)

        # 2. Closeness Centrality (se il grafo è connesso e ha più di 1 nodo)
        if num_nodes > 1 and nx.is_connected(graph_nx):
            closeness_centrality = torch.tensor(list(nx.closeness_centrality(graph_nx).values()), dtype=torch.float).unsqueeze(1)
            x_features_list.append(closeness_centrality)
        else:
            x_features_list.append(torch.zeros((num_nodes, 1), dtype=torch.float)) # Se non connesso o 1 nodo, zeri

        # 3. Betweenness Centrality
        if num_nodes > 1:
            betweenness_centrality = torch.tensor(list(nx.betweenness_centrality(graph_nx).values()), dtype=torch.float).unsqueeze(1)
            x_features_list.append(betweenness_centrality)
        else:
            x_features_list.append(torch.zeros((num_nodes, 1), dtype=torch.float)) # Se 1 nodo, zeri

        # 4. PageRank (gestisce i nodi isolati implicitamente)
        pagerank = torch.tensor(list(nx.pagerank(graph_nx).values()), dtype=torch.float).unsqueeze(1)
        x_features_list.append(pagerank)

        # 5. Clustering Coefficient
        clustering_coeff = torch.tensor(list(nx.clustering(graph_nx).values()), dtype=torch.float).unsqueeze(1)
        x_features_list.append(clustering_coeff)

        # 6. Laplacian Eigenfeatures (k=5 per 5 features)
        laplacian_eigenfeatures = calculate_laplacian_eigenfeatures(graph_nx, k=5)
        x_features_list.append(laplacian_eigenfeatures)
    else: # Se num_nodes è 0, aggiungi tensori vuoti per tutte le features per mantenere la consistenza
        for _ in range(5): # Per Degree, Closeness, Betweenness, PageRank, Clustering
            x_features_list.append(torch.empty((0, 1), dtype=torch.float))
        x_features_list.append(torch.empty((0, 5), dtype=torch.float)) # Per Laplacian Eigenfeatures

    # Attributi degli archi
    edge_attr = None
    additional_node_features_tensor = None
    edge_attr_raw = graph_dict.get('edge_attr')
    if edge_attr_raw is None:
        edge_attr_raw = graph_dict.get('edge_attributes') # o qualsiasi altra chiave che il dataset usa

    if edge_attr_raw is not None and len(edge_attr_raw) > 0:
        try:
            converted_attr_float = []
            for sublist in edge_attr_raw:
                # Assicurati che tutti gli elementi siano convertibili in float (gestisce Decimal)
                converted_attr_float.append([float(val) for val in sublist])
            
            potential_attr_tensor = torch.tensor(converted_attr_float, dtype=torch.float)

            # Caso 1: I numeri corrispondono direttamente (ad es. archi diretti o attributi già raddoppiati)
            if edge_index.shape[1] > 0 and potential_attr_tensor.shape[0] == edge_index.shape[1]:
                edge_attr = potential_attr_tensor
                # print(f"DEBUG: Graph {graph_id} - Direct match for edge_attr: {potential_attr_tensor.shape}") # Debug
            
            # Caso 2: Gli attributi di bordo sono la metà degli archi (molto comune per grafi non diretti come OGBG-PPA)
            elif edge_index.shape[1] > 0 and potential_attr_tensor.shape[0] * 2 == edge_index.shape[1]:
                # Duplica gli attributi di bordo per l'altra direzione dell'arco
                # Assumiamo che l'ordine degli archi in edge_index sia tale che (u,v) è seguito da (v,u) o viceversa
                # e che l'attributo si riferisca a una singola coppia non diretta.
                edge_attr = torch.cat([potential_attr_tensor, potential_attr_tensor], dim=0)
                # print(f"DEBUG: Graph {graph_id} - Duplicating edge_attr: original {potential_attr_tensor.shape}, new {edge_attr.shape}") # Debug

            # Caso 3: Attributi di bordo interpretati come feature aggiuntive dei nodi (se non ci sono archi ma il conteggio corrisponde ai nodi)
            elif edge_index.shape[1] == 0 and potential_attr_tensor.shape[0] == num_nodes:
                print(f"INFO: Graph '{graph_id}' (num_nodes={num_nodes}) has {potential_attr_tensor.shape[0]} edge_attributes but 0 edges. Interpreting as additional node features.")
                additional_node_features_tensor = potential_attr_tensor
            
            # Caso 4: Mismatch non gestito - genera il warning
            else:
                print(f"Warning for graph {graph_id}: Mismatch between number of edges ({edge_index.shape[1]}) and edge_attributes ({potential_attr_tensor.shape[0]}). Setting edge_attr to None. DEBUG: num_nodes={num_nodes}.")
        except Exception as e:
            print(f"Error processing edge_attr for graph {graph_id}: {e}. Setting edge_attr to None.")
            edge_attr = None
    
    # Concatena tutte le features dei nodi
    if additional_node_features_tensor is not None:
        if x_features_list: # Se ci sono già features del nodo
            x = torch.cat(x_features_list + [additional_node_features_tensor], dim=1) # Aggiungi alla fine
        else: # Se non ci sono altre features del nodo
            x = additional_node_features_tensor
    elif x_features_list:
        x = torch.cat(x_features_list, dim=1)
    else: # Nessuna feature nodo rilevata o num_nodes = 0
        # Questo caso dovrebbe essere gestito da x_features_list che è già empty per num_nodes=0
        # Se num_nodes > 0 ma nessuna feature è stata aggiunta, allora crea un tensore vuoto
        if num_nodes > 0:
            x = torch.empty((num_nodes, 0), dtype=torch.float)
        else:
            x = torch.empty((0, 0), dtype=torch.float) # Per coerenza con num_nodes=0

    # Controlla che le features del nodo non siano vuote dopo la concatenazione (a meno che num_nodes=0)
    if num_nodes > 0 and x.shape[1] == 0:
        # print(f"Skipping graph {graph_id} due to 0 features after processing.") # Già gestito nel loader
        return None, "NO_FEATURES"

    # Crea l'oggetto Data di PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)
    data.graph_id = graph_id # Aggiungi l'ID del grafo per debug/riferimento
    
    return data, None


class GraphDataset(Dataset):
    def __init__(self, raw_dir=None, transform=None, pre_transform=None, is_test_set=False):
        super().__init__(raw_dir, transform, pre_transform)
        self.is_test_set = is_test_set
        self.data_list = [] # Lista per contenere gli oggetti Data caricati

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def loadGraphs(self, path):
        self.data_list = []
        no_nodes_skipped_count = 0
        invalid_labels_count = 0
        no_features_count = 0
        valid_graphs_count = 0
        label_distribution = Counter()
        
        # Prova ad aprire il file, sia esso gzip compresso o meno
        try:
            if path.endswith('.gz'):
                file_opener = gzip.open
            else:
                file_opener = open
            
            with file_opener(path, 'rb') as f:
                # Utilizza ijson.items per iterare sugli oggetti di livello superiore
                # Assicurati che il percorso ijson sia corretto per la tua struttura JSON
                # Se è un array di oggetti, sarà 'item'
                # Se è un oggetto con chiave 'graphs' che contiene un array, sarà 'graphs.item'
                # Assumiamo che il file .json.gz contenga un array di oggetti grafo direttamente
                print(f"Loading graphs from {os.path.basename(path)}...")
                items_generator = ijson.items(f, 'item')
                
                # Wrap the generator with tqdm for a progress bar
                for i, item in enumerate(tqdm(items_generator)):
                    graph_id = item.get('id', f"graph_{i}") # Ottieni l'ID del grafo se presente, altrimenti usa l'indice
                    
                    data, skip_reason = dictToGraphObject(item, graph_id, self.is_test_set)
                    
                    if data is None:
                        if skip_reason == "NO_NODES":
                            no_nodes_skipped_count += 1
                        elif skip_reason == "INVALID_LABEL":
                            invalid_labels_count += 1
                        elif skip_reason == "NO_FEATURES":
                            no_features_count += 1
                        # print(f"Skipping graph {graph_id} due to: {skip_reason}") # DEBUG: troppo verboso
                        continue # Passa al prossimo grafo

                    self.data_list.append(data)
                    valid_graphs_count += 1
                    if not self.is_test_set and data.y is not None:
                        label_distribution[data.y.item()] += 1
                        
        except ijson.common.IncompleteJSONError as e:
            print(f"Error: Incomplete JSON data in {os.path.basename(path)}: {e}. Some graphs might be missing.")
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}.")
            
        print(f"\n--- Dataset Loading Summary ({os.path.basename(path)}) ---")
        print(f"Total graphs attempted to load: {i + 1 if 'i' in locals() else 0}")
        print(f"Graphs with 0 nodes skipped: {no_nodes_skipped_count}")
        print(f"Graphs with invalid/missing labels skipped (non-test sets only): {invalid_labels_count}")
        print(f"Graphs with 0 features after processing skipped: {no_features_count}")
        print(f"Successfully loaded graphs: {valid_graphs_count}")
        
        if not self.is_test_set:
            print(f"Label distribution of loaded graphs: {label_distribution}")
        else:
            print("Label distribution not applicable for test set (no labels expected).")
        print("--------------------------------------------------\n")
        
        return self.data_list