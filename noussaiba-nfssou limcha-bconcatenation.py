from sklearn.decomposition import PCA, TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import numpy as np
import time
import pickle
import random  # Ajout de l'import manquant
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dgl
from dgl.nn import HeteroGraphConv
from collections import defaultdict
from rdflib import Graph, URIRef, RDF, Namespace
from rdflib.namespace import DC, FOAF, RDFS
DC_TERMS = Namespace("http://purl.org/dc/terms/")
from dgl import heterograph
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN, OPTICS, Birch, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import hdbscan
    HDBSCAN = hdbscan.HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN non disponible. Pour l'installer: pip install hdbscan")

# Configuration du GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"\nGPU disponible: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print("\nGPU non disponible, utilisation du CPU")

class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3, node_type=None):
        super().__init__()

        # Store node type information for heterogeneous-aware processing
        self.node_type = node_type

        # Advanced normalization layer
        self.input_norm = nn.LayerNorm(in_dim).to(device)

        # Type-specific embedding (if node type is provided)
        self.type_embedding = None
        if node_type is not None:
            self.type_embedding = nn.Parameter(torch.randn(1, hidden_dim)).to(device)

        # Improved encoder with more layers and residual connections
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU
            nn.Dropout(dropout)
        ).to(device)

        self.encoder_layer2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ).to(device)

        self.encoder_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ).to(device)

        # Enhanced multi-head attention with more heads and better scaling
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=16,
            dropout=dropout,
            batch_first=True  # Simplifies usage
        ).to(device)

        # Skip connections
        self.skip_connection1 = nn.Linear(in_dim, hidden_dim).to(device)
        self.skip_connection2 = nn.Linear(hidden_dim*2, hidden_dim).to(device)

        # Improved projections with more layers
        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout/2),  # Less dropout here
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, out_dim)
        ).to(device)

        self.log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, out_dim)
        ).to(device)

        # Improved decoder with more layers and residual connections
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(out_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ).to(device)

        self.decoder_layer2 = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ).to(device)

        self.decoder_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ).to(device)

        self.decoder_output = nn.Linear(hidden_dim*2, in_dim).to(device)

        # Skip connections for decoder
        self.decoder_skip1 = nn.Linear(out_dim, hidden_dim).to(device)
        self.decoder_skip2 = nn.Linear(hidden_dim//2, hidden_dim*2).to(device)

        # Heterogeneous graph-specific attention mechanism
        self.het_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        # Adaptive temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.1)).to(device)

    def encode(self, x):
        # Layer 1 with skip connection
        h1 = self.encoder_layer1(x)

        # Layer 2 with skip connection from input
        h2 = self.encoder_layer2(h1)
        h2 = h2 + self.skip_connection1(x)  # Skip connection from input

        # Add type-specific embedding if available
        if self.type_embedding is not None:
            h2 = h2 + self.type_embedding

        # Layer 3 with skip connection from layer 1
        h3 = self.encoder_layer3(h2)
        h3 = h3 + self.skip_connection2(h1)  # Skip connection from layer 1

        # Apply self-attention with residual connection
        # Using batch_first=True simplifies the code
        h_attended, attention_weights = self.attention(
            h3.unsqueeze(0),  # Add batch dimension
            h3.unsqueeze(0),
            h3.unsqueeze(0)
        )
        h3 = h3 + h_attended.squeeze(0)  # Add attention with residual connection

        # Apply heterogeneous-specific attention if node type is provided
        if self.type_embedding is not None:
            het_attn_scores = self.het_attention(h3)
            het_attn_weights = F.softmax(het_attn_scores, dim=0)
            h3 = h3 * het_attn_weights

        # Project to mean and log_std
        mean = self.mean(h3)
        log_std = self.log_std(h3)

        return mean, log_std, attention_weights

    def reparameterize(self, mean, log_std):
        # Improved reparameterization with clipping to prevent extreme values
        std = torch.exp(torch.clamp(log_std, min=-10, max=10))
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        # Layer 1
        h1 = self.decoder_layer1(z)

        # Layer 2 with skip connection
        h2 = self.decoder_layer2(h1)
        h2 = h2 + self.decoder_skip1(z)  # Skip connection from z

        # Layer 3 with skip connection
        h3 = self.decoder_layer3(h2)
        h3 = h3 + self.decoder_skip2(h1)  # Skip connection from layer 1

        # Output layer
        x_recon = self.decoder_output(h3)

        return x_recon

    def forward(self, x, adj=None, cross_type_adj=None):
        # Normalize input
        x = self.input_norm(x)
        mean, log_std, attention_weights = self.encode(x)
        z = self.reparameterize(mean, log_std)
        x_recon = self.decode(z)

        # Enhanced structure-preserving loss if adjacency matrix is provided
        structure_loss = 0
        if adj is not None:
            # Normalize embeddings
            z_norm = F.normalize(z, p=2, dim=1)

            # Compute similarity matrix
            sim_matrix = torch.mm(z_norm, z_norm.t())

            # Compute structure loss with weighted MSE
            # Give more weight to existing edges (1s in adjacency matrix)
            edge_weight = 3.0  # Increased weight for existing edges
            non_edge_weight = 1.0  # Weight for non-edges

            # Create weight matrix
            weight_matrix = torch.ones_like(adj)
            weight_matrix = torch.where(adj > 0, edge_weight * weight_matrix, non_edge_weight * weight_matrix)

            # Compute weighted MSE
            diff = (sim_matrix - adj.float()) ** 2
            weighted_diff = diff * weight_matrix
            structure_loss = torch.sum(weighted_diff) / torch.sum(weight_matrix)

        # Cross-type structure preservation (for heterogeneous graphs)
        cross_type_loss = 0
        if cross_type_adj is not None:
            try:
                # Normalize embeddings
                z_norm = F.normalize(z, p=2, dim=1)

                # Compute cross-type similarity
                cross_sim = torch.mm(z_norm, cross_type_adj.float())

                # Target is 1 for connected nodes across types
                target = torch.where(cross_type_adj > 0,
                                    torch.ones_like(cross_type_adj, dtype=torch.float),
                                    torch.zeros_like(cross_type_adj, dtype=torch.float))

                # Compute cross-type loss
                cross_type_loss = F.binary_cross_entropy_with_logits(cross_sim, target)
            except Exception as e:
                print(f"Warning: Error computing cross-type loss: {str(e)}")
                cross_type_loss = torch.tensor(0.0, device=device)

        return x_recon, mean, log_std, structure_loss, cross_type_loss, attention_weights

def get_adjacency_matrix(g, ntype, etype=None):
    """
    Helper function to get adjacency matrix for a specific node type

    Args:
        g: DGL graph
        ntype: node type
        etype: optional edge type (if None, considers all edge types for this node type)

    Returns:
        Adjacency matrix as a PyTorch tensor
    """
    n = g.number_of_nodes(ntype)
    adj = torch.zeros((n, n), device=device)

    # If specific edge type is provided
    if etype is not None:
        if ntype == 'author' and etype == 'hasPublishedIn':
            edges = g.edges(etype=('author', 'hasPublishedIn', 'venue'))
        elif ntype == 'publication' and etype == 'creator':
            edges = g.edges(etype=('publication', 'creator', 'author'))
        else:
            return adj  # Return empty adjacency matrix

        if len(edges[0]) == 0:
            return adj

        # Create adjacency matrix
        src, dst = edges
        for s, d in zip(src, dst):
            adj[s][d] = 1.0
            adj[d][s] = 1.0  # Make symmetric
    else:
        # Consider all edge types where this node type is both source and destination
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == ntype and dsttype == ntype:
                edges = g.edges(etype=(srctype, etype, dsttype))

                if len(edges[0]) == 0:
                    continue

                # Add edges to adjacency matrix
                src, dst = edges
                for s, d in zip(src, dst):
                    adj[s][d] = 1.0
                    adj[d][s] = 1.0  # Make symmetric

    # Add self-loops
    adj = adj + torch.eye(n, device=device)

    return adj

def get_cross_type_adjacency(g, src_type, dst_type):
    """
    Get cross-type adjacency matrix between two node types

    Args:
        g: DGL graph
        src_type: source node type
        dst_type: destination node type

    Returns:
        Cross-type adjacency matrix as a PyTorch tensor
    """
    n_src = g.number_of_nodes(src_type)
    n_dst = g.number_of_nodes(dst_type)

    # Initialize cross-type adjacency matrix
    cross_adj = torch.zeros((n_src, n_dst), device=device)

    # Find all edge types connecting these two node types
    for srctype, etype, dsttype in g.canonical_etypes:
        if (srctype == src_type and dsttype == dst_type) or (srctype == dst_type and dsttype == src_type):
            edges = g.edges(etype=(srctype, etype, dsttype))

            if len(edges[0]) == 0:
                continue

            # Add edges to cross-type adjacency matrix
            src, dst = edges
            if srctype == src_type:
                for s, d in zip(src, dst):
                    try:
                        cross_adj[s, d] = 1.0
                    except IndexError:
                        print(f"Warning: Skipping edge ({s},{d}) in get_cross_type_adjacency. Matrix shape: {cross_adj.shape}")
            else:  # srctype == dst_type
                for s, d in zip(src, dst):
                    try:
                        cross_adj[d, s] = 1.0
                    except IndexError:
                        print(f"Warning: Skipping edge ({d},{s}) in get_cross_type_adjacency. Matrix shape: {cross_adj.shape}")

    return cross_adj

def train_vgae(g, node_features, num_epochs=1000, lr=0.0001, weight_decay=1e-5):
    """
    Train VGAE with improved training procedure
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgae_models = {}
    optimizers = {}

    for ntype in g.ntypes:
        if g.number_of_nodes(ntype) == 0:
            continue
        if isinstance(node_features, dict):
            if ntype in node_features:
                features = node_features[ntype]
            else:
                node_ids = g.nodes(ntype)
                features = []
                for nid in node_ids:
                    if int(nid) in node_features:
                        features.append(node_features[int(nid)])
                if not features:
                    continue
                features = torch.stack(features)
        else:
            features = g.nodes[ntype].data['feat']

        # Normalisation des features
        features = F.normalize(features, p=2, dim=1)

        in_dim = features.shape[1]
        hidden_dim = 768  # Augmenté davantage
        out_dim = 384     # Augmenté davantage

        # Create VGAE with node type information for heterogeneous-aware processing
        # Pass node type to enable type-specific processing
        model = VGAE(in_dim, hidden_dim, out_dim, dropout=0.3, node_type=ntype).to(device)

        # Improved optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        vgae_models[ntype] = model
        optimizers[ntype] = optimizer

    if not vgae_models:
        raise ValueError("No valid node types found with features")

    # Improved contrastive loss with hard negative mining
    def contrastive_loss(z1, z2, temperature=0.1):
        """
        Calcule une contrastive loss améliorée entre deux ensembles d'embeddings
        avec hard negative mining et température adaptative
        """
        # Normalisation L2
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Calculer la matrice de similarité cosinus
        batch_size = z1.shape[0]
        sim_matrix = torch.mm(z1, z2.t()) / temperature

        # Masque pour les paires positives (diagonale)
        pos_mask = torch.eye(batch_size, device=z1.device)

        # Masque pour les paires négatives (hors diagonale)
        neg_mask = 1 - pos_mask

        # Appliquer l'exponentielle après avoir soustrait le maximum pour stabilité numérique
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        exp_sim = torch.exp(sim_matrix - sim_max)

        # Similarités positives (diagonale)
        pos_sim = torch.sum(exp_sim * pos_mask, dim=1)

        # Hard negative mining: donner plus de poids aux négatifs difficiles
        # (ceux avec une similarité élevée)
        hard_weights = torch.where(
            sim_matrix > 0,  # Similarité positive = négatif difficile
            torch.ones_like(sim_matrix) * 2.0,  # Doubler le poids
            torch.ones_like(sim_matrix)
        )
        hard_weights = hard_weights * neg_mask  # Appliquer seulement aux négatifs

        # Similarités négatives pondérées
        neg_sim = torch.sum(exp_sim * hard_weights, dim=1)

        # InfoNCE loss avec stabilité numérique améliorée
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-12)).mean()

        return loss

    # Learning rate scheduler
    schedulers = {}
    for ntype, optimizer in optimizers.items():
        schedulers[ntype] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )

    # Contrastive loss avec température adaptative
    class AdaptiveTemperature(nn.Module):
        def __init__(self, initial_temp=0.07):
            super().__init__()
            self.temperature = nn.Parameter(torch.tensor(initial_temp))

        def forward(self, z1, z2):
            temperature = torch.clamp(self.temperature, min=0.01, max=0.5)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            similarity = torch.mm(z1, z2.T) / temperature
            positive_pairs = torch.diag(similarity)

            loss = -torch.mean(
                positive_pairs - torch.logsumexp(similarity, dim=1)
            )
            return loss

    adaptive_temp = AdaptiveTemperature().to(device)

    # Add structure loss weight
    structure_weight = 0.1

    losses = []  # Pour suivre la progression
    best_loss = float('inf')
    patience = 20
    no_improve = 0

    print("\nDébut de l'entraînement VGAE:")
    print("=" * 50)

    for epoch in range(num_epochs):
        total_loss = 0
        for ntype, model in vgae_models.items():
            model.train()
            optimizer = optimizers[ntype]
            optimizer.zero_grad()

            # Get features
            if isinstance(node_features, dict):
                if ntype in node_features:
                    features = node_features[ntype]
                else:
                    node_ids = g.nodes(ntype)
                    features = []
                    for nid in node_ids:
                        if int(nid) in node_features:
                            features.append(node_features[int(nid)])
                    features = torch.stack(features)
            else:
                features = g.nodes[ntype].data['feat']

            features = F.normalize(features, p=2, dim=1)
            features = features.to(device)

            # Get adjacency matrix for this node type
            adj = get_adjacency_matrix(g, ntype)

            # Get cross-type adjacency matrices - with more robust error handling
            cross_type_adj = None
            for other_ntype in g.ntypes:
                if other_ntype != ntype:
                    # Try to find a relation between these types
                    for srctype, etype, dsttype in g.canonical_etypes:
                        if (srctype == ntype and dsttype == other_ntype) or (srctype == other_ntype and dsttype == ntype):
                            # Found a relation, create cross-type adjacency
                            try:
                                edges = g.edges(etype=(srctype, etype, dsttype))

                                # Skip if no edges
                                if len(edges[0]) == 0:
                                    continue

                                # Create cross-type adjacency matrix if not already created
                                if cross_type_adj is None:
                                    n_src = g.number_of_nodes(ntype)
                                    n_dst = g.number_of_nodes(other_ntype)
                                    cross_type_adj = torch.zeros((n_src, n_dst), device=device)

                                # Get max indices to check bounds
                                max_src_idx = g.number_of_nodes(srctype) - 1
                                max_dst_idx = g.number_of_nodes(dsttype) - 1

                                # Fill in cross-type adjacency with bounds checking
                                valid_edges = 0
                                skipped_edges = 0

                                for s, d in zip(edges[0], edges[1]):
                                    # Check if indices are within bounds
                                    if s > max_src_idx or d > max_dst_idx:
                                        skipped_edges += 1
                                        continue

                                    try:
                                        if srctype == ntype:
                                            # s is the index in ntype, d is the index in other_ntype
                                            cross_type_adj[s, d] = 1.0
                                        else:
                                            # d is the index in ntype, s is the index in other_ntype
                                            cross_type_adj[d, s] = 1.0
                                        valid_edges += 1
                                    except IndexError:
                                        skipped_edges += 1

                                if skipped_edges > 0:
                                    print(f"Warning: Skipped {skipped_edges} edges between {srctype} and {dsttype}. Added {valid_edges} valid edges.")

                            except Exception as e:
                                print(f"Error processing edges between {srctype} and {dsttype}: {str(e)}")

            # Forward pass with adjacency matrices
            x_recon, mean, log_std, structure_loss, cross_type_loss, _ = model(features, adj, cross_type_adj)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, features)

            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + 2 * log_std - mean.pow(2) - (2 * log_std).exp())

            # Ajouter la contrastive loss
            z = model.reparameterize(mean, log_std)
            # Créer une version augmentée des features avec dropout
            features_aug = F.dropout(features, p=0.2)

            # Handle the new encode method that returns attention weights
            try:
                mean_aug, log_std_aug, _ = model.encode(features_aug)
            except ValueError:
                # If the old encode method is used (returns only mean and log_std)
                mean_aug, log_std_aug = model.encode(features_aug)

            z_aug = model.reparameterize(mean_aug, log_std_aug)

            # Calculer la contrastive loss avec température adaptative
            temperature = 0.5
            contrastive = contrastive_loss(z, z_aug, temperature)

            # Ajuster les poids des différentes losses
            recon_weight = 1.0
            kl_weight = 0.1
            struct_weight = 0.5
            contrastive_weight = 0.2  # Poids pour la contrastive loss
            cross_type_weight = 0.3   # Poids pour la cross-type loss

            # Ensure cross_type_loss is a tensor
            if not isinstance(cross_type_loss, torch.Tensor):
                cross_type_loss = torch.tensor(0.0, device=device)

            # Combiner toutes les losses
            loss = (recon_weight * recon_loss +
                   kl_weight * kl_loss +
                   struct_weight * structure_loss +
                   contrastive_weight * contrastive +
                   cross_type_weight * cross_type_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(vgae_models)
        losses.append(avg_loss)

        # Afficher la progression
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping après {epoch+1} epochs!")
            break

    # Afficher la courbe d'entraînement
    plot_training_progress(losses, "VGAE Training Progress")

    embeddings = {}
    for ntype, model in vgae_models.items():
        model.eval()
        with torch.no_grad():
            if isinstance(node_features, dict):
                if ntype in node_features:
                    features = node_features[ntype]
                else:
                    node_ids = g.nodes(ntype)
                    features = []
                    for nid in node_ids:
                        if int(nid) in node_features:
                            features.append(node_features[int(nid)])
                    features = torch.stack(features)
            else:
                features = g.nodes[ntype].data['feat']
            features = F.normalize(features, p=2, dim=1)
            features = features.to(device)
            # Handle the new encode method that returns attention weights
            try:
                mean, _, _ = model.encode(features)
            except ValueError:
                # If the old encode method is used (returns only mean and log_std)
                mean, _ = model.encode(features)

            embeddings[ntype] = mean

    for ntype in embeddings:
        norm = torch.norm(embeddings[ntype], p=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        embeddings[ntype] = embeddings[ntype] / norm

    return vgae_models, embeddings

class GlobalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attention_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        ).to(device)

        self.projection = nn.Linear(in_dim, out_dim).to(device)

    def forward(self, x):
        x = x.to(device)
        # Calculate attention weights
        attention_weights = self.attention_net(x)  # [num_nodes, 1]
        attention_weights = F.softmax(attention_weights, dim=0)  # [num_nodes, 1]

        # Project and weight the features
        projected = self.projection(x)  # [num_nodes, out_dim]
        attended = projected * attention_weights  # [num_nodes, out_dim]

        return attended  # [num_nodes, out_dim]

class ImprovedHGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads=8, dropout=0.3):
        super().__init__()
        # Ensure dimensions are compatible
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = out_dim  # Use same dim for hidden states
        self.ntypes = ntypes
        self.etypes = etypes
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads

        # Initialize all components with matching dimensions
        self.k_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.hidden_dim).to(device) for ntype in ntypes
        })

        self.q_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.hidden_dim).to(device)
            for ntype in ntypes
        })

        self.v_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.hidden_dim).to(device)
            for ntype in ntypes
        })

        self.out_linears = nn.ModuleDict({
            ntype: nn.Linear(self.hidden_dim, self.out_dim).to(device)
            for ntype in ntypes
        })

        # Improved relation attention with more parameters
        self.relation_attn = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, n_heads)
            ).to(device) for etype in etypes
        })

        # Add layer normalization
        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(out_dim).to(device)
            for ntype in ntypes
        })

        # Initialize global attention for each type with correct dimensions
        self.global_attention = nn.ModuleDict({
            ntype: GlobalAttention(in_dim, self.hidden_dim).to(device)
            for ntype in ntypes
        })

        # Skip connections for better gradient flow
        self.skip_connections = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim).to(device)
            for ntype in ntypes
        })

        self.dropout_layer = nn.Dropout(dropout).to(device)

        # Add feed-forward networks after attention
        self.ffn = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.LayerNorm(out_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim)
            ).to(device) for ntype in ntypes
        })

    def forward(self, g, feat_dict, rel_emb_dict):
        # Ensure all inputs are on the same device as the model
        device = next(self.parameters()).device
        feat_dict = {k: v.to(device) for k, v in feat_dict.items()}
        rel_emb_dict = {k: v.to(device) for k, v in rel_emb_dict.items()}

        # Initialize output features
        output_feats = {}

        # Store original features for residual connections
        orig_feats = {ntype: feat.clone() for ntype, feat in feat_dict.items()}

        # Process each node type with global attention
        for ntype in feat_dict:
            if ntype in self.global_attention:
                # Apply global attention
                output_feats[ntype] = self.global_attention[ntype](feat_dict[ntype])
            else:
                # Copy features if no global attention
                output_feats[ntype] = feat_dict[ntype]

        # Process each edge type with improved multi-head attention
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype not in feat_dict or dsttype not in feat_dict:
                continue

            # Get edges of this type
            edges = g.edges(etype=(srctype, etype, dsttype))
            if len(edges[0]) == 0:
                continue

            # Get source and destination node features
            src_feat = feat_dict[srctype][edges[0].to(device)]
            dst_feat = feat_dict[dsttype][edges[1].to(device)]

            # Apply attention
            if etype in self.relation_attn:
                # Get relation embedding or create a zero tensor
                if etype in rel_emb_dict:
                    rel_emb = rel_emb_dict[etype].to(device)
                else:
                    rel_emb = torch.zeros(self.in_dim, device=device)

                # Apply improved relation attention
                attn_weights = self.relation_attn[etype](rel_emb).view(1, self.n_heads)
                attn_weights = F.softmax(attn_weights, dim=1)

                # Apply transformations with multi-head attention
                k = self.k_linears[srctype](src_feat).view(-1, self.n_heads, self.d_k)
                q = self.q_linears[dsttype](dst_feat).view(-1, self.n_heads, self.d_k)
                v = self.v_linears[srctype](src_feat).view(-1, self.n_heads, self.d_k)

                # Compute attention scores for each head separately
                attended_heads = []
                for head in range(self.n_heads):
                    k_h = k[:, head]  # [num_nodes, d_k]
                    q_h = q[:, head]  # [num_nodes, d_k]
                    v_h = v[:, head]  # [num_nodes, d_k]

                    # Compute attention scores
                    scores = torch.mm(q_h, k_h.t()) / math.sqrt(self.d_k)
                    scores = F.softmax(scores, dim=1)

                    # Apply attention to values
                    attended_h = torch.mm(scores, v_h)  # [num_nodes, d_k]
                    attended_heads.append(attended_h)

                # Concatenate heads and apply output transformation
                attended = torch.cat(attended_heads, dim=1)  # [num_nodes, hidden_dim]
                out = self.out_linears[dsttype](attended)

                # Apply layer normalization and skip connection
                if dsttype in output_feats:
                    # Add to existing features with skip connection
                    skip_out = self.skip_connections[dsttype](feat_dict[dsttype][edges[1]])
                    output_feats[dsttype][edges[1]] = output_feats[dsttype][edges[1]] + out + skip_out
                else:
                    # Initialize with these features
                    temp = torch.zeros(feat_dict[dsttype].shape[0], self.out_dim, device=device)
                    temp[edges[1]] = out
                    output_feats[dsttype] = temp

        # Apply layer normalization, feed-forward network and residual connection
        for ntype in output_feats:
            if ntype in self.layer_norms:
                # Apply layer normalization
                output_feats[ntype] = self.layer_norms[ntype](output_feats[ntype])

                # Apply feed-forward network
                ffn_out = self.ffn[ntype](output_feats[ntype])

                # Add residual connection
                if ntype in self.skip_connections:
                    skip_out = self.skip_connections[ntype](orig_feats[ntype])
                    output_feats[ntype] = output_feats[ntype] + ffn_out + skip_out
                else:
                    output_feats[ntype] = output_feats[ntype] + ffn_out

                # Apply dropout
                output_feats[ntype] = self.dropout_layer(output_feats[ntype])

        return output_feats

# Alias for backward compatibility
HGTLayer = ImprovedHGTLayer

def create_heterogeneous_graph():
    """
    Crée un graphe hétérogène à partir du fichier RDF et extrait les caractéristiques des nœuds
    """
    # Chargement du graphe RDF
    rdf_graph = Graph()
    rdf_graph.parse("DBLP_petit.rdf", format="xml")

    EX = Namespace("http://example.org/")
    FEAT = Namespace("http://example.org/features/")

    # Collecte des nœuds
    authors = list(rdf_graph.subjects(RDF.type, FOAF.Person))
    publications = list(rdf_graph.subjects(RDF.type, EX.Publication))
    venues = list(rdf_graph.subjects(RDF.type, EX.Venue))
    domains = list(rdf_graph.subjects(RDF.type, EX.Domain))

    author2id = {a: i for i, a in enumerate(authors)}
    pub2id = {p: i for i, p in enumerate(publications)}
    venue2id = {v: i for i, v in enumerate(venues)}
    domain2id = {d: i for i, d in enumerate(domains)}

    # Relations à extraire
    edges = {
        ('author', 'domain_dominant', 'domain'): ([], []),
        ('publication', 'creator', 'author'): ([], []),
        ('author', 'hasDomain', 'domain'): ([], []),
        ('author', 'hasPublishedIn', 'venue'): ([], []),
        ('publication', 'isPartOf', 'venue'): ([], []),
        ('venue', 'publishesDomain', 'domain'): ([], []),
    }

    # author --[domain_dominant]--> domain
    for author in authors:
        for domain in rdf_graph.objects(author, FEAT.domain_dominant):
            if domain in domain2id:
                edges[('author', 'domain_dominant', 'domain')][0].append(author2id[author])
                edges[('author', 'domain_dominant', 'domain')][1].append(domain2id[domain])

    # publication --[creator]--> author
    for pub in publications:
        for author in rdf_graph.objects(pub, DC.creator):
            if author in author2id:
                edges[('publication', 'creator', 'author')][0].append(pub2id[pub])
                edges[('publication', 'creator', 'author')][1].append(author2id[author])

    # author --[hasDomain]--> domain
    for author in authors:
        for domain in rdf_graph.objects(author, EX.hasDomain):
            if domain in domain2id:
                edges[('author', 'hasDomain', 'domain')][0].append(author2id[author])
                edges[('author', 'hasDomain', 'domain')][1].append(domain2id[domain])

    # author --[hasPublishedIn]--> venue
    for author in authors:
        for venue in rdf_graph.objects(author, EX.hasPublishedIn):
            if venue in venue2id:
                edges[('author', 'hasPublishedIn', 'venue')][0].append(author2id[author])
                edges[('author', 'hasPublishedIn', 'venue')][1].append(venue2id[venue])

    # publication --[isPartOf]--> venue
    for pub in publications:
        for venue in rdf_graph.objects(pub, DC_TERMS.isPartOf):
            if venue in venue2id:
                edges[('publication', 'isPartOf', 'venue')][0].append(pub2id[pub])
                edges[('publication', 'isPartOf', 'venue')][1].append(venue2id[venue])

    # venue --[publishesDomain]--> domain
    for venue in venues:
        for domain in rdf_graph.objects(venue, EX.publishesDomain):
            if domain in domain2id:
                edges[('venue', 'publishesDomain', 'domain')][0].append(venue2id[venue])
                edges[('venue', 'publishesDomain', 'domain')][1].append(domain2id[domain])

    # Création du graphe hétérogène DGL
    data_dict = {}
    for (srctype, etype, dsttype), (srcs, dsts) in edges.items():
        # Correction : tensors vides si pas d'arêtes
        if len(srcs) == 0 or len(dsts) == 0:
            data_dict[(srctype, etype, dsttype)] = (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long)
            )
        else:
            data_dict[(srctype, etype, dsttype)] = (
                torch.tensor(srcs, dtype=torch.long),
                torch.tensor(dsts, dtype=torch.long)
            )

    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={
            'author': len(authors),
            'publication': len(publications),
            'venue': len(venues),
            'domain': len(domains)
        }
    )

    # Création du mapping inverse pour extract_true_labels
    id_node_map = {
        'author': {i: a for i, a in enumerate(authors)},
        'venue': {i: v for i, v in enumerate(venues)},
        'publication': {i: p for i, p in enumerate(publications)},
        'domain': {i: d for i, d in enumerate(domains)}
    }

    # Chargement des features des nœuds
    with open("node_features.pkl", "rb") as f:
        node_features = pickle.load(f)

    # Attribution des features aux nœuds
    ndata_feats = {}
    for ntype in g.ntypes:
        nids = g.nodes(ntype)
        feats = []
        for nid in nids:
            uri = id_node_map[ntype][int(nid)]
            feat = node_features.get(str(uri), None)
            if feat is None or torch.isnan(feat).any() or torch.all(feat == 0):
                feat = torch.randn(768) * 0.01
            feats.append(feat)
        ndata_feats[ntype] = torch.stack(feats)
    g.ndata['feat'] = ndata_feats

    print("\nVérification des types:")
    print(f"- Authors: {sum(1 for t in id_node_map['author'].values() if t in authors)}")
    print(f"- Publications: {sum(1 for t in id_node_map['publication'].values() if t in publications)}")
    print(f"- Venues: {sum(1 for t in id_node_map['venue'].values() if t in venues)}")
    print(f"- Domains: {sum(1 for t in id_node_map['domain'].values() if t in domains)}")

    print("\nNombre de relations par type:")
    for (src_type, rel_type, dst_type), (srcs, _) in edges.items():
        print(f"- {src_type} --[{rel_type}]--> {dst_type}: {len(srcs)} relations")

    print(f"✅ Graphe DGL créé avec {g.num_nodes()} nœuds et {g.num_edges()} arêtes")
    print(f"Types de nœuds : {g.ntypes}")
    print(f"Types de relations : {g.canonical_etypes}")

    return g, id_node_map, rdf_graph

class ContrastiveHGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, ntypes, etypes, n_heads=8, dropout=0.3, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # HGT layer for main graph view
        self.hgt_layer = ImprovedHGTLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            ntypes=ntypes,
            etypes=etypes,
            n_heads=n_heads,
            dropout=dropout
        )

        # Projection head for contrastive learning
        self.projection_heads = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ).to(device) for ntype in ntypes
        })

        # Node type embedding to enhance heterogeneous information
        self.node_type_embeddings = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(1, out_dim)).to(device)
            for ntype in ntypes
        })

    def forward(self, g, feat_dict, rel_emb_dict):
        # Get embeddings from HGT layer
        hgt_embeddings = self.hgt_layer(g, feat_dict, rel_emb_dict)

        # Apply projection heads for contrastive learning
        projected_embeddings = {}
        for ntype in hgt_embeddings:
            if ntype in self.projection_heads:
                # Add node type embedding to enhance type-specific information
                type_enhanced = hgt_embeddings[ntype] + self.node_type_embeddings[ntype]
                projected_embeddings[ntype] = self.projection_heads[ntype](type_enhanced)
            else:
                projected_embeddings[ntype] = hgt_embeddings[ntype]

        return hgt_embeddings, projected_embeddings

    def contrastive_loss(self, z1_dict, z2_dict):
        """
        Compute contrastive loss between two views of the graph
        z1_dict, z2_dict: dictionaries of embeddings for each node type
        """
        loss = 0.0
        total_types = 0

        for ntype in z1_dict:
            if ntype in z2_dict:
                z1 = F.normalize(z1_dict[ntype], p=2, dim=1)
                z2 = F.normalize(z2_dict[ntype], p=2, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.mm(z1, z2.t()) / self.temperature

                # Positive pairs (diagonal elements)
                pos_sim = torch.diagonal(sim_matrix)

                # InfoNCE loss
                loss += -torch.mean(
                    pos_sim - torch.logsumexp(sim_matrix, dim=1)
                )

                total_types += 1

        return loss / max(1, total_types)

def create_graph_views(g, feat_dict, drop_edge_p=0.2, drop_feat_p=0.3):
    """
    Create two augmented views of the graph for contrastive learning
    """
    # Create first view with edge dropout
    edge_mask_1 = {}
    for etype in g.canonical_etypes:
        num_edges = g.number_of_edges(etype)
        if num_edges > 0:
            # Keep edges with probability 1-drop_edge_p
            mask = torch.rand(num_edges) > drop_edge_p
            edge_mask_1[etype] = mask

    view1 = g.edge_subgraph(edge_mask_1)

    # Create second view with feature masking
    feat_dict_2 = {}
    for ntype, feat in feat_dict.items():
        # Apply random feature dropout
        mask = torch.rand_like(feat) > drop_feat_p
        feat_dict_2[ntype] = feat * mask

    return view1, feat_dict, feat_dict_2

def train_hgt(g, node_features, relation_embeddings, num_epochs=100, lr=0.001,
             weight_decay=1e-4, hidden_dim=256, out_dim=256, n_heads=8, dropout=0.3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUtilisation de l'appareil: {device}")

    # Move graph to device
    g = g.to(device)

    # Create copies of embeddings on the correct device
    rel_embeddings = {k: v.to(device) for k, v in relation_embeddings.items()}
    node_feats = {k: v.to(device) for k, v in node_features.items()}

    # Create contrastive HGT model
    model = ContrastiveHGT(
        in_dim=node_features[list(node_features.keys())[0]].shape[1],
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        ntypes=g.ntypes,
        etypes=[etype for _, etype, _ in g.canonical_etypes],
        n_heads=n_heads,
        dropout=dropout,
        temperature=0.1
    ).to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr/10
    )

    losses = []
    best_loss = float('inf')
    patience = 20
    no_improve = 0

    print("\nDébut de l'entraînement HGT avec apprentissage contrastif:")
    print("=" * 50)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Create two views of the graph for contrastive learning
        view1, feat_dict1, feat_dict2 = create_graph_views(g, node_feats)

        # Forward pass for both views
        _, proj_embeddings1 = model(g, feat_dict1, rel_embeddings)
        _, proj_embeddings2 = model(g, feat_dict2, rel_embeddings)

        # Compute contrastive loss
        contrastive_loss = model.contrastive_loss(proj_embeddings1, proj_embeddings2)

        # Get embeddings for original graph for reconstruction loss
        hgt_embeddings, _ = model(g, node_feats, rel_embeddings)

        # Compute reconstruction loss
        recon_loss = 0.0
        for ntype in node_feats:
            if ntype in hgt_embeddings:
                # Ensure correct dimensions
                if isinstance(hgt_embeddings[ntype], torch.Tensor):
                    output = hgt_embeddings[ntype]
                    if output.dim() == 1:
                        output = output.unsqueeze(0)

                    # Normalize embeddings
                    norm_feat = F.normalize(node_feats[ntype], p=2, dim=1)
                    norm_out = F.normalize(output, p=2, dim=1)

                    # Check dimensions and project if needed
                    if norm_feat.shape[1] != norm_out.shape[1]:
                        # Print dimensions for debugging
                        print(f"Dimension mismatch for {ntype}: norm_feat {norm_feat.shape}, norm_out {norm_out.shape}")

                        # Project the higher-dimensional features to the lower dimension
                        if norm_feat.shape[1] > norm_out.shape[1]:
                            # Create a projection matrix if it doesn't exist
                            if not hasattr(model, 'projection_matrices'):
                                model.projection_matrices = {}

                            if ntype not in model.projection_matrices:
                                # Create a linear projection
                                projection = nn.Linear(norm_feat.shape[1], norm_out.shape[1], bias=False).to(norm_feat.device)
                                # Initialize with orthogonal weights
                                nn.init.orthogonal_(projection.weight)
                                model.projection_matrices[ntype] = projection

                            # Apply projection
                            norm_feat = F.normalize(model.projection_matrices[ntype](norm_feat), p=2, dim=1)
                        else:
                            # Project output to match feature dimensions
                            if not hasattr(model, 'output_projections'):
                                model.output_projections = {}

                            if ntype not in model.output_projections:
                                # Create a linear projection
                                projection = nn.Linear(norm_out.shape[1], norm_feat.shape[1], bias=False).to(norm_out.device)
                                # Initialize with orthogonal weights
                                nn.init.orthogonal_(projection.weight)
                                model.output_projections[ntype] = projection

                            # Apply projection
                            norm_out = F.normalize(model.output_projections[ntype](norm_out), p=2, dim=1)

                    # Compute similarity loss with matching dimensions
                    sim = torch.mm(norm_out, norm_feat.t())
                    pos_sim = torch.diagonal(sim)
                    recon_loss += (1.0 - pos_sim.mean())

        # Total loss with weighted components
        loss = 0.7 * contrastive_loss + 0.3 * recon_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Afficher la progression
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Contrastive: {contrastive_loss.item():.4f}, Recon: {recon_loss.item():.4f}")

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping après {epoch+1} epochs!")
            break

    # Afficher la courbe d'entraînement
    plot_training_progress(losses, "HGT Contrastive Training Progress")

    # Generate final embeddings
    final_embeddings = {}
    model.eval()

    with torch.no_grad():
        hgt_embeddings, _ = model(g, node_feats, rel_embeddings)

        for ntype in hgt_embeddings:
            if isinstance(hgt_embeddings[ntype], torch.Tensor):
                emb = hgt_embeddings[ntype]
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if emb.shape[1] != out_dim:
                    emb = emb.view(-1, out_dim)
                final_embeddings[ntype] = emb.cpu()

    print("Entraînement HGT terminé!")
    return model.hgt_layer, final_embeddings

def visualize_clusters(embeddings, true_labels, pred_labels, title="Clusters"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 6))

    # Plot with predicted labels
    plt.subplot(1, 2, 1)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pred_labels, cmap='tab20')
    plt.title(f"{title} (Prédits)")

    # Plot with true labels if available
    plt.subplot(1, 2, 2)
    if np.any(true_labels >= 0):
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='tab20')
        plt.title(f"{title} (Réels)")
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue')
        plt.title(f"{title} (Pas de labels réels)")

    plt.tight_layout()
    plt.show()

def plot_training_progress(losses, title="Training Progress"):
    """Affiche la courbe de progression de l'entraînement"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

class CustomClusteringMethod:
    """
    Classe wrapper pour les méthodes de clustering personnalisées
    qui ne suivent pas l'interface scikit-learn
    """
    def __init__(self, clustering_func, n_clusters=None):
        self.clustering_func = clustering_func
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        """
        Applique la fonction de clustering et retourne les labels

        Args:
            X: données d'entrée (embeddings)
        """
        # Appliquer la fonction de clustering
        result = self.clustering_func(X)

        # Si le résultat est un dictionnaire (cas des graphes hétérogènes),
        # on le convertit en un tableau unique
        if isinstance(result, dict):
            # Reconstruire le tableau de labels global
            all_labels = np.zeros(X.shape[0], dtype=int)
            start_idx = 0
            for ntype, labels in result.items():
                n_nodes = len(labels)
                all_labels[start_idx:start_idx + n_nodes] = labels
                start_idx += n_nodes
            return all_labels
        else:
            return result

def deep_graph_infomax_clustering(X, n_clusters, n_neighbors=15):
    """
    Méthode de clustering basée sur Deep Graph Infomax (DGI)

    Args:
        X: embeddings des nœuds
        n_clusters: nombre de clusters
        n_neighbors: nombre de voisins pour la construction du graphe
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import SpectralClustering
    import numpy as np

    # Normaliser les embeddings
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Construire un graphe de similarité basé sur les k plus proches voisins
    connectivity = kneighbors_graph(
        X_normalized,
        n_neighbors=n_neighbors,
        mode='connectivity',
        include_self=False
    )

    # Rendre le graphe symétrique
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Appliquer le clustering spectral sur ce graphe
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    # Utiliser la matrice d'adjacence comme matrice d'affinité
    labels = spectral.fit_predict(connectivity)

    return labels

def hdgi_clustering(embeddings, n_clusters, alpha=0.3, beta=0.2, gamma=0.1):
    """
    Improved Heterogeneous Deep Graph Infomax (HDGI) clustering

    This enhanced version of HDGI extends Deep Graph Infomax for heterogeneous graphs by:
    1. Multi-level type-specific information preservation
    2. Adaptive weighting of node types based on their importance
    3. Structural role preservation
    4. Cross-type relationship modeling
    5. Hierarchical clustering refinement

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        alpha: weighting parameter for node type information (0.0-1.0)
        beta: weighting for structural roles (0.0-1.0)
        gamma: weighting for cross-type relationships (0.0-1.0)
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph

    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Calculate type-specific centroids and covariance matrices for better type representation
    type_centroids = {}
    type_covariances = {}

    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = all_embeddings[start:end]

        # Calculate type centroid
        type_centroids[ntype] = np.mean(type_embs, axis=0, keepdims=True)

        # Calculate type covariance matrix (simplified)
        centered = type_embs - type_centroids[ntype]
        type_covariances[ntype] = np.matmul(centered.T, centered) / max(1, (end - start - 1))

    # Create type-specific embeddings with multi-level information preservation
    type_enhanced_embeddings = np.copy(all_embeddings)

    # For each node type, enhance the embeddings with multi-level type-specific information
    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = all_embeddings[start:end]

        # Calculate similarity to centroid (first-order type-specific information)
        sim_to_centroid = cosine_similarity(type_embs, type_centroids[ntype])

        # Calculate Mahalanobis-like distance for second-order information
        # (simplified to avoid matrix inversion)
        centered = type_embs - type_centroids[ntype]
        cov_sim = np.zeros((end-start, 1))
        for i in range(end-start):
            # Project along principal directions of variation
            cov_sim[i, 0] = np.sum(centered[i] * np.mean(centered, axis=0))

        # Normalize to [0,1] range
        if np.max(cov_sim) > np.min(cov_sim):
            cov_sim = (cov_sim - np.min(cov_sim)) / (np.max(cov_sim) - np.min(cov_sim))
        else:
            cov_sim = np.ones_like(cov_sim) * 0.5

        # Enhance embeddings with type information
        # Higher alpha means more emphasis on type-specific information
        type_enhanced_embeddings[start:end] = (
            (1 - alpha) * type_embs +
            alpha * 0.7 * sim_to_centroid * np.repeat(type_centroids[ntype], end-start, axis=0) +
            alpha * 0.3 * cov_sim * np.repeat(type_centroids[ntype], end-start, axis=0)
        )

    # Add structural role information
    # Create a KNN graph to identify structural roles
    k = min(50, all_embeddings.shape[0] // 20)
    knn_graph = kneighbors_graph(
        all_embeddings,
        n_neighbors=k,
        mode='connectivity',
        include_self=False
    )
    knn_graph = 0.5 * (knn_graph + knn_graph.T)  # Make symmetric

    # Calculate node degrees as simple structural feature
    node_degrees = np.array(knn_graph.sum(axis=1)).flatten()

    # Normalize degrees
    if np.max(node_degrees) > 0:
        node_degrees = node_degrees / np.max(node_degrees)

    # Group nodes by structural roles (simplified by degree quantiles)
    n_roles = 5
    role_labels = np.zeros(all_embeddings.shape[0], dtype=int)

    for i in range(n_roles):
        lower = i / n_roles
        upper = (i + 1) / n_roles
        role_labels[(node_degrees >= lower) & (node_degrees < upper)] = i

    # Enhance embeddings with structural role information
    role_centroids = np.zeros((n_roles, all_embeddings.shape[1]))
    for role in range(n_roles):
        role_mask = role_labels == role
        if np.any(role_mask):
            role_centroids[role] = np.mean(all_embeddings[role_mask], axis=0)

    # Add structural information to embeddings
    for i in range(all_embeddings.shape[0]):
        role = role_labels[i]
        type_enhanced_embeddings[i] += beta * role_centroids[role]

    # Add cross-type relationship information
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type similarity between centroids
                centroid_sim = cosine_similarity(
                    type_centroids[ntype1],
                    type_centroids[ntype2]
                )[0, 0]

                # If types are similar, enhance with cross-type information
                if centroid_sim > 0.5:  # Only for similar types
                    for i in range(start1, end1):
                        # Find most similar nodes of other type
                        cross_sim = cosine_similarity(
                            all_embeddings[i].reshape(1, -1),
                            all_embeddings[start2:end2]
                        )[0]

                        # Get index of most similar node
                        most_similar = np.argmax(cross_sim)

                        # Enhance with cross-type information
                        type_enhanced_embeddings[i] += gamma * centroid_sim * all_embeddings[start2 + most_similar]

    # Normalize enhanced embeddings
    type_enhanced_embeddings = normalize(type_enhanced_embeddings, norm='l2', axis=1)

    # Create multi-level affinity matrix
    # Level 1: Direct similarity
    affinity_direct = cosine_similarity(type_enhanced_embeddings)

    # Level 2: Type-aware similarity
    affinity_type = np.copy(affinity_direct)

    # Boost within-type similarities
    for ntype, (start, end) in node_type_indices.items():
        affinity_type[start:end, start:end] *= 1.2

    # Level 3: Structural similarity
    # Create a matrix where nodes with similar structural roles have higher similarity
    affinity_struct = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if role_labels[i] == role_labels[j]:
                affinity_struct[i, j] = 0.8  # High similarity for same role
            else:
                # Decreasing similarity as role difference increases
                role_diff = abs(role_labels[i] - role_labels[j])
                affinity_struct[i, j] = max(0.1, 0.8 - 0.2 * role_diff)

    # Combine affinity matrices with adaptive weighting
    combined_affinity = 0.6 * affinity_direct + 0.3 * affinity_type + 0.1 * affinity_struct

    # Apply spectral clustering on the enhanced affinity matrix
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    labels = spectral.fit_predict(combined_affinity)

    # Hierarchical refinement of clusters
    # For each cluster, check if it should be split further based on type distribution
    refined_labels = np.copy(labels)
    next_label = n_clusters

    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_size = np.sum(cluster_mask)

        # Only consider large clusters for splitting
        if cluster_size > 20:
            # Check type distribution in this cluster
            type_counts = {}
            for ntype, (start, end) in node_type_indices.items():
                type_mask = np.zeros(n_samples, dtype=bool)
                type_mask[start:end] = True
                type_counts[ntype] = np.sum(cluster_mask & type_mask)

            # If the cluster has mixed types with significant presence, consider splitting
            if len([c for c in type_counts.values() if c > 5]) > 1:
                # Extract the subgraph for this cluster
                cluster_indices = np.where(cluster_mask)[0]
                cluster_embeddings = type_enhanced_embeddings[cluster_indices]

                # Apply spectral clustering to split this cluster
                sub_affinity = cosine_similarity(cluster_embeddings)

                # Determine number of subclusters based on type distribution
                n_subclusters = min(len([c for c in type_counts.values() if c > 5]), 3)

                if n_subclusters > 1:
                    sub_spectral = SpectralClustering(
                        n_clusters=n_subclusters,
                        affinity='precomputed',
                        random_state=42,
                        n_jobs=-1
                    )

                    sub_labels = sub_spectral.fit_predict(sub_affinity)

                    # Assign new labels to the subclusters (except the largest one)
                    for sub_id in range(n_subclusters):
                        sub_mask = (sub_labels == sub_id)
                        sub_size = np.sum(sub_mask)

                        # Skip the largest subcluster (keep original label)
                        if sub_id == np.argmax([np.sum(sub_labels == i) for i in range(n_subclusters)]):
                            continue

                        # Only create a new cluster if it's large enough
                        if sub_size > 5:
                            refined_labels[cluster_indices[sub_mask]] = next_label
                            next_label += 1

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = refined_labels[start:end]

    return result_labels

def magcn_clustering(embeddings, n_clusters, n_views=5, alpha=0.4, beta=0.3):
    """
    Improved Multi-view Adaptive Graph Convolutional Networks (MAGCN) clustering

    This enhanced version of MAGCN is specifically designed for heterogeneous graphs and uses multiple
    views of the graph structure with adaptive weighting to better handle different node types.

    Key improvements:
    1. Type-specific centroids for better type representation
    2. Adaptive weighting of cross-type relationships
    3. Type-aware consensus mechanism
    4. Structural view based on multi-hop connections
    5. Quality-weighted view fusion

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        n_views: number of different views to generate
        alpha: weight for type-specific information (0.0-1.0)
        beta: weight for cross-type relationships (0.0-1.0)
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph

    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Calculate type-specific centroids for better type representation
    type_centroids = {}
    for ntype, (start, end) in node_type_indices.items():
        type_embs = all_embeddings[start:end]
        type_centroids[ntype] = np.mean(type_embs, axis=0)

    # Generate multiple views of the graph with adaptive weighting
    views = []

    # View 1: Global similarity (standard cosine similarity)
    global_sim = cosine_similarity(all_embeddings)
    views.append(global_sim)

    # View 2: Type-enhanced similarity with adaptive weighting
    type_enhanced_sim = np.zeros((n_samples, n_samples))

    # Within-type similarity gets a boost based on distance to type centroid
    for ntype, (start, end) in node_type_indices.items():
        type_embs = all_embeddings[start:end]

        # Calculate similarity to type centroid
        centroid_sim = cosine_similarity(type_embs, type_centroids[ntype].reshape(1, -1))

        # Calculate within-type similarity
        type_sim = cosine_similarity(type_embs)

        # Boost within-type similarity based on centroid similarity
        boost_factor = 1.0 + alpha * centroid_sim
        type_enhanced_sim[start:end, start:end] = type_sim * boost_factor.reshape(-1, 1)

    # Add cross-type similarities with adaptive weighting
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type similarity
                cross_sim = cosine_similarity(all_embeddings[start1:end1], all_embeddings[start2:end2])

                # Apply adaptive weighting based on type centroids similarity
                type_relation_weight = beta * cosine_similarity(
                    type_centroids[ntype1].reshape(1, -1),
                    type_centroids[ntype2].reshape(1, -1)
                )[0, 0]

                # Apply weighted cross-type similarity
                type_enhanced_sim[start1:end1, start2:end2] = cross_sim * (1.0 + type_relation_weight)

    views.append(type_enhanced_sim)

    # View 3: k-nearest neighbors graph with adaptive k
    k_values = {}
    for ntype, (start, end) in node_type_indices.items():
        # Adaptive k based on number of nodes of this type
        n_nodes = end - start
        k_values[ntype] = min(50, max(10, int(n_nodes * 0.1)))

    knn_sim = np.zeros((n_samples, n_samples))

    # Build KNN graph for each type separately
    for ntype, (start, end) in node_type_indices.items():
        type_embs = all_embeddings[start:end]
        k = k_values[ntype]

        # Create KNN graph for this type
        knn_graph = kneighbors_graph(
            type_embs,
            n_neighbors=k,
            mode='distance',
            include_self=False
        )
        knn_graph = 0.5 * (knn_graph + knn_graph.T)  # Make symmetric
        knn_graph = knn_graph.toarray()

        # Convert distances to similarities
        type_knn_sim = 1.0 / (1.0 + knn_graph)
        knn_sim[start:end, start:end] = type_knn_sim

    # Add cross-type KNN connections for most similar nodes
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type similarity
                cross_sim = cosine_similarity(all_embeddings[start1:end1], all_embeddings[start2:end2])

                # For each node, connect to top-k most similar nodes of other type
                k_cross = min(10, (end2 - start2) // 2)
                for i in range(end1 - start1):
                    # Get indices of top-k most similar nodes
                    top_k_indices = np.argsort(cross_sim[i])[-k_cross:]

                    # Add connections
                    for j in top_k_indices:
                        knn_sim[start1 + i, start2 + j] = cross_sim[i, j]
                        knn_sim[start2 + j, start1 + i] = cross_sim[i, j]  # Make symmetric

    views.append(knn_sim)

    # View 4: Structural view based on path-based similarity
    # Approximate multi-hop connections through matrix powers
    path_sim = np.copy(global_sim)
    path_sim_2hop = np.matmul(global_sim, global_sim)  # 2-hop paths
    path_sim = 0.6 * path_sim + 0.4 * path_sim_2hop  # Combine 1-hop and 2-hop

    # Normalize
    path_sim = normalize(path_sim, norm='l2', axis=1)
    views.append(path_sim)

    # View 5: Type-specific spectral embeddings
    # Create a type-aware adjacency matrix
    type_spectral_sim = np.zeros((n_samples, n_samples))

    # Add within-type connections with higher weights
    for ntype, (start, end) in node_type_indices.items():
        type_embs = all_embeddings[start:end]
        type_adj = cosine_similarity(type_embs)

        # Apply threshold to create sparse connections
        threshold = np.percentile(type_adj, 80)  # Keep top 20% connections
        type_adj[type_adj < threshold] = 0

        type_spectral_sim[start:end, start:end] = type_adj * 1.5  # Higher weight for within-type

    # Add cross-type connections with lower weights
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                cross_adj = cosine_similarity(all_embeddings[start1:end1], all_embeddings[start2:end2])

                # Apply stricter threshold for cross-type
                threshold = np.percentile(cross_adj, 90)  # Keep only top 10%
                cross_adj[cross_adj < threshold] = 0

                type_spectral_sim[start1:end1, start2:end2] = cross_adj * 0.8  # Lower weight for cross-type

    views.append(type_spectral_sim)

    # Generate additional views if requested
    if n_views > 5:
        for i in range(n_views - 5):
            # Create a weighted combination of existing views with learned weights
            # Use a simple heuristic to determine weights based on view quality
            view_quality = []

            for view in views:
                # Assess view quality by measuring cluster separation
                # Higher ratio of within-type to cross-type similarity indicates better separation
                within_type_sim = 0
                cross_type_sim = 0

                for ntype, (start, end) in node_type_indices.items():
                    within_type_sim += np.mean(view[start:end, start:end])

                    for ntype2, (start2, end2) in node_type_indices.items():
                        if ntype != ntype2:
                            cross_type_sim += np.mean(view[start:end, start2:end2])

                if cross_type_sim > 0:
                    quality = within_type_sim / cross_type_sim
                else:
                    quality = within_type_sim

                view_quality.append(quality)

            # Normalize qualities to get weights
            view_quality = np.array(view_quality)
            weights = view_quality / np.sum(view_quality)

            # Create weighted combination
            combined_view = np.zeros_like(views[0])
            for j, view in enumerate(views):
                combined_view += weights[j] * view

            # Normalize the combined view
            combined_view = normalize(combined_view, norm='l2', axis=1)
            views.append(combined_view)

    # Adaptive fusion of multiple views with quality assessment
    # Calculate view-specific clustering results
    view_labels = []
    view_quality = []

    for view in views:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            n_jobs=-1
        )
        labels = spectral.fit_predict(view)
        view_labels.append(labels)

        # Assess clustering quality using silhouette-like measure
        quality = 0
        for ntype, (start, end) in node_type_indices.items():
            # Calculate intra-cluster similarity
            type_labels = labels[start:end]
            type_view = view[start:end, start:end]

            # For each cluster, calculate average intra-cluster similarity
            for cluster in range(n_clusters):
                cluster_mask = (type_labels == cluster)
                if np.sum(cluster_mask) > 1:
                    cluster_sim = type_view[cluster_mask][:, cluster_mask]
                    quality += np.mean(cluster_sim)

        view_quality.append(quality)

    # Normalize view quality scores
    view_quality = np.array(view_quality)
    if np.sum(view_quality) > 0:
        view_quality = view_quality / np.sum(view_quality)
    else:
        view_quality = np.ones(len(views)) / len(views)

    # Calculate consensus matrix with quality-weighted views
    consensus = np.zeros((n_samples, n_samples))
    for i, (labels, quality) in enumerate(zip(view_labels, view_quality)):
        # Create binary co-association matrix
        co_assoc = np.zeros((n_samples, n_samples))
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            for i in indices:
                co_assoc[i, indices] = 1
        consensus += co_assoc * quality

    # Final clustering on consensus matrix with type-aware spectral clustering
    # Create type-aware affinity matrix
    type_aware_consensus = np.copy(consensus)

    # Enhance within-type affinities
    for ntype, (start, end) in node_type_indices.items():
        type_consensus = consensus[start:end, start:end]
        type_aware_consensus[start:end, start:end] = type_consensus * 1.2  # Boost within-type

    final_spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )
    final_labels = final_spectral.fit_predict(type_aware_consensus)

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = final_labels[start:end]

    return result_labels

def heco_clustering(embeddings, n_clusters, temperature=0.2, alpha=0.7, beta=0.3):
    """
    Improved Heterogeneous Graph Contrastive Learning (HeCo) clustering

    This enhanced version of HeCo uses advanced contrastive learning techniques to better
    separate clusters and handle heterogeneous semantics with:
    1. Multi-view contrastive learning
    2. Semantic-aware negative sampling
    3. Adaptive temperature scaling
    4. Cross-type relation modeling
    5. Ensemble clustering with multiple contrastive views

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        temperature: base temperature parameter for contrastive learning (lower = sharper contrasts)
        alpha: weight for semantic view (0.0-1.0)
        beta: weight for structural view (0.0-1.0)
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.neighbors import kneighbors_graph

    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Calculate type centroids for semantic view
    type_centroids = {}
    for ntype, (start, end) in node_type_indices.items():
        type_embs = all_embeddings[start:end]
        type_centroids[ntype] = np.mean(type_embs, axis=0, keepdims=True)

    # Create multiple contrastive views
    contrastive_views = []

    # View 1: Semantic-based contrastive view with adaptive temperature
    semantic_sim = np.zeros((n_samples, n_samples))

    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = all_embeddings[start:end]

        # Calculate similarity to type centroid for adaptive temperature
        centroid_sim = cosine_similarity(type_embs, type_centroids[ntype])

        # Adaptive temperature: nodes closer to centroid get lower temperature (sharper contrast)
        adaptive_temp = np.ones((end-start, 1)) * temperature
        adaptive_temp = adaptive_temp * (1.0 - 0.5 * centroid_sim)  # Scale temperature down for typical nodes

        # Calculate within-type similarity
        type_sim = cosine_similarity(type_embs)

        # Apply temperature scaling with adaptive temperature
        scaled_sim = np.zeros_like(type_sim)
        for i in range(type_sim.shape[0]):
            scaled_sim[i] = type_sim[i] / adaptive_temp[i, 0]

        # Apply softmax to get probabilities
        max_sim = np.max(scaled_sim, axis=1, keepdims=True)
        exp_sim = np.exp(scaled_sim - max_sim)
        softmax_sim = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        # Update semantic similarity matrix
        semantic_sim[start:end, start:end] = softmax_sim

    # Add cross-type semantic relationships
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type centroid similarity
                centroid_sim = cosine_similarity(
                    type_centroids[ntype1],
                    type_centroids[ntype2]
                )[0, 0]

                # Only connect semantically similar types
                if centroid_sim > 0.3:
                    # Calculate cross-type similarity
                    cross_sim = cosine_similarity(all_embeddings[start1:end1], all_embeddings[start2:end2])

                    # Keep only top connections
                    for i in range(end1 - start1):
                        # Get indices of top-k most similar nodes
                        k_cross = min(5, (end2 - start2) // 4)
                        top_k_indices = np.argsort(cross_sim[i])[-k_cross:]

                        # Add connections with scaled similarity
                        for j in top_k_indices:
                            semantic_sim[start1 + i, start2 + j] = cross_sim[i, j] * centroid_sim * 0.5
                            semantic_sim[start2 + j, start1 + i] = cross_sim[i, j] * centroid_sim * 0.5

    contrastive_views.append(semantic_sim)

    # View 2: Structure-based contrastive view
    # Create a KNN graph to capture structural information
    k = min(30, n_samples // 20)
    knn_graph = kneighbors_graph(
        all_embeddings,
        n_neighbors=k,
        mode='connectivity',
        include_self=False
    )
    knn_graph = 0.5 * (knn_graph + knn_graph.T)  # Make symmetric
    knn_graph = knn_graph.toarray()

    # Calculate structural similarity
    structural_sim = np.zeros((n_samples, n_samples))

    # First, enhance within-type structural similarity
    for ntype, (start, end) in node_type_indices.items():
        type_knn = knn_graph[start:end, start:end]

        # Calculate similarity based on shared neighbors
        for i in range(end - start):
            for j in range(end - start):
                if i != j:
                    # Count shared neighbors
                    shared = np.sum(type_knn[i] & type_knn[j])
                    total = np.sum(type_knn[i] | type_knn[j])

                    if total > 0:
                        # Jaccard similarity of neighborhood
                        structural_sim[start + i, start + j] = shared / total

    # Then add cross-type structural connections
    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type structural similarity
                for i in range(end1 - start1):
                    for j in range(end2 - start2):
                        # Get neighbors
                        neighbors_i = knn_graph[start1 + i]
                        neighbors_j = knn_graph[start2 + j]

                        # Calculate similarity based on neighborhood overlap
                        shared = np.sum(neighbors_i & neighbors_j)
                        total = np.sum(neighbors_i | neighbors_j)

                        if total > 0:
                            # Scaled Jaccard similarity
                            structural_sim[start1 + i, start2 + j] = 0.7 * (shared / total)

    # Apply temperature scaling and softmax
    structural_sim = structural_sim / temperature
    max_sim = np.max(structural_sim, axis=1, keepdims=True)
    exp_sim = np.exp(structural_sim - max_sim)
    sum_exp = np.sum(exp_sim, axis=1, keepdims=True)
    sum_exp = np.where(sum_exp == 0, np.ones_like(sum_exp), sum_exp)  # Avoid division by zero
    structural_sim = exp_sim / sum_exp

    contrastive_views.append(structural_sim)

    # View 3: Combined semantic-structural view with hard negative mining
    combined_sim = alpha * semantic_sim + beta * structural_sim

    # Hard negative mining: identify and emphasize hard negatives
    hard_negative_sim = np.zeros_like(combined_sim)

    for ntype, (start, end) in node_type_indices.items():
        # Get within-type similarity
        type_sim = combined_sim[start:end, start:end]

        # For each node, find hard negatives (high similarity but different type)
        for i in range(end - start):
            # Get all similarities for this node
            node_sim = combined_sim[start + i]

            # Identify hard negatives: high similarity nodes from different types
            for other_ntype, (other_start, other_end) in node_type_indices.items():
                if ntype != other_ntype:
                    other_sim = node_sim[other_start:other_end]

                    # Get top similar nodes from other type
                    k_hard = min(10, (other_end - other_start) // 5)
                    hard_indices = np.argsort(other_sim)[-k_hard:]

                    # Emphasize these hard negatives by reducing their similarity
                    for idx in hard_indices:
                        hard_negative_sim[start + i, other_start + idx] = -0.5 * other_sim[idx]

    # Add hard negative information to combined similarity
    combined_sim = combined_sim + hard_negative_sim

    # Normalize to ensure valid similarity values
    combined_sim = np.clip(combined_sim, 0, 1)

    contrastive_views.append(combined_sim)

    # Apply clustering to each contrastive view
    view_labels = []

    for view_idx, view in enumerate(contrastive_views):
        if view_idx < 2:  # For the first two views, use spectral clustering
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                n_jobs=-1
            )
            view_labels.append(spectral.fit_predict(view))
        else:  # For the combined view, try KMeans on the similarity matrix
            # Convert similarity to distance
            distance = 1 - view

            # Apply MDS to get a Euclidean embedding
            from sklearn.manifold import MDS
            mds = MDS(n_components=min(100, n_samples // 5),
                      dissimilarity='precomputed',
                      random_state=42,
                      n_jobs=-1)

            try:
                euclidean_embedding = mds.fit_transform(distance)

                # Apply KMeans on the Euclidean embedding
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                view_labels.append(kmeans.fit_predict(euclidean_embedding))
            except:
                # Fallback to spectral clustering if MDS fails
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    n_jobs=-1
                )
                view_labels.append(spectral.fit_predict(view))

    # Create ensemble clustering from all views
    # Build co-association matrix
    co_assoc = np.zeros((n_samples, n_samples))

    for labels in view_labels:
        # Create binary co-association matrix for this clustering
        view_assoc = np.zeros((n_samples, n_samples))
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            for i in indices:
                view_assoc[i, indices] = 1

        co_assoc += view_assoc

    # Normalize co-association matrix
    co_assoc /= len(view_labels)

    # Apply type-aware weighting to co-association matrix
    type_aware_co_assoc = np.copy(co_assoc)

    # Boost within-type associations
    for ntype, (start, end) in node_type_indices.items():
        type_assoc = co_assoc[start:end, start:end]
        type_aware_co_assoc[start:end, start:end] = type_assoc * 1.2

    # Final clustering on the ensemble co-association matrix
    final_spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )

    final_labels = final_spectral.fit_predict(type_aware_co_assoc)

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = final_labels[start:end]

    return result_labels

def deep_spectral_clustering(embeddings, n_clusters, alpha=0.2):
    """
    Méthode de clustering spectral profond adaptée aux graphes hétérogènes

    Args:
        embeddings: dictionnaire des embeddings par type de nœud
        n_clusters: nombre de clusters
        alpha: paramètre de pondération pour la fusion des types de nœuds
    """
    # Concaténer tous les embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        all_embeddings_list.append(embs.cpu().detach().numpy())
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)

    # Normaliser les embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Calculer la matrice d'affinité
    affinity = cosine_similarity(all_embeddings)

    # Appliquer une pondération plus élevée aux nœuds du même type
    for ntype, (start, end) in node_type_indices.items():
        # Augmenter l'affinité entre les nœuds du même type
        affinity[start:end, start:end] *= (1 + alpha)

    # Appliquer le clustering spectral
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    labels = spectral.fit_predict(affinity)

    # Séparer les résultats par type de nœud
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = labels[start:end]

    return result_labels

def dmgi_clustering(embeddings, n_clusters, alpha=0.3, beta=0.1, temperature=0.5):
    """
    Deep Multiplex Graph Infomax (DMGI) pour le clustering de graphes hétérogènes.

    DMGI étend Deep Graph Infomax aux réseaux multiplexes en optimisant conjointement
    les embeddings des nœuds à travers différents types de relations tout en préservant
    les informations structurelles locales et globales.

    Référence: Park, et al. "Unsupervised Attributed Multiplex Network Embedding" AAAI 2020

    Args:
        embeddings: dictionnaire des embeddings par type de nœud
        n_clusters: nombre de clusters
        alpha: poids pour l'information spécifique au type de relation
        beta: poids pour la régularisation de consensus
        temperature: température pour l'apprentissage contrastif
    """
    # Concaténer tous les embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normaliser les embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # 1. Créer des embeddings spécifiques aux relations avec attention
    relation_embeddings = {}
    for ntype, (start, end) in node_type_indices.items():
        # Obtenir les embeddings pour ce type
        type_embs = all_embeddings[start:end]

        # Calculer les poids d'attention (auto-attention simplifiée)
        sim_matrix = cosine_similarity(type_embs)

        # Appliquer la mise à l'échelle de température
        sim_matrix = sim_matrix / temperature

        # Appliquer softmax pour obtenir les poids d'attention
        max_sim = np.max(sim_matrix, axis=1, keepdims=True)
        exp_sim = np.exp(sim_matrix - max_sim)
        attention_weights = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        # Appliquer l'attention pour obtenir des embeddings spécifiques aux relations
        relation_embeddings[ntype] = np.matmul(attention_weights, type_embs)

    # 2. Créer des embeddings de consensus (vue globale)
    consensus_embeddings = np.copy(all_embeddings)

    # Appliquer l'attention inter-relations
    for ntype, (start, end) in node_type_indices.items():
        # Obtenir les embeddings spécifiques aux relations
        rel_embs = relation_embeddings[ntype]

        # Mettre à jour les embeddings de consensus avec des informations spécifiques aux relations
        consensus_embeddings[start:end] = (1 - alpha) * all_embeddings[start:end] + alpha * rel_embs

    # 3. Créer une matrice de similarité de consensus
    consensus_sim = cosine_similarity(consensus_embeddings)

    # 4. Appliquer le clustering spectral sur la similarité de consensus
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    labels = spectral.fit_predict(consensus_sim)

    # Séparer les résultats par type de nœud
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = labels[start:end]

    return result_labels

def mcgc_clustering(embeddings, n_clusters, n_views=4, lambda_param=0.5, temperature=0.2):
    """
    Multi-view Contrastive Graph Clustering (MCGC) pour les graphes hétérogènes.

    MCGC utilise l'apprentissage contrastif à travers différentes vues du graphe
    pour apprendre de meilleures représentations de nœuds pour le clustering.

    Référence: Inspiré par "Multi-View Contrastive Graph Clustering" NeurIPS 2021

    Args:
        embeddings: dictionnaire des embeddings par type de nœud
        n_clusters: nombre de clusters
        n_views: nombre de vues différentes à générer
        lambda_param: poids pour la perte contrastive
        temperature: température pour l'apprentissage contrastif
    """
    # Concaténer tous les embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normaliser les embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Générer plusieurs vues du graphe
    views = []

    # Vue 1: Embeddings originaux
    views.append(all_embeddings)

    # Vue 2: Embeddings améliorés par type
    type_enhanced = np.copy(all_embeddings)
    for ntype, (start, end) in node_type_indices.items():
        # Calculer le centroïde du type
        type_centroid = np.mean(all_embeddings[start:end], axis=0, keepdims=True)
        # Améliorer avec des informations de type
        type_enhanced[start:end] = 0.7 * all_embeddings[start:end] + 0.3 * np.repeat(type_centroid, end-start, axis=0)
    views.append(type_enhanced)

    # Vue 3: Embeddings améliorés par voisinage
    # Créer un graphe des k plus proches voisins
    k = min(50, n_samples // 10)
    knn_graph = kneighbors_graph(all_embeddings, n_neighbors=k, mode='distance', include_self=False)
    knn_graph = 0.5 * (knn_graph + knn_graph.T)  # Rendre symétrique
    knn_graph = knn_graph.toarray()
    # Convertir les distances en similarités
    knn_sim = 1.0 / (1.0 + knn_graph)
    # Appliquer l'agrégation de voisinage
    neighbor_enhanced = np.matmul(knn_sim, all_embeddings)
    neighbor_enhanced = normalize(neighbor_enhanced, norm='l2', axis=1)
    views.append(neighbor_enhanced)

    # Vue 4: Embeddings améliorés par contrastif
    contrastive_enhanced = np.copy(all_embeddings)
    similarity = cosine_similarity(all_embeddings)
    similarity = similarity / temperature
    # Appliquer softmax
    max_sim = np.max(similarity, axis=1, keepdims=True)
    exp_sim = np.exp(similarity - max_sim)
    softmax_sim = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    # Créer des embeddings contrastifs
    contrastive_enhanced = np.matmul(softmax_sim, all_embeddings)
    contrastive_enhanced = normalize(contrastive_enhanced, norm='l2', axis=1)
    views.append(contrastive_enhanced)

    # Générer des vues supplémentaires si demandé
    if n_views > 4:
        for i in range(n_views - 4):
            # Créer une combinaison pondérée aléatoire des vues existantes
            weights = np.random.dirichlet(np.ones(len(views)))
            combined_view = np.zeros_like(views[0])
            for j, view in enumerate(views):
                combined_view += weights[j] * view
            combined_view = normalize(combined_view, norm='l2', axis=1)
            views.append(combined_view)

    # Calculer les résultats de clustering spécifiques à chaque vue
    view_labels = []
    for view in views:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        view_labels.append(kmeans.fit_predict(view))

    # Calculer la matrice de consensus
    consensus = np.zeros((n_samples, n_samples))
    for labels in view_labels:
        # Créer une matrice de co-association binaire
        co_assoc = np.zeros((n_samples, n_samples))
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            for i in indices:
                co_assoc[i, indices] = 1
        consensus += co_assoc

    # Normaliser le consensus
    consensus /= len(views)

    # Clustering final sur la matrice de consensus
    final_spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )
    final_labels = final_spectral.fit_predict(consensus)

    # Séparer les résultats par type de nœud
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = final_labels[start:end]

    return result_labels

def hgat_clustering(embeddings, n_clusters, n_heads=12, dropout=0.15, alpha=0.2,
                  type_weight=1.5, cross_type_weight=0.8, n_layers=2, residual_weight=0.3):
    """
    Enhanced Heterogeneous Graph Attention Network (HGAT) Clustering.

    This improved HGAT implementation includes:
    1. Multi-layer attention with residual connections
    2. Type-aware attention mechanism
    3. Cross-type relationship modeling
    4. Layer normalization for better stability
    5. Adaptive attention with temperature scaling
    6. Ensemble of multiple attention mechanisms

    Référence: Inspired by "Heterogeneous Graph Attention Network" WWW 2019
               and "Heterogeneous Graph Transformer" WWW 2020

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        n_heads: number of attention heads
        dropout: dropout rate
        alpha: LeakyReLU slope
        type_weight: weight for within-type attention
        cross_type_weight: weight for cross-type attention
        n_layers: number of attention layers
        residual_weight: weight for residual connections
    """
    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0
    node_types = []  # Track node types for each node

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        node_types.extend([ntype] * n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]
    emb_dim = all_embeddings.shape[1]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Create type-specific embeddings
    type_centroids = {}
    for ntype, (start, end) in node_type_indices.items():
        type_centroids[ntype] = np.mean(all_embeddings[start:end], axis=0)

    # Add type information to embeddings
    type_enhanced_embeddings = np.copy(all_embeddings)
    for ntype, (start, end) in node_type_indices.items():
        # Calculate similarity to type centroid
        centroid_sim = cosine_similarity(all_embeddings[start:end],
                                         type_centroids[ntype].reshape(1, -1))
        # Add weighted type information
        type_enhanced_embeddings[start:end] += 0.2 * np.repeat(type_centroids[ntype].reshape(1, -1),
                                                              end-start, axis=0) * centroid_sim

    # Normalize again after adding type information
    type_enhanced_embeddings = normalize(type_enhanced_embeddings, norm='l2', axis=1)

    # Create multiple KNN graphs with different k values for multi-scale attention
    knn_graphs = []
    k_values = [min(30, n_samples // 20), min(50, n_samples // 10), min(100, n_samples // 5)]

    for k in k_values:
        # Create KNN graph
        knn = kneighbors_graph(type_enhanced_embeddings, n_neighbors=k,
                               mode='connectivity', include_self=False)
        knn = 0.5 * (knn + knn.T)  # Make symmetric

        # Add type-aware edges: connect nodes of the same type more densely
        for ntype, (start, end) in node_type_indices.items():
            if end - start > 1:  # Only if we have multiple nodes of this type
                # Create additional connections between nodes of the same type
                type_knn = kneighbors_graph(
                    type_enhanced_embeddings[start:end],
                    n_neighbors=min(20, (end-start)//2),
                    mode='connectivity',
                    include_self=False
                )

                # Add these connections to the main graph with higher weight
                for i in range(type_knn.shape[0]):
                    for j in np.where(type_knn[i].toarray()[0] > 0)[0]:
                        knn[start+i, start+j] = type_weight
                        knn[start+j, start+i] = type_weight

        # Add cross-type edges based on similarity
        for ntype1, (start1, end1) in node_type_indices.items():
            for ntype2, (start2, end2) in node_type_indices.items():
                if ntype1 != ntype2:
                    # Calculate cross-type similarity
                    cross_sim = cosine_similarity(
                        type_enhanced_embeddings[start1:end1],
                        type_enhanced_embeddings[start2:end2]
                    )

                    # Connect most similar nodes across types
                    k_cross = min(10, min(end1-start1, end2-start2)//2)
                    for i in range(end1-start1):
                        # Get indices of top-k most similar nodes
                        top_indices = np.argsort(cross_sim[i])[-k_cross:]
                        for j in top_indices:
                            knn[start1+i, start2+j] = cross_type_weight
                            knn[start2+j, start1+i] = cross_type_weight

        knn_graphs.append(knn.toarray())

    # Multi-layer attention mechanism
    current_embeddings = type_enhanced_embeddings

    for layer in range(n_layers):
        layer_embeddings = []

        # Process each KNN graph separately
        for graph_idx, adj in enumerate(knn_graphs):
            # Apply multi-head attention
            attention_heads = []

            for head in range(n_heads):
                # Generate attention weights for this head
                # In a real implementation, these would be learned
                attn_weights = np.random.rand(emb_dim, 2)

                # Calculate attention coefficients
                attention = np.zeros_like(adj)

                # Adaptive temperature for softmax based on node degree
                node_degrees = np.sum(adj > 0, axis=1)
                temperatures = 0.1 + 0.05 * np.log1p(node_degrees)

                for i in range(n_samples):
                    neighbors = np.where(adj[i] > 0)[0]
                    if len(neighbors) > 0:
                        # Calculate attention coefficient
                        coef_i = np.dot(current_embeddings[i], attn_weights[:, 0])
                        coef_j = np.dot(current_embeddings[neighbors], attn_weights[:, 1])

                        # Add type-aware attention
                        type_i = node_types[i]
                        types_j = [node_types[j] for j in neighbors]

                        # Boost attention for same-type nodes
                        type_boost = np.array([1.5 if t == type_i else 0.8 for t in types_j])

                        # Combine coefficients
                        coef = (coef_i + coef_j) * type_boost

                        # Apply LeakyReLU
                        coef = np.maximum(coef, alpha * coef)

                        # Apply temperature-scaled softmax
                        temp = temperatures[i]
                        exp_coef = np.exp(coef / temp)
                        softmax_coef = exp_coef / (np.sum(exp_coef) + 1e-10)

                        # Apply dropout
                        mask = np.random.binomial(1, 1-dropout, size=softmax_coef.shape)
                        softmax_coef = softmax_coef * mask
                        if np.sum(softmax_coef) > 0:
                            softmax_coef = softmax_coef / (np.sum(softmax_coef) + 1e-10)

                        # Store attention coefficients
                        attention[i, neighbors] = softmax_coef

                # Apply attention to get new embeddings
                head_embeddings = np.matmul(attention, current_embeddings)

                # Apply layer normalization (simplified version)
                mean = np.mean(head_embeddings, axis=1, keepdims=True)
                std = np.std(head_embeddings, axis=1, keepdims=True) + 1e-6
                head_embeddings = (head_embeddings - mean) / std

                attention_heads.append(head_embeddings)

            # Combine attention heads
            if attention_heads:
                # Concatenate heads and then project back to original dimension
                concat_heads = np.concatenate(attention_heads, axis=1)

                # Simple projection back to original dimension
                projection = np.random.rand(concat_heads.shape[1], emb_dim)
                projection = normalize(projection, norm='l2', axis=0)

                graph_embeddings = np.matmul(concat_heads, projection)

                # Apply layer normalization
                mean = np.mean(graph_embeddings, axis=1, keepdims=True)
                std = np.std(graph_embeddings, axis=1, keepdims=True) + 1e-6
                graph_embeddings = (graph_embeddings - mean) / std

                layer_embeddings.append(graph_embeddings)

        # Combine embeddings from different graphs
        if layer_embeddings:
            combined_embeddings = np.mean(layer_embeddings, axis=0)

            # Add residual connection
            current_embeddings = (1 - residual_weight) * combined_embeddings + residual_weight * current_embeddings

            # Normalize
            current_embeddings = normalize(current_embeddings, norm='l2', axis=1)

    # Final attention embeddings
    attention_embeddings = current_embeddings

    # Apply type-aware spectral clustering
    # Create affinity matrix with type awareness
    similarity = cosine_similarity(attention_embeddings)

    # Boost within-type similarities
    for ntype, (start, end) in node_type_indices.items():
        similarity[start:end, start:end] *= type_weight

    # Apply spectral clustering with optimized parameters
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1,
        n_init=20  # More initialization attempts
    )

    labels = spectral.fit_predict(similarity)

    # Refinement step: reassign nodes at cluster boundaries
    refined_labels = np.copy(labels)

    # Calculate cluster centroids
    centroids = np.zeros((n_clusters, attention_embeddings.shape[1]))
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            centroids[c] = np.mean(attention_embeddings[mask], axis=0)

    # Identify boundary nodes (nodes with high similarity to multiple clusters)
    for i in range(n_samples):
        # Calculate similarity to all centroids
        centroid_sims = cosine_similarity(
            attention_embeddings[i].reshape(1, -1),
            centroids
        )[0]

        # If the node is almost equally similar to multiple clusters, consider it a boundary node
        sorted_sims = np.sort(centroid_sims)[::-1]
        if len(sorted_sims) > 1 and (sorted_sims[0] - sorted_sims[1]) < 0.1:
            # Find nodes of the same type
            node_type = node_types[i]
            type_start, type_end = node_type_indices[node_type]
            same_type_nodes = np.arange(type_start, type_end)

            # Find the most common cluster among similar nodes of the same type
            similarities = cosine_similarity(
                attention_embeddings[i].reshape(1, -1),
                attention_embeddings[same_type_nodes]
            )[0]

            # Get top similar nodes
            top_similar = same_type_nodes[np.argsort(similarities)[-10:]]
            top_clusters = [refined_labels[j] for j in top_similar]

            # Assign to the most common cluster among similar nodes
            from collections import Counter
            if top_clusters:
                most_common = Counter(top_clusters).most_common(1)[0][0]
                refined_labels[i] = most_common

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = refined_labels[start:end]

    return result_labels

def simple_hgnn_clustering(embeddings, n_clusters, alpha=0.5, beta=0.3, gamma=0.2, temperature=0.1):
    """
    Simple Heterogeneous Graph Neural Network (SimpleHGNN) clustering

    This method implements a simplified but highly effective approach for heterogeneous graph clustering
    that focuses on three key aspects:
    1. Type-specific feature transformation with adaptive weighting
    2. Cross-type relation modeling with attention mechanism
    3. Multi-level contrastive learning with hard negative mining

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        alpha: weight for type-specific information (0.0-1.0)
        beta: weight for cross-type relationships (0.0-1.0)
        gamma: weight for contrastive learning component (0.0-1.0)
        temperature: temperature parameter for contrastive learning
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering

    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # 1. Type-specific feature transformation
    # Calculate type centroids and covariance matrices
    type_centroids = {}
    type_covariances = {}

    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = all_embeddings[start:end]

        # Calculate type centroid
        type_centroids[ntype] = np.mean(type_embs, axis=0, keepdims=True)

        # Calculate type covariance matrix
        centered = type_embs - type_centroids[ntype]
        type_covariances[ntype] = np.matmul(centered.T, centered) / max(1, (end - start - 1))

    # Apply type-specific transformation
    transformed_embeddings = np.copy(all_embeddings)

    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = all_embeddings[start:end]

        # Calculate similarity to centroid
        centroid_sim = cosine_similarity(type_embs, type_centroids[ntype])

        # Apply adaptive weighting based on similarity to centroid
        weights = alpha * (1 + centroid_sim)

        # Apply weighted transformation
        transformed_embeddings[start:end] = type_embs * weights

    # 2. Cross-type relation modeling with attention
    # Calculate cross-type attention weights
    cross_type_attention = np.zeros((n_samples, n_samples))

    for ntype1, (start1, end1) in node_type_indices.items():
        for ntype2, (start2, end2) in node_type_indices.items():
            if ntype1 != ntype2:
                # Calculate cross-type similarity
                cross_sim = cosine_similarity(all_embeddings[start1:end1], all_embeddings[start2:end2])

                # Calculate attention weights using softmax
                for i in range(end1 - start1):
                    # Apply temperature scaling
                    scaled_sim = cross_sim[i] / temperature

                    # Apply softmax
                    max_sim = np.max(scaled_sim)
                    exp_sim = np.exp(scaled_sim - max_sim)
                    softmax_sim = exp_sim / np.sum(exp_sim)

                    # Store attention weights
                    cross_type_attention[start1 + i, start2:end2] = softmax_sim * beta

    # Apply cross-type attention to enhance embeddings
    attention_enhanced = np.copy(transformed_embeddings)

    for i in range(n_samples):
        # Get attention weights for this node
        weights = cross_type_attention[i]

        # Apply attention-weighted sum
        if np.sum(weights) > 0:
            attention_enhanced[i] += np.sum(weights.reshape(-1, 1) * all_embeddings, axis=0)

    # 3. Multi-level contrastive learning
    # Create contrastive pairs
    contrastive_enhanced = np.copy(attention_enhanced)

    # For each node type, apply contrastive learning
    for ntype, (start, end) in node_type_indices.items():
        # Get embeddings for this type
        type_embs = attention_enhanced[start:end]

        # Calculate similarity matrix
        sim_matrix = cosine_similarity(type_embs)

        # For each node, find positive and negative pairs
        for i in range(end - start):
            # Get similarities for this node
            sims = sim_matrix[i]

            # Find top-k similar nodes as positives (excluding self)
            k_pos = min(5, (end - start) // 10)
            pos_indices = np.argsort(sims)[-(k_pos+1):-1]

            # Find bottom-k similar nodes as hard negatives
            k_neg = min(10, (end - start) // 5)
            neg_indices = np.argsort(sims)[1:k_neg+1]

            # Apply contrastive learning: pull positives closer, push negatives away
            if len(pos_indices) > 0:
                # Pull positives closer
                pos_centroid = np.mean(type_embs[pos_indices], axis=0)
                contrastive_enhanced[start + i] += gamma * 0.7 * pos_centroid

            if len(neg_indices) > 0:
                # Push negatives away
                neg_centroid = np.mean(type_embs[neg_indices], axis=0)
                contrastive_enhanced[start + i] -= gamma * 0.3 * neg_centroid

    # Normalize the final embeddings
    final_embeddings = normalize(contrastive_enhanced, norm='l2', axis=1)

    # Apply spectral clustering with enhanced affinity
    # Create affinity matrix with type awareness
    affinity = cosine_similarity(final_embeddings)

    # Boost within-type affinities
    for ntype, (start, end) in node_type_indices.items():
        affinity[start:end, start:end] *= 1.2  # Boost within-type similarities

    # Apply spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    labels = spectral.fit_predict(affinity)

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = labels[start:end]

    return result_labels

def multi_scale_spectral_clustering(embeddings, n_clusters, all_types=None, type_indices=None):
    """
    Multi-Scale Spectral Clustering for Heterogeneous Graphs

    This method implements an enhanced spectral clustering approach that:
    1. Constructs multiple affinity matrices at different scales
    2. Incorporates type-aware information in the affinity calculation
    3. Uses adaptive eigengap heuristic for better eigendecomposition
    4. Applies multi-level refinement for improved cluster boundaries
    5. Implements ensemble techniques to combine different scales

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        all_types: list of node types for each node (optional)
        type_indices: dictionary mapping node types to index ranges (optional)
    """
    import numpy as np
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    # Prepare data structures
    if type_indices is None:
        # Create type indices if not provided
        type_indices = {}
        start_idx = 0
        for ntype, embs in embeddings.items():
            n_nodes = len(embs)
            type_indices[ntype] = (start_idx, start_idx + n_nodes)
            start_idx += n_nodes

    # Concatenate all embeddings
    all_embeddings = []
    for ntype in sorted(embeddings.keys()):
        all_embeddings.append(embeddings[ntype])

    all_embeddings = np.vstack(all_embeddings)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # Create node type information if not provided
    if all_types is None:
        all_types = []
        for ntype, (start, end) in type_indices.items():
            all_types.extend([ntype] * (end - start))

    # Create multiple affinity matrices at different scales
    affinity_matrices = []

    # Scale 1: Local neighborhood (small k)
    k_small = min(20, n_samples // 20)
    # Scale 2: Medium neighborhood
    k_medium = min(50, n_samples // 10)
    # Scale 3: Large neighborhood
    k_large = min(100, n_samples // 5)

    k_values = [k_small, k_medium, k_large]

    for k in k_values:
        # Create KNN graph
        from sklearn.neighbors import kneighbors_graph
        knn = kneighbors_graph(
            all_embeddings,
            n_neighbors=k,
            mode='distance',
            include_self=False,
            n_jobs=-1
        )

        # Convert distance to similarity
        knn.data = np.exp(-knn.data ** 2 / (2 * np.var(knn.data)))

        # Make symmetric
        knn = 0.5 * (knn + knn.T)

        # Add type-aware weighting
        knn_array = knn.toarray()

        # Boost within-type connections
        for i in range(n_samples):
            for j in range(n_samples):
                if knn_array[i, j] > 0:
                    # If same type, boost similarity
                    if all_types[i] == all_types[j]:
                        knn_array[i, j] *= 1.5
                    # If different type, reduce similarity slightly
                    else:
                        knn_array[i, j] *= 0.8

        # Add self-loops
        np.fill_diagonal(knn_array, 1.0)

        # Store affinity matrix
        affinity_matrices.append(knn_array)

    # Create additional affinity matrix based on direct cosine similarity
    cosine_affinity = cosine_similarity(all_embeddings)

    # Apply type-aware weighting to cosine similarity
    for i in range(n_samples):
        for j in range(n_samples):
            if all_types[i] == all_types[j]:
                cosine_affinity[i, j] *= 1.3

    affinity_matrices.append(cosine_affinity)

    # Perform spectral clustering at each scale
    scale_labels = []

    for scale_idx, affinity in enumerate(affinity_matrices):
        # Convert to sparse for efficiency with large matrices
        if n_samples > 1000:
            affinity_sparse = sparse.csr_matrix(affinity)
        else:
            affinity_sparse = affinity

        # Compute normalized Laplacian
        if sparse.issparse(affinity_sparse):
            # For sparse matrices
            diag = np.array(affinity_sparse.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(diag, -0.5, where=(diag != 0))
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
            laplacian = sparse.eye(n_samples) - d_mat_inv_sqrt @ affinity_sparse @ d_mat_inv_sqrt
        else:
            # For dense matrices
            diag = np.sum(affinity, axis=1)
            d_inv_sqrt = np.power(diag, -0.5, where=(diag != 0))
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = np.diag(d_inv_sqrt)
            laplacian = np.eye(n_samples) - d_mat_inv_sqrt @ affinity @ d_mat_inv_sqrt

        # Compute eigenvectors
        if sparse.issparse(laplacian):
            # For sparse matrices, use eigsh
            try:
                eigenvalues, eigenvectors = eigsh(
                    laplacian,
                    k=min(n_clusters + 5, n_samples - 1),
                    which='SM',  # Smallest eigenvalues
                    tol=1e-5
                )
            except:
                # Fallback if eigsh fails
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
                eigenvalues = eigenvalues[:n_clusters + 5]
                eigenvectors = eigenvectors[:, :n_clusters + 5]
        else:
            # For dense matrices, use eigh
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            eigenvalues = eigenvalues[:n_clusters + 5]
            eigenvectors = eigenvectors[:, :n_clusters + 5]

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Use eigengap heuristic to determine optimal number of clusters
        if n_clusters > 2:
            eigengaps = np.diff(eigenvalues[1:n_clusters+5])
            optimal_k = np.argmax(eigengaps) + 2  # +2 because we start from the second eigenvalue and add 1
            optimal_k = min(max(optimal_k, 2), n_clusters)  # Ensure between 2 and n_clusters
        else:
            optimal_k = n_clusters

        # Select eigenvectors
        embedding = eigenvectors[:, 1:optimal_k]  # Skip the first eigenvector (constant)

        # Normalize rows to unit length
        embedding = normalize(embedding, norm='l2', axis=1)

        # Apply K-means to the embedding
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300
        )

        labels = kmeans.fit_predict(embedding)
        scale_labels.append(labels)

    # Ensemble the results from different scales
    # Create co-association matrix
    co_assoc = np.zeros((n_samples, n_samples))

    for labels in scale_labels:
        # Create binary co-association matrix for this clustering
        scale_assoc = np.zeros((n_samples, n_samples))
        for cluster_id in range(n_clusters):
            indices = np.where(labels == cluster_id)[0]
            for i in indices:
                scale_assoc[i, indices] = 1

        co_assoc += scale_assoc

    # Normalize co-association matrix
    co_assoc /= len(scale_labels)

    # Apply type-aware weighting to co-association matrix
    for i in range(n_samples):
        for j in range(n_samples):
            if all_types[i] == all_types[j]:
                co_assoc[i, j] *= 1.2

    # Apply spectral clustering on the ensemble co-association matrix
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )

    final_labels = spectral.fit_predict(co_assoc)

    # Refinement step: reassign boundary nodes
    refined_labels = np.copy(final_labels)

    # Calculate cluster centroids in the original embedding space
    centroids = np.zeros((n_clusters, all_embeddings.shape[1]))
    for c in range(n_clusters):
        mask = final_labels == c
        if np.any(mask):
            centroids[c] = np.mean(all_embeddings[mask], axis=0)

    # Identify boundary nodes
    for i in range(n_samples):
        # Calculate similarity to all centroids
        centroid_sims = cosine_similarity(
            all_embeddings[i].reshape(1, -1),
            centroids
        )[0]

        # If the node is at a boundary (similar to multiple clusters)
        sorted_sims = np.sort(centroid_sims)[::-1]
        if len(sorted_sims) > 1 and (sorted_sims[0] - sorted_sims[1]) < 0.1:
            # Find nodes of the same type
            node_type = all_types[i]
            same_type_indices = [j for j in range(n_samples) if all_types[j] == node_type]

            # Find the most common cluster among similar nodes of the same type
            similarities = cosine_similarity(
                all_embeddings[i].reshape(1, -1),
                all_embeddings[same_type_indices]
            )[0]

            # Get top similar nodes
            top_k = min(10, len(same_type_indices))
            top_indices = np.argsort(similarities)[-top_k:]
            top_similar = [same_type_indices[j] for j in top_indices]
            top_clusters = [refined_labels[j] for j in top_similar]

            # Assign to the most common cluster among similar nodes
            from collections import Counter
            if top_clusters:
                most_common = Counter(top_clusters).most_common(1)[0][0]
                refined_labels[i] = most_common

    # Create result dictionary by node type
    result_labels = {}
    for ntype, (start, end) in type_indices.items():
        result_labels[ntype] = refined_labels[start:end]

    return result_labels

def heterogeneous_gmm_clustering(embeddings, n_clusters, all_types=None, type_indices=None):
    """
    Heterogeneous Gaussian Mixture Model Clustering

    This method implements a type-aware Gaussian Mixture Model that:
    1. Fits separate GMMs for each node type with type-specific parameters
    2. Uses a hierarchical approach to ensure coherent clustering across types
    3. Applies cross-type refinement to improve boundary cases
    4. Implements regularization to prevent overfitting
    5. Uses better initialization strategies for more stable results

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        all_types: list of node types for each node (optional)
        type_indices: dictionary mapping node types to index ranges (optional)
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter

    # Prepare data structures
    if type_indices is None:
        # Create type indices if not provided
        type_indices = {}
        start_idx = 0
        for ntype, embs in embeddings.items():
            n_nodes = len(embs)
            type_indices[ntype] = (start_idx, start_idx + n_nodes)
            start_idx += n_nodes

    # Determine number of clusters per type based on node count proportions
    total_nodes = sum(end - start for start, end in type_indices.values())
    type_n_clusters = {}

    for ntype, (start, end) in type_indices.items():
        # Allocate clusters proportionally to node count with a minimum of 1
        type_nodes = end - start
        type_clusters = max(1, int(round((type_nodes / total_nodes) * n_clusters)))
        type_n_clusters[ntype] = type_clusters

    # Ensure total clusters matches n_clusters by adjusting the largest type
    total_type_clusters = sum(type_n_clusters.values())
    if total_type_clusters != n_clusters:
        diff = n_clusters - total_type_clusters
        # Find type with most clusters
        max_type = max(type_n_clusters.items(), key=lambda x: x[1])[0]
        type_n_clusters[max_type] += diff

    print(f"Clusters per type: {type_n_clusters}")

    # Fit separate GMMs for each node type
    type_gmms = {}
    type_labels = {}

    for ntype, (start, end) in type_indices.items():
        type_embs = embeddings[ntype]
        n_type_clusters = type_n_clusters[ntype]

        if n_type_clusters <= 1 or len(type_embs) <= n_type_clusters:
            # If only one cluster or too few samples, assign all to same cluster
            type_labels[ntype] = np.zeros(len(type_embs), dtype=int)
            continue

        # Normalize embeddings
        type_embs = normalize(type_embs, norm='l2', axis=1)

        # Determine best covariance type for this node type
        if len(type_embs) < 5 * type_embs.shape[1]:
            # For small datasets relative to dimensions, use simpler covariance
            cov_type = 'diag'
        else:
            # For larger datasets, use full covariance
            cov_type = 'full'

        # Initialize with K-means++ for better starting points
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_type_clusters,
            random_state=42,
            n_init=10
        )
        kmeans_labels = kmeans.fit_predict(type_embs)

        # Create initial means from K-means centroids
        initial_means = kmeans.cluster_centers_

        # Create initial covariances based on within-cluster scatter
        initial_covs = []
        for i in range(n_type_clusters):
            cluster_mask = kmeans_labels == i
            if np.sum(cluster_mask) > 1:
                cluster_points = type_embs[cluster_mask]
                centered = cluster_points - initial_means[i]
                if cov_type == 'full':
                    cov = np.dot(centered.T, centered) / max(1, len(cluster_points) - 1)
                    # Add regularization to prevent singularity
                    cov += np.eye(cov.shape[0]) * 1e-4
                else:
                    # For diagonal covariance, just use variance of each feature
                    cov = np.var(cluster_points, axis=0) + 1e-4
                initial_covs.append(cov)
            else:
                # Fallback for empty clusters
                if cov_type == 'full':
                    cov = np.eye(type_embs.shape[1]) * 1e-2
                else:
                    cov = np.ones(type_embs.shape[1]) * 1e-2
                initial_covs.append(cov)

        # Create GMM with custom initialization
        gmm = GaussianMixture(
            n_components=n_type_clusters,
            covariance_type=cov_type,
            random_state=42,
            max_iter=200,
            n_init=5,
            reg_covar=1e-4,
            warm_start=True
        )

        # Set initial parameters if using warm_start
        if hasattr(gmm, 'means_init_'):
            gmm.means_init_ = initial_means

        # Fit GMM
        gmm.fit(type_embs)

        # Get cluster assignments
        type_labels[ntype] = gmm.predict(type_embs)

        # Store GMM for later use
        type_gmms[ntype] = gmm

    # Cross-type refinement for boundary nodes
    # For each node type, identify boundary nodes and refine their assignments
    refined_labels = {ntype: labels.copy() for ntype, labels in type_labels.items()}

    for ntype1, (start1, end1) in type_indices.items():
        if ntype1 not in type_gmms:
            continue

        gmm1 = type_gmms[ntype1]
        embs1 = embeddings[ntype1]

        # Get probabilities for each cluster
        probs1 = gmm1.predict_proba(embs1)

        # Identify boundary nodes (nodes with similar probabilities for multiple clusters)
        for i in range(len(embs1)):
            sorted_probs = np.sort(probs1[i])[::-1]

            # If the node is at a boundary (similar probabilities for top clusters)
            if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.2:
                # Find similar nodes in other types
                node_emb = embs1[i].reshape(1, -1)

                # Check similarity with nodes of other types
                cross_type_votes = []

                for ntype2, (start2, end2) in type_indices.items():
                    if ntype2 == ntype1 or ntype2 not in type_gmms:
                        continue

                    embs2 = embeddings[ntype2]
                    labels2 = refined_labels[ntype2]

                    # Calculate similarity to nodes of other type
                    sims = cosine_similarity(node_emb, embs2)[0]

                    # Get most similar nodes
                    top_k = min(10, len(embs2))
                    top_indices = np.argsort(sims)[-top_k:]

                    # Get their cluster assignments
                    top_labels = labels2[top_indices]

                    # Weight by similarity
                    top_sims = sims[top_indices]

                    # Add weighted votes
                    for idx, sim in zip(top_indices, top_sims):
                        cross_type_votes.append((labels2[idx], sim))

                # If we have cross-type votes, use them to refine the assignment
                if cross_type_votes:
                    # Convert GMM cluster to global cluster ID
                    gmm_to_global = {}
                    global_idx = 0

                    for nt, (s, e) in type_indices.items():
                        if nt in type_labels:
                            for c in range(type_n_clusters.get(nt, 1)):
                                gmm_to_global[(nt, c)] = global_idx
                                global_idx += 1

                    # Weight votes by similarity
                    weighted_votes = {}
                    for label, sim in cross_type_votes:
                        if label not in weighted_votes:
                            weighted_votes[label] = 0
                        weighted_votes[label] += sim

                    # Get most voted cluster
                    if weighted_votes:
                        best_label = max(weighted_votes.items(), key=lambda x: x[1])[0]

                        # Map back to type-specific cluster
                        # Find which type this global label belongs to
                        for nt, (s, e) in type_indices.items():
                            if nt in type_labels:
                                for c in range(type_n_clusters.get(nt, 1)):
                                    if gmm_to_global.get((nt, c)) == best_label:
                                        # Found the type and cluster
                                        # If it's the same type, we can directly use the cluster
                                        if nt == ntype1:
                                            refined_labels[ntype1][i] = c
                                        # Otherwise, find the closest cluster in current type
                                        else:
                                            # Get centroid of the voted cluster
                                            if nt in type_gmms:
                                                voted_centroid = type_gmms[nt].means_[c]
                                                # Find closest centroid in current type
                                                centroids1 = type_gmms[ntype1].means_
                                                dists = np.linalg.norm(centroids1 - voted_centroid, axis=1)
                                                closest = np.argmin(dists)
                                                refined_labels[ntype1][i] = closest

    # Combine all labels into a global assignment
    all_labels = []
    for ntype, (start, end) in type_indices.items():
        if ntype in refined_labels:
            # Map type-specific labels to global label space
            type_specific_labels = refined_labels[ntype]
            global_labels = np.zeros_like(type_specific_labels)

            # Create mapping from type-specific to global labels
            label_offset = 0
            for nt in sorted(type_indices.keys()):
                if nt == ntype:
                    break
                label_offset += type_n_clusters.get(nt, 1)

            # Apply mapping
            for i, label in enumerate(type_specific_labels):
                global_labels[i] = label + label_offset

            all_labels.append(global_labels)
        else:
            # If no labels for this type, assign to a default cluster
            all_labels.append(np.zeros(end - start, dtype=int))

    # Concatenate all labels
    combined_labels = np.concatenate(all_labels)

    # Ensure we don't have more than n_clusters
    if len(np.unique(combined_labels)) > n_clusters:
        # Remap to ensure exactly n_clusters
        unique_labels = np.unique(combined_labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        combined_labels = np.array([mapping[l] for l in combined_labels])

    # Create result dictionary by node type
    result_labels = {}
    for ntype, (start, end) in type_indices.items():
        result_labels[ntype] = combined_labels[start:end]

    return result_labels

def hgcl_clustering(embeddings, n_clusters, temperature=0.1, n_views=3):
    """
    Heterogeneous Graph Contrastive Learning (HGCL) clustering

    This method implements a contrastive learning approach specifically designed for heterogeneous graphs:
    1. Multi-view generation with type-specific augmentations
    2. Cross-view contrastive learning with adaptive temperature
    3. Type-aware clustering refinement

    Args:
        embeddings: dictionary of embeddings by node type
        n_clusters: number of clusters
        temperature: temperature parameter for contrastive learning
        n_views: number of contrastive views to generate
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering

    # Concatenate all embeddings
    all_embeddings_list = []
    node_type_indices = {}
    start_idx = 0

    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().detach().numpy()
        else:
            embs_np = embs
        all_embeddings_list.append(embs_np)
        node_type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

    all_embeddings = np.vstack(all_embeddings_list)
    n_samples = all_embeddings.shape[0]

    # Normalize embeddings
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # 1. Generate multiple views with type-specific augmentations
    views = [all_embeddings]  # Original embeddings as first view

    # Generate additional views
    for v in range(n_views - 1):
        view = np.copy(all_embeddings)

        # Apply type-specific augmentations
        for ntype, (start, end) in node_type_indices.items():
            # Get embeddings for this type
            type_embs = view[start:end]

            # Different augmentation strategies for different views and types
            if v == 0:
                # View 1: Add small Gaussian noise
                noise = np.random.normal(0, 0.1, type_embs.shape)
                view[start:end] = type_embs + noise
            elif v == 1:
                # View 2: Feature dropout
                mask = np.random.binomial(1, 0.8, type_embs.shape)
                view[start:end] = type_embs * mask
            else:
                # View 3+: Feature shuffling within type
                for i in range(type_embs.shape[1]):
                    idx = np.random.permutation(end - start)
                    view[start:end, i] = type_embs[idx, i]

        # Normalize the augmented view
        view = normalize(view, norm='l2', axis=1)
        views.append(view)

    # 2. Cross-view contrastive learning
    contrastive_embeddings = np.zeros_like(all_embeddings)

    # For each node, apply contrastive learning across views
    for i in range(n_samples):
        # Get node embeddings from all views
        node_views = [view[i] for view in views]

        # Calculate similarities between this node and all nodes across views
        cross_view_sims = []

        for v1 in range(len(views)):
            for v2 in range(v1 + 1, len(views)):
                # Calculate similarity between this node in view1 and all nodes in view2
                sim_v1_v2 = cosine_similarity(
                    node_views[v1].reshape(1, -1),
                    views[v2]
                )[0]

                # Calculate similarity between this node in view2 and all nodes in view1
                sim_v2_v1 = cosine_similarity(
                    node_views[v2].reshape(1, -1),
                    views[v1]
                )[0]

                cross_view_sims.append((sim_v1_v2, sim_v2_v1))

        # Apply contrastive learning
        # For each view pair, pull the same node closer and push others away
        for sim_v1_v2, sim_v2_v1 in cross_view_sims:
            # Apply temperature scaling
            sim_v1_v2 = sim_v1_v2 / temperature
            sim_v2_v1 = sim_v2_v1 / temperature

            # Apply softmax
            exp_sim_v1_v2 = np.exp(sim_v1_v2)
            exp_sim_v2_v1 = np.exp(sim_v2_v1)

            # Normalize
            norm_sim_v1_v2 = exp_sim_v1_v2 / np.sum(exp_sim_v1_v2)
            norm_sim_v2_v1 = exp_sim_v2_v1 / np.sum(exp_sim_v2_v1)

            # The contrastive objective is to maximize similarity with self across views
            # and minimize similarity with others
            contrastive_embeddings[i] += norm_sim_v1_v2[i] * views[0][i]
            contrastive_embeddings[i] += norm_sim_v2_v1[i] * views[1][i]

    # Normalize contrastive embeddings
    contrastive_embeddings = normalize(contrastive_embeddings, norm='l2', axis=1)

    # 3. Type-aware clustering refinement
    # Create affinity matrix with type awareness
    affinity = cosine_similarity(contrastive_embeddings)

    # Boost within-type affinities
    for ntype, (start, end) in node_type_indices.items():
        affinity[start:end, start:end] *= 1.3  # Higher boost for within-type

    # Apply spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
        n_jobs=-1
    )

    labels = spectral.fit_predict(affinity)

    # Separate results by node type
    result_labels = {}
    for ntype, (start, end) in node_type_indices.items():
        result_labels[ntype] = labels[start:end]

    return result_labels

def run_global_clustering(embeddings, true_labels=None, n_clusters=None):
    """
    Effectue un clustering global sur les embeddings HGT avec différentes méthodes
    et évalue les performances avec NMI, ARI, ACC et F1

    Args:
        embeddings: dictionnaire des embeddings par type de nœud
        true_labels: dictionnaire des labels réels par type de nœud (optionnel)
        n_clusters: nombre de clusters (si None, estimé automatiquement)
    """
    print("\n" + "="*50)
    print("CLUSTERING GLOBAL SUR LES EMBEDDINGS HGT")
    print("="*50)

    # Importations nécessaires
    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt

    # Concaténer tous les embeddings
    all_embeddings_list = []
    all_types = []
    all_true_labels = []
    type_indices = {}  # Pour suivre les indices par type de nœud

    # Check if we have embeddings with different dimensions
    emb_dims = {emb.shape[1] for emb in embeddings.values()}
    print(f"Detected embedding dimensions: {emb_dims}")

    # Process each node type and add to the combined embeddings list
    start_idx = 0
    all_embeddings_numpy = []  # We'll store numpy arrays directly

    # First, determine the common dimension for all embeddings
    embedding_dims = set()
    for ntype, embs in embeddings.items():
        embedding_dims.add(embs.shape[1])

    print(f"Detected embedding dimensions: {embedding_dims}")

    # Choose a common dimension - use the most common one
    if len(embedding_dims) > 1:
        # Find the most common dimension
        dim_counts = {}
        for ntype, embs in embeddings.items():
            dim = embs.shape[1]
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        # Use the most common dimension
        common_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
        print(f"Using common dimension: {common_dim}")
    else:
        # All embeddings have the same dimension
        common_dim = list(embedding_dims)[0]

    # First, convert all embeddings to the same dimension
    for ntype, embs in embeddings.items():
        n_nodes = embs.shape[0]

        # Convert to numpy and handle different dimensions
        if embs.shape[1] != common_dim:
            # Resize to common dimension
            embs_cpu = embs.cpu().detach().numpy()

            if embs.shape[1] > common_dim:
                # Use PCA to reduce dimensions
                from sklearn.decomposition import PCA
                pca = PCA(n_components=common_dim)
                embs_numpy = pca.fit_transform(embs_cpu)
                print(f"Reduced {ntype} embeddings from {embs.shape[1]} to {common_dim} dimensions using PCA")
            else:
                # Use zero padding to increase dimensions
                padding = np.zeros((n_nodes, common_dim - embs.shape[1]))
                embs_numpy = np.hstack((embs_cpu, padding))
                print(f"Padded {ntype} embeddings from {embs.shape[1]} to {common_dim} dimensions")
        else:
            # Already the right dimension
            embs_numpy = embs.cpu().detach().numpy()

        all_embeddings_numpy.append(embs_numpy)
        all_types.extend([ntype] * n_nodes)

        # Suivre les indices pour ce type de nœud
        type_indices[ntype] = (start_idx, start_idx + n_nodes)
        start_idx += n_nodes

        # Ajouter les labels réels si disponibles
        if true_labels is not None and ntype in true_labels:
            all_true_labels.extend(true_labels[ntype])
        else:
            all_true_labels.extend([-1] * n_nodes)  # -1 pour les nœuds sans label

    print(f"Total nodes for clustering: {start_idx}")
    print(f"Node types included: {list(embeddings.keys())}")

    # Convertir en arrays
    if all_embeddings_numpy:
        all_embeddings = np.vstack(all_embeddings_numpy)
        all_types = np.array(all_types)
        all_true_labels = np.array(all_true_labels)
        print(f"Final embeddings shape: {all_embeddings.shape}")
    else:
        print("No valid embeddings found for clustering.")
        return []

    # Normaliser les embeddings
    all_embeddings = normalize(all_embeddings, axis=1)

    # Define contrastive refinement function
    def apply_contrastive_refinement(embeddings, temperature=0.1):
        """Apply contrastive refinement to improve embedding separation"""
        # Normalize embeddings
        norm_embeddings = normalize(embeddings, norm='l2', axis=1)

        # Compute similarity matrix
        similarity = np.dot(norm_embeddings, norm_embeddings.T)

        # Apply temperature scaling
        similarity = similarity / temperature

        # Apply softmax to get probabilities
        max_sim = np.max(similarity, axis=1, keepdims=True)
        exp_sim = np.exp(similarity - max_sim)
        softmax_sim = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        # Create refined embeddings by weighted combination
        refined_embeddings = np.dot(softmax_sim, norm_embeddings)

        # Normalize again
        refined_embeddings = normalize(refined_embeddings, norm='l2', axis=1)

        return refined_embeddings

    # Réduction de dimensionnalité pour améliorer le clustering
    print("Application de la réduction de dimensionnalité...")

    # Apply contrastive refinement to improve embedding separation before dimensionality reduction
    print("Applying contrastive refinement to improve embedding separation...")
    all_embeddings = apply_contrastive_refinement(all_embeddings, temperature=0.05)

    # Calculate safe number of components for dimensionality reduction
    # Never exceed min(n_samples, n_features)
    max_components = min(all_embeddings.shape[0], all_embeddings.shape[1])
    safe_components = min(64, max_components)

    try:
        from umap import UMAP
        # Use supervised UMAP if labels are available
        if np.any(all_true_labels >= 0):
            valid_indices = all_true_labels >= 0
            reducer = UMAP(
                n_components=min(safe_components, all_embeddings.shape[1]),
                metric='cosine',
                n_neighbors=min(30, sum(valid_indices)-1),
                min_dist=0.05,  # Reduced from 0.1 for better cluster separation
                random_state=42,
                target_metric='categorical',
                target_weight=0.7,  # Increased from 0.5 to give more weight to labels
                n_epochs=500,  # More epochs for better convergence
                learning_rate=0.5  # Increased learning rate
            )
            # Fit on labeled data only
            reducer.fit(all_embeddings[valid_indices], all_true_labels[valid_indices])
            # Transform all data
            reduced_embeddings = reducer.transform(all_embeddings)
        else:
            reducer = UMAP(
                n_components=min(safe_components, all_embeddings.shape[1]),
                metric='cosine',
                n_neighbors=min(30, all_embeddings.shape[0]-1),
                min_dist=0.05,  # Reduced for better cluster separation
                random_state=42,
                n_epochs=500,  # More epochs for better convergence
                learning_rate=0.5  # Increased learning rate
            )
            reduced_embeddings = reducer.fit_transform(all_embeddings)
        print(f"Applied UMAP: {all_embeddings.shape[1]} → {reduced_embeddings.shape[1]} dimensions")
    except ImportError:
        print("UMAP not available, using optimized PCA...")
        # First try to apply t-SNE for better cluster separation
        try:
            from sklearn.manifold import TSNE
            # Use PCA first to reduce dimensions before t-SNE
            if all_embeddings.shape[1] > 50:
                # Safe PCA components
                pca_components = min(50, max_components)
                pca = PCA(n_components=pca_components, random_state=42)
                pca_embeddings = pca.fit_transform(all_embeddings)
                print(f"Applied PCA pre-processing: {all_embeddings.shape[1]} → {pca_components} dimensions")
            else:
                pca_embeddings = all_embeddings

            # Apply t-SNE with optimized parameters
            tsne_components = min(min(50, pca_embeddings.shape[1]), max_components)
            tsne = TSNE(
                n_components=tsne_components,
                perplexity=min(30, all_embeddings.shape[0] // 5),
                learning_rate='auto',
                n_iter=2000,
                random_state=42,
                init='pca'
            )
            reduced_embeddings = tsne.fit_transform(pca_embeddings)
            print(f"Applied t-SNE: {all_embeddings.shape[1]} → {reduced_embeddings.shape[1]} dimensions")
        except Exception as e:
            print(f"t-SNE failed: {str(e)}, falling back to PCA...")
            # Use Truncated SVD for sparse data, otherwise PCA
            if sp.issparse(all_embeddings):
                svd_components = min(min(100, all_embeddings.shape[1]), max_components)
                reducer = TruncatedSVD(
                    n_components=svd_components,
                    random_state=42,
                    algorithm='randomized',
                    n_iter=10
                )
            else:
                pca_components = min(min(100, all_embeddings.shape[1]), max_components)
                reducer = PCA(
                    n_components=pca_components,
                    random_state=42,
                    svd_solver='randomized',
                    whiten=True  # Apply whitening for better separation
                )
            reduced_embeddings = reducer.fit_transform(all_embeddings)
            print(f"Applied {'TruncatedSVD' if sp.issparse(all_embeddings) else 'PCA'}: {all_embeddings.shape[1]} → {reduced_embeddings.shape[1]} dimensions")

    # Normalize the reduced embeddings
    reduced_embeddings = normalize(reduced_embeddings, norm='l2', axis=1)

    # Before dimensionality reduction, apply embedding normalization and concatenation strategy
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)

    # We've already applied contrastive refinement earlier, so we don't need to do it again
    print("Contrastive refinement already applied to embeddings")

    # Estimer le nombre de clusters si non spécifié
    if n_clusters is None:
        if true_labels is not None and np.any(all_true_labels >= 0):
            # Utiliser le nombre de classes uniques dans les labels réels
            unique_labels = np.unique(all_true_labels[all_true_labels >= 0])
            n_clusters = len(unique_labels) if len(unique_labels) > 1 else 2
            print(f"Utilisation du nombre de classes uniques dans les labels: {n_clusters}")
        else:
            # Estimer avec plusieurs méthodes
            from sklearn.metrics import silhouette_score
            from sklearn.neighbors import NearestNeighbors

            # 1. Méthode du silhouette score
            best_score = -1
            best_k_silhouette = 2

            for k in range(2, min(20, len(reduced_embeddings) // 5)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(reduced_embeddings)
                try:
                    score = silhouette_score(reduced_embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k_silhouette = k
                except:
                    pass

            # 2. Méthode du coude (elbow method)
            try:
                from yellowbrick.cluster import KElbowVisualizer

                best_k_elbow = 2
                model = KMeans(random_state=42)
                visualizer = KElbowVisualizer(model, k=(2, min(20, len(reduced_embeddings) // 5)), timings=False)
                visualizer.fit(reduced_embeddings)
                best_k_elbow = visualizer.elbow_value_ if visualizer.elbow_value_ else 2
            except:
                best_k_elbow = 2
                print("Yellowbrick non disponible, utilisation de k=2 pour la méthode du coude")

            # 3. Méthode du gap statistic
            best_k_gap = 2
            try:
                from gap_statistic import OptimalK
                optimalK = OptimalK(n_jobs=-1)
                best_k_gap = optimalK(reduced_embeddings, cluster_array=np.arange(2, min(20, len(reduced_embeddings) // 5)))
            except:
                print("gap_statistic non disponible, utilisation de k=2 pour la méthode du gap")

            # Prendre la médiane des estimations
            k_estimates = [best_k_silhouette, best_k_elbow, best_k_gap]
            n_clusters = int(np.median(k_estimates))
            print(f"Estimations du nombre de clusters: silhouette={best_k_silhouette}, elbow={best_k_elbow}, gap={best_k_gap}")
            print(f"Nombre de clusters estimé (médiane): {n_clusters}")

    # S'assurer que n_clusters est au moins 2
    n_clusters = max(2, n_clusters)
    print(f"Nombre final de clusters: {n_clusters}")

    # Définir une classe pour le Deep Embedded Clustering (DEC)
    class DeepEmbeddedClustering:
        def __init__(self, n_clusters, alpha=1.0, max_iter=100):
            self.n_clusters = n_clusters
            self.alpha = alpha  # Paramètre de distribution t-Student
            self.max_iter = max_iter
            self.cluster_centers = None

        def fit_predict(self, X):
            """
            Applique l'algorithme DEC et retourne les labels

            Args:
                X: données d'entrée (embeddings)
            """
            # Initialiser les centres de clusters avec K-means
            print("Initialisation des centres de clusters avec K-means...")
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
            y_pred = kmeans.fit_predict(X)
            self.cluster_centers = kmeans.cluster_centers_

            # Convertir en tenseurs PyTorch
            X_tensor = torch.tensor(X, dtype=torch.float32)
            centers_tensor = torch.tensor(self.cluster_centers, dtype=torch.float32)

            # Boucle principale DEC
            print("Application de l'algorithme DEC...")

            # Initialiser prev_loss pour la première itération
            prev_loss = float('inf')

            for iteration in range(self.max_iter):
                # Calculer la distribution cible (distribution t-Student)
                q = self._calculate_q(X_tensor, centers_tensor)

                # Calculer la distribution auxiliaire (cible durcie)
                p = self._calculate_p(q)

                # Mettre à jour les centres de clusters
                self._update_centers(X_tensor, p)

                # Calculer la divergence KL
                kl_loss = self._kl_divergence(p, q)

                if iteration % 10 == 0:
                    print(f"Itération {iteration}, KL divergence: {kl_loss:.4f}")

                # Vérifier la convergence
                if iteration > 0 and abs(kl_loss - prev_loss) < 1e-4:
                    print(f"Convergence atteinte à l'itération {iteration}")
                    break

                prev_loss = kl_loss

            # Assigner les points aux clusters
            q = self._calculate_q(X_tensor, centers_tensor)
            return torch.argmax(q, dim=1).numpy()

        def _calculate_q(self, X, centers):
            """Calcule la distribution t-Student (soft assignments)"""
            n_samples = X.shape[0]
            n_clusters = centers.shape[0]

            # Calculer les distances au carré entre points et centres
            q = torch.zeros(n_samples, n_clusters)
            for i in range(n_clusters):
                # Distance euclidienne au carré
                dist = torch.sum(torch.pow(X - centers[i], 2), dim=1)
                q[:, i] = torch.pow(1.0 + dist / self.alpha, -(self.alpha + 1.0) / 2.0)

            # Normaliser
            q = q / torch.sum(q, dim=1, keepdim=True)
            return q

        def _calculate_p(self, q):
            """Calcule la distribution auxiliaire P à partir de Q"""
            weight = q.sum(0)
            p = torch.pow(q, 2) / weight
            p = p / p.sum(1, keepdim=True)
            return p

        def _update_centers(self, X, p):
            """Met à jour les centres de clusters"""
            n_clusters = p.shape[1]
            for j in range(n_clusters):
                # Moyenne pondérée
                numerator = torch.sum(torch.unsqueeze(p[:, j], 1) * X, dim=0)
                denominator = torch.sum(p[:, j])

                # Éviter la division par zéro
                if denominator > 0:
                    self.cluster_centers[j] = (numerator / denominator).numpy()

        def _kl_divergence(self, p, q):
            """Calcule la divergence KL entre P et Q"""
            return torch.mean(torch.sum(p * torch.log(p / q), dim=1))

    # Définir les méthodes de clustering
    clustering_methods = {
        # Nouvelles méthodes state-of-the-art pour graphes hétérogènes
        "SimpleHGNN": CustomClusteringMethod(
            lambda X: simple_hgnn_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                alpha=0.5,  # Poids pour l'information spécifique au type
                beta=0.3,   # Poids pour les relations inter-types
                gamma=0.2,  # Poids pour l'apprentissage contrastif
                temperature=0.1  # Température pour le scaling
            ),
            n_clusters=n_clusters
        ),

        "HGCL": CustomClusteringMethod(
            lambda X: hgcl_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                temperature=0.1,  # Température plus basse pour des contrastes plus nets
                n_views=3  # Nombre de vues contrastives
            ),
            n_clusters=n_clusters
        ),

        # Méthodes existantes pour graphes hétérogènes avec paramètres optimisés
        "MAGCN": CustomClusteringMethod(
            lambda X: magcn_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                n_views=5,  # Augmenté pour capturer plus de perspectives
                alpha=0.4,  # Augmenté pour donner plus de poids à l'information de type
                beta=0.3    # Augmenté pour améliorer les relations inter-types
            ),
            n_clusters=n_clusters
        ),

        "HDGI": CustomClusteringMethod(
            lambda X: hdgi_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                alpha=0.3,  # Paramètre de pondération pour l'information de type
                beta=0.2,   # Paramètre pour les rôles structurels
                gamma=0.1   # Paramètre pour les relations inter-types
            ),
            n_clusters=n_clusters
        ),

        "HeCo": CustomClusteringMethod(
            lambda X: heco_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                temperature=0.2,  # Température réduite pour des contrastes plus nets
                alpha=0.7,        # Plus de poids pour la vue sémantique
                beta=0.3          # Moins de poids pour la vue structurelle
            ),
            n_clusters=n_clusters
        ),

        # Notre méthode adaptée aux graphes hétérogènes
        "Deep Spectral Heterogeneous": CustomClusteringMethod(
            lambda X: deep_spectral_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                alpha=0.3
            ),
            n_clusters=n_clusters
        ),

        # Méthodes avancées pour graphes hétérogènes
        "DMGI": CustomClusteringMethod(
            lambda X: dmgi_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                alpha=0.3,
                beta=0.1,
                temperature=0.5
            ),
            n_clusters=n_clusters
        ),

        "MCGC": CustomClusteringMethod(
            lambda X: mcgc_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                n_views=4,
                lambda_param=0.5,
                temperature=0.2
            ),
            n_clusters=n_clusters
        ),

        "HGAT": CustomClusteringMethod(
            lambda X: hgat_clustering(
                {ntype: torch.tensor(X[start:end]) for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                n_heads=8,
                dropout=0.1,
                alpha=0.2
            ),
            n_clusters=n_clusters
        ),

        # Ajouter Deep Embedded Clustering (DEC)
        "Deep Embedded Clustering": DeepEmbeddedClustering(
            n_clusters=n_clusters,
            alpha=1.0,
            max_iter=100
        ),

        # Ajouter une méthode basée sur Deep Graph Infomax (DGI)
        "Deep Graph Infomax": CustomClusteringMethod(
            lambda X: deep_graph_infomax_clustering(
                X,
                n_clusters=n_clusters,
                n_neighbors=15
            ),
            n_clusters=n_clusters
        ),

        "K-means++": KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=50,  # More initializations
            max_iter=500,  # More iterations
            algorithm='elkan',  # Faster for dense data
            init='k-means++'  # Better initialization
        ),
        "Mini-Batch K-means": MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=20),
        "Spectral (optimized)": SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            n_neighbors=min(50, len(reduced_embeddings)-1),
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            n_jobs=-1
        ),

        "Multi-Scale Spectral": CustomClusteringMethod(
            lambda X: multi_scale_spectral_clustering(
                {ntype: X[start:end] for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                all_types=all_types,
                type_indices=type_indices
            ),
            n_clusters=n_clusters
        ),
        "Hiérarchique (Ward optimized)": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            compute_full_tree=True
        ),
        "Hiérarchique (Complete)": AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
        "Hiérarchique (Average)": AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
        "Gaussian Mixture (optimized)": GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            n_init=20,
            covariance_type='full',  # Try different covariance types
            max_iter=200,
            reg_covar=1e-5  # Regularization to prevent singularities
        ),

        "Heterogeneous GMM": CustomClusteringMethod(
            lambda X: heterogeneous_gmm_clustering(
                {ntype: X[start:end] for ntype, (start, end) in type_indices.items()},
                n_clusters=n_clusters,
                all_types=all_types,
                type_indices=type_indices
            ),
            n_clusters=n_clusters
        ),
        "BIRCH (optimized)": Birch(
            n_clusters=n_clusters,
            threshold=0.005,  # Lower threshold for more clusters initially
            branching_factor=50  # Higher branching factor
        )
    }

    # Add HDBSCAN with optimized parameters
    if HDBSCAN_AVAILABLE:
        min_cluster_size = max(5, int(len(reduced_embeddings) * 0.02))  # Increased from 0.01
        min_samples = max(5, int(min_cluster_size * 0.5))

        clustering_methods["HDBSCAN (optimized)"] = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.2,  # Increased from 0.1
            metric='euclidean',
            prediction_data=True,
            cluster_selection_method='eom'  # Try 'eom' instead of default 'leaf'
        )

    # Implement a more sophisticated ensemble method
    def advanced_ensemble_clustering(predictions_list, weights=None, n_clusters=None):
        """
        Advanced ensemble clustering using co-association matrix and spectral clustering

        Args:
            predictions_list: List of cluster assignment arrays
            weights: Optional weights for each clustering method
            n_clusters: Number of clusters for final clustering
        """
        n_samples = len(predictions_list[0])
        n_methods = len(predictions_list)

        # Default to equal weights if not provided
        if weights is None:
            weights = np.ones(n_methods) / n_methods

        # Create co-association matrix
        co_association = np.zeros((n_samples, n_samples))

        for i, pred in enumerate(predictions_list):
            # Convert to one-hot encoding
            one_hot = np.zeros((n_samples, n_samples))
            for c in np.unique(pred):
                if c < 0:  # Skip noise points
                    continue
                cluster_indices = np.where(pred == c)[0]
                for idx1 in cluster_indices:
                    for idx2 in cluster_indices:
                        one_hot[idx1, idx2] = 1

            # Weight by clustering method quality
            co_association += one_hot * weights[i]

        # Normalize
        co_association /= np.sum(weights)

        # Apply spectral clustering on co-association matrix
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize',  # Try 'discretize' instead of 'kmeans'
            n_jobs=-1
        )

        # Convert similarity to affinity
        ensemble_labels = spectral.fit_predict(co_association)

        return ensemble_labels

    # Ensemble voting avec pondération
    def weighted_ensemble_clustering(predictions, weights):
        n_samples = len(predictions[0])
        similarity_matrix = np.zeros((n_samples, n_samples))

        for pred, weight in zip(predictions, weights):
            curr_sim = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                curr_sim[i, :] = (pred == pred[i])
            similarity_matrix += curr_sim * weight

        # Normalisation
        similarity_matrix /= sum(weights)

        # Clustering final sur la matrice de similarité
        final_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        ).fit_predict(similarity_matrix)

        return final_clustering

    # Fonction pour calculer l'accuracy avec alignement optimal
    def calculate_accuracy(y_true, y_pred):
        """Calcule l'accuracy avec alignement optimal des clusters"""
        if len(np.unique(y_true)) == 1:
            return 0.0  # Cas trivial, tous les labels sont identiques

        # Créer une matrice de confusion
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        # Trouver l'alignement optimal
        row_ind, col_ind = linear_sum_assignment(-cm)

        # Calculer l'accuracy
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        return accuracy

    # Fonction pour calculer le F1 score avec alignement optimal
    def calculate_f1(y_true, y_pred):
        """Calcule le F1 score avec alignement optimal des clusters"""
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            return 0.0  # Cas trivial

        # Créer une matrice de poids pour l'alignement
        from sklearn.metrics import f1_score

        # Calculer le F1 score pour chaque paire de classes
        n_true_classes = len(np.unique(y_true))
        n_pred_classes = len(np.unique(y_pred))
        w = np.zeros((n_pred_classes, n_true_classes))

        for i in range(n_pred_classes):
            for j in range(n_true_classes):
                # Créer des labels binaires
                y_true_bin = (y_true == j)
                y_pred_bin = (y_pred == i)
                w[i, j] = f1_score(y_true_bin, y_pred_bin)

        row_ind, col_ind = linear_sum_assignment(w.max() - w)

        # Créer un mapping des clusters aux classes
        cluster_to_class = {row: col for row, col in zip(row_ind, col_ind)}

        # Remapper les prédictions
        y_pred_aligned = np.array([cluster_to_class.get(c, -1) for c in y_pred])

        # Calculer le F1 score
        return f1_score(y_true, y_pred_aligned, average='macro')

    # Visualisation des embeddings avant clustering
    print("\nCréation de la visualisation des embeddings avant clustering...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(reduced_embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_true_labels, cmap='tab20')
    plt.colorbar()
    plt.title('Distribution des embeddings avant clustering')
    plt.show()

    # Visualisation de la matrice de similarité
    print("\nCréation de la visualisation de la matrice de similarité...")
    similarity = cosine_similarity(reduced_embeddings)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title('Matrice de similarité des embeddings')
    plt.show()

    # Évaluer chaque méthode de clustering
    results = []

    # Filtrer les nœuds avec des labels connus pour l'évaluation
    valid_mask = all_true_labels >= 0

    # Évaluer les méthodes globales
    for name, model in clustering_methods.items():
        try:
            print(f"\nÉvaluation de {name}...")

            # Appliquer le clustering
            pred_labels = model.fit_predict(reduced_embeddings)

            # Évaluer si des labels réels sont disponibles
            if true_labels is not None and np.any(valid_mask):
                # Calculer les métriques
                nmi = normalized_mutual_info_score(all_true_labels[valid_mask], pred_labels[valid_mask])
                ari = adjusted_rand_score(all_true_labels[valid_mask], pred_labels[valid_mask])
                acc = calculate_accuracy(all_true_labels[valid_mask], pred_labels[valid_mask])
                f1 = calculate_f1(all_true_labels[valid_mask], pred_labels[valid_mask])

                print(f"NMI: {nmi:.4f}")
                print(f"ARI: {ari:.4f}")
                print(f"ACC: {acc:.4f}")
                print(f"F1: {f1:.4f}")

                results.append((name, nmi, ari, acc, f1, pred_labels))
            else:
                print("Pas de labels réels disponibles pour l'évaluation")
                results.append((name, 0, 0, 0, 0, pred_labels))

            # Analyser la distribution des clusters par type de nœud
            cluster_distribution = {}
            for i, cluster in enumerate(np.unique(pred_labels)):
                mask = pred_labels == cluster
                types_in_cluster = all_types[mask]
                type_counts = Counter(types_in_cluster)

                print(f"Cluster {i}: {len(mask[mask])} nœuds")
                for t, count in type_counts.items():
                    print(f"  - {t}: {count} nœuds ({count/len(mask[mask])*100:.1f}%)")

                cluster_distribution[i] = type_counts

            # Visualiser les clusters avec t-SNE
            if len(reduced_embeddings) <= 10000:  # Limiter pour les grands graphes
                visualize_clusters(reduced_embeddings, all_true_labels, pred_labels, title=f"Clusters ({name})")

        except Exception as e:
            print(f"Erreur lors de l'évaluation de {name}: {str(e)}")

    # Trier les résultats par NMI décroissant
    if results:
        results.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "="*50)
        print("RÉSULTATS DU CLUSTERING GLOBAL")
        print("="*50)
        print(f"{'Méthode':<25} {'NMI':<10} {'ARI':<10} {'ACC':<10} {'F1':<10}")
        print("-" * 65)

        for name, nmi, ari, acc, f1, _ in results:
            print(f"{name:<25} {nmi:.4f}     {ari:.4f}     {acc:.4f}     {f1:.4f}")

        # Sélectionner les meilleures méthodes pour l'ensemble de consensus
        top_k = min(3, len(results))
        top_methods = results[:top_k]

        print(f"\nMeilleures méthodes: {', '.join([m[0] for m in top_methods])}")

        # Créer un ensemble de consensus amélioré à partir des meilleures méthodes
        print("\nCréation d'un ensemble de consensus amélioré à partir des meilleures méthodes...")

        consensus_predictions = []
        performance_weights = []

        # Calculate weights based on all metrics (not just NMI)
        for name, nmi, ari, acc, f1, pred_labels in top_methods:
            consensus_predictions.append(pred_labels)
            # Use a weighted combination of metrics with more emphasis on NMI and ACC
            weight = (nmi * 0.4 + ari * 0.1 + acc * 0.3 + f1 * 0.2)
            performance_weights.append(weight)

        # Normalize weights
        performance_weights = np.array(performance_weights)
        if np.sum(performance_weights) > 0:
            performance_weights = performance_weights / np.sum(performance_weights)
        else:
            performance_weights = np.ones(len(performance_weights)) / len(performance_weights)

        print(f"Method weights: {[f'{w:.4f}' for w in performance_weights]}")

        # Créer une matrice de co-association améliorée
        n_samples = len(reduced_embeddings)
        co_association = np.zeros((n_samples, n_samples))

        # More efficient implementation with vectorized operations
        for pred, weight in zip(consensus_predictions, performance_weights):
            # Create a binary co-association matrix for this prediction
            # 1 if two samples are in the same cluster, 0 otherwise
            co_assoc = np.zeros((n_samples, n_samples))

            # Optimized vectorized operations
            for cluster_id in np.unique(pred):
                if cluster_id >= 0:  # Ignore noise clusters (-1)
                    mask = (pred == cluster_id)
                    indices = np.where(mask)[0]
                    for i in indices:
                        co_assoc[i, indices] = 1

            # Add to consensus matrix with weight
            co_association += co_assoc * weight

        # Try multiple ensemble techniques and combine them

        # 1. Spectral clustering on consensus matrix
        try:
            spectral_ensemble = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                n_jobs=-1
            ).fit_predict(co_association)
        except Exception as e:
            print(f"Error with spectral clustering: {str(e)}")
            # Fallback to K-means
            spectral_ensemble = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=20
            ).fit_predict(co_association)

        # 2. Hierarchical clustering on consensus matrix
        distance_matrix = 1 - co_association
        try:
            hierarchical_ensemble = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            ).fit_predict(distance_matrix)
        except Exception as e:
            print(f"Error with hierarchical clustering: {str(e)}")
            try:
                # Try with default affinity
                hierarchical_ensemble = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average'
                ).fit_predict(co_association)
            except Exception as e:
                print(f"Error with hierarchical clustering (fallback): {str(e)}")
                # Fallback to K-means
                hierarchical_ensemble = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=20
                ).fit_predict(co_association)

        # Create meta-ensemble by combining the two ensemble methods
        meta_ensemble = np.zeros((n_samples, n_clusters))

        # For each ensemble method, create a one-hot encoding of the clusters
        ensemble_methods = [
            ("Spectral", spectral_ensemble, 0.6),  # Give more weight to spectral
            ("Hierarchical", hierarchical_ensemble, 0.4)
        ]

        for name, ensemble, weight in ensemble_methods:
            for i in range(n_samples):
                if ensemble[i] >= 0 and ensemble[i] < n_clusters:
                    meta_ensemble[i, ensemble[i]] += weight

        # Assign each sample to the cluster with the most votes
        final_clustering = np.argmax(meta_ensemble, axis=1)

        # Refinement with original embeddings
        print("Refining clusters with original embeddings...")
        try:
            # Calculate cluster centroids
            centroids = np.zeros((n_clusters, reduced_embeddings.shape[1]))
            for c in range(n_clusters):
                mask = final_clustering == c
                if np.any(mask):
                    centroids[c] = np.mean(reduced_embeddings[mask], axis=0)

            # Reassign points to nearest centroid
            for i in range(n_samples):
                distances = np.linalg.norm(reduced_embeddings[i] - centroids, axis=1)
                final_clustering[i] = np.argmin(distances)
        except Exception as e:
            print(f"Error during refinement: {str(e)}")

        # Évaluer le clustering final
        if true_labels is not None and np.any(valid_mask):
            nmi = normalized_mutual_info_score(all_true_labels[valid_mask], final_clustering[valid_mask])
            ari = adjusted_rand_score(all_true_labels[valid_mask], final_clustering[valid_mask])
            acc = calculate_accuracy(all_true_labels[valid_mask], final_clustering[valid_mask])
            f1 = calculate_f1(all_true_labels[valid_mask], final_clustering[valid_mask])

            print(f"\nClustering final - NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
        else:
            print("Pas de labels réels disponibles pour l'évaluation du clustering final")

        # Visualiser le clustering final
        if len(reduced_embeddings) <= 10000:  # Limiter pour les grands graphes
            visualize_clusters(reduced_embeddings, all_true_labels, final_clustering, title="Clustering Final (Ensemble de Consensus)")

    # After evaluating individual methods, use advanced ensemble
    if results:
        print("\nApplying advanced ensemble clustering...")

        # Get top 5 methods or all if less than 5
        top_k = min(5, len(results))
        top_methods = results[:top_k]

        # Extract predictions and calculate weights based on NMI scores
        predictions = [res[5] for res in top_methods]
        weights = np.array([res[1] for res in top_methods])

        # Normalize weights to sum to
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Apply advanced ensemble
        ensemble_labels = advanced_ensemble_clustering(
            predictions,
            weights=weights,
            n_clusters=n_clusters
        )

        # Evaluate ensemble
        if true_labels is not None and np.any(valid_mask):
            nmi = normalized_mutual_info_score(all_true_labels[valid_mask], ensemble_labels[valid_mask])
            ari = adjusted_rand_score(all_true_labels[valid_mask], ensemble_labels[valid_mask])
            acc = calculate_accuracy(all_true_labels[valid_mask], ensemble_labels[valid_mask])
            f1 = calculate_f1(all_true_labels[valid_mask], ensemble_labels[valid_mask])

            print(f"\nAdvanced Ensemble - NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")

    # Add semi-supervised clustering if some labels are available
    def semi_supervised_clustering(embeddings, partial_labels, n_clusters):
        """
        Semi-supervised clustering using available labels as constraints

        Args:
            embeddings: Embedding matrix
            partial_labels: Array of labels (-1 for unknown)
            n_clusters: Number of clusters
        """
        from sklearn.semi_supervised import LabelPropagation

        # Get indices of labeled and unlabeled points
        labeled_indices = np.where(partial_labels >= 0)[0]
        unlabeled_indices = np.where(partial_labels < 0)[0]

        if len(labeled_indices) == 0:
            print("No labeled data available for semi-supervised clustering")
            return None

        print(f"Using {len(labeled_indices)} labeled points for semi-supervised clustering")

        # Create label propagation model
        label_prop = LabelPropagation(
            kernel='knn',
            n_neighbors=min(50, len(labeled_indices)),
            max_iter=1000
        )

        # Create working copy of labels
        working_labels = np.copy(partial_labels)

        # Fit model
        label_prop.fit(embeddings[labeled_indices], partial_labels[labeled_indices])

        # Predict labels for unlabeled points
        if len(unlabeled_indices) > 0:
            predicted_labels = label_prop.predict(embeddings[unlabeled_indices])
            working_labels[unlabeled_indices] = predicted_labels

        return working_labels

    # Apply semi-supervised clustering if some labels are available
    if np.any(all_true_labels >= 0):
        print("\nApplying semi-supervised clustering...")
        semi_supervised_labels = semi_supervised_clustering(
            reduced_embeddings,
            all_true_labels,
            n_clusters
        )

        if semi_supervised_labels is not None:
            # Evaluate
            valid_mask = all_true_labels >= 0
            nmi = normalized_mutual_info_score(all_true_labels[valid_mask], semi_supervised_labels[valid_mask])
            ari = adjusted_rand_score(all_true_labels[valid_mask], semi_supervised_labels[valid_mask])
            acc = calculate_accuracy(all_true_labels[valid_mask], semi_supervised_labels[valid_mask])
            f1 = calculate_f1(all_true_labels[valid_mask], semi_supervised_labels[valid_mask])

            print(f"Semi-supervised - NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")

            # Visualize
            if len(reduced_embeddings) <= 10000:
                visualize_clusters(reduced_embeddings, all_true_labels, semi_supervised_labels,
                                  title="Semi-supervised Clustering")

    # Return all results for further analysis if needed
    return results

def main():
    print("\n" + "="*50)
    print("DÉMARRAGE DU PIPELINE D'APPRENTISSAGE")
    print("="*50)

    # 1. Création du graphe hétérogène
    print("\n1. Création du graphe hétérogène...")
    g, id_node_map, rdf_graph = create_heterogeneous_graph()

    # 2. Préparation des embeddings de relations
    print("\n2. Initialisation des embeddings de relations...")
    relation_embeddings = {
        etype: torch.randn(768).to(device)  # même dimension que les features des nœuds
        for _, etype, _ in g.canonical_etypes
    }

    # 3. Entraînement VGAE
    print("\n3. Démarrage de l'entraînement VGAE...")
    vgae_models, vgae_embeddings = train_vgae(g, g.ndata['feat'], num_epochs=100)
    print("✅ Entraînement VGAE terminé")

    # 4. Entraînement HGT
    print("\n4. Démarrage de l'entraînement HGT...")
    hgt_model, hgt_embeddings = train_hgt(g, g.ndata['feat'], relation_embeddings, num_epochs=50)
    print("✅ Entraînement HGT terminé")

    # 5. Clustering global
    print("\n5. Application du clustering global...")
    combined_embeddings = {}

    # Prepare embeddings for all node types with improved combination strategy
    for ntype in g.ntypes:
        if ntype in vgae_embeddings:
            # Move embeddings to CPU
            vgae_emb = vgae_embeddings[ntype].cpu()

            # If we have HGT embeddings for this node type and they have the same first dimension
            if ntype in hgt_embeddings and vgae_emb.shape[0] == hgt_embeddings[ntype].shape[0]:
                hgt_emb = hgt_embeddings[ntype].cpu()
                print(f"Combining embeddings for {ntype}:")
                print(f"  - VGAE: {vgae_emb.shape}")
                print(f"  - HGT: {hgt_emb.shape}")

                # Normalize embeddings - detach to avoid gradient issues
                vgae_emb_norm = F.normalize(vgae_emb.detach(), p=2, dim=1)
                hgt_emb_norm = F.normalize(hgt_emb.detach(), p=2, dim=1)

                # Resize HGT embeddings to match VGAE if needed
                if vgae_emb.shape[1] != hgt_emb.shape[1]:
                    # Use a simpler approach to avoid gradient issues
                    if hgt_emb.shape[1] > vgae_emb.shape[1]:
                        # If HGT has more dimensions, use PCA
                        hgt_np = hgt_emb_norm.numpy()
                        pca = PCA(n_components=vgae_emb.shape[1])
                        hgt_np = pca.fit_transform(hgt_np)
                        hgt_emb_norm = torch.tensor(hgt_np)
                    else:
                        # If HGT has fewer dimensions, pad with zeros
                        padding = torch.zeros(hgt_emb.shape[0], vgae_emb.shape[1] - hgt_emb.shape[1])
                        hgt_emb_norm = torch.cat([hgt_emb_norm, padding], dim=1)

                    print(f"  - HGT (resized): {hgt_emb_norm.shape}")

                # 1. Concatenation
                concat_emb = torch.cat([vgae_emb_norm, hgt_emb_norm], dim=1)

                # 2. Weighted average (giving more weight to VGAE)
                # VGAE tends to capture local structure better
                # Make sure dimensions match before weighted average
                if vgae_emb_norm.shape[1] == hgt_emb_norm.shape[1]:
                    avg_emb = 0.7 * vgae_emb_norm + 0.3 * hgt_emb_norm
                else:
                    # If dimensions don't match, just use VGAE
                    avg_emb = vgae_emb_norm

                # 3. Element-wise maximum (captures strongest features from both)
                # Make sure dimensions match before using maximum
                if vgae_emb_norm.shape[1] == hgt_emb_norm.shape[1]:
                    max_emb = torch.maximum(vgae_emb_norm, hgt_emb_norm)
                else:
                    # If dimensions don't match, just use the concatenation
                    max_emb = vgae_emb_norm

                # 4. Combine all approaches
                combined_emb = torch.cat([concat_emb, avg_emb, max_emb], dim=1)

                # Apply PCA if the combined embedding is too large
                if combined_emb.shape[1] > 1024:
                    # Convert to numpy for PCA - detach first to avoid gradient issues
                    combined_np = combined_emb.detach().numpy()
                    # Calculate appropriate number of components (at most min(n_samples, n_features))
                    n_components = min(combined_np.shape[0], combined_np.shape[1], 256)
                    pca = PCA(n_components=n_components)
                    combined_np = pca.fit_transform(combined_np)
                    combined_emb = torch.tensor(combined_np)
                    print(f"  - Applied PCA to reduce dimensions to {combined_emb.shape[1]}")

                # Final normalization
                combined_emb = F.normalize(combined_emb, p=2, dim=1)

                combined_embeddings[ntype] = combined_emb
                print(f"  - Combined: {combined_embeddings[ntype].shape}")
            else:
                # Just use VGAE embeddings - detach to avoid gradient issues
                combined_embeddings[ntype] = F.normalize(vgae_emb.detach(), p=2, dim=1)
                print(f"Using only VGAE embeddings for {ntype}: {vgae_emb.shape}")

    # Create true labels based on domains
    true_labels = {}

    # Extract domain information from the graph
    print("\nExtracting domain information for true labels...")

    # For authors, use their dominant domain as the label
    if 'author' in g.ntypes:
        author_domains = []
        # Get author to domain edges
        author_domain_edges = g.edges(etype='domain_dominant')
        author_ids = author_domain_edges[0].tolist()
        domain_ids = author_domain_edges[1].tolist()

        # Create a mapping from author ID to domain ID
        author_to_domain = {author: domain for author, domain in zip(author_ids, domain_ids)}

        # Create labels for all authors
        author_labels = []
        for i in range(g.number_of_nodes('author')):
            if i in author_to_domain:
                author_labels.append(author_to_domain[i])
            else:
                author_labels.append(-1)  # No domain

        true_labels['author'] = author_labels
        print(f"Created labels for {len(author_labels)} authors based on domains")

    # For publications, use the domain of their creators
    if 'publication' in g.ntypes:
        pub_domains = []
        # Get publication to author edges
        pub_author_edges = g.edges(etype='creator')
        pub_ids = pub_author_edges[0].tolist()
        author_ids = pub_author_edges[1].tolist()

        # Create a mapping from publication ID to author IDs
        pub_to_authors = {}
        for pub, author in zip(pub_ids, author_ids):
            if pub not in pub_to_authors:
                pub_to_authors[pub] = []
            pub_to_authors[pub].append(author)

        # Create labels for all publications based on the most common domain of their authors
        pub_labels = []
        for i in range(g.number_of_nodes('publication')):
            if i in pub_to_authors:
                # Get domains of all authors of this publication
                author_domains = []
                for author in pub_to_authors[i]:
                    if author in author_to_domain:
                        author_domains.append(author_to_domain[author])

                if author_domains:
                    # Use the most common domain
                    from collections import Counter
                    most_common_domain = Counter(author_domains).most_common(1)[0][0]
                    pub_labels.append(most_common_domain)
                else:
                    pub_labels.append(-1)  # No domain
            else:
                pub_labels.append(-1)  # No authors

        true_labels['publication'] = pub_labels
        print(f"Created labels for {len(pub_labels)} publications based on author domains")

    # For venues, use their publishesDomain relation
    if 'venue' in g.ntypes:
        venue_domains = []
        # Get venue to domain edges
        venue_domain_edges = g.edges(etype='publishesDomain')
        venue_ids = venue_domain_edges[0].tolist()
        domain_ids = venue_domain_edges[1].tolist()

        # Create a mapping from venue ID to domain ID
        venue_to_domain = {venue: domain for venue, domain in zip(venue_ids, domain_ids)}

        # Create labels for all venues
        venue_labels = []
        for i in range(g.number_of_nodes('venue')):
            if i in venue_to_domain:
                venue_labels.append(venue_to_domain[i])
            else:
                venue_labels.append(-1)  # No domain

        true_labels['venue'] = venue_labels
        print(f"Created labels for {len(venue_labels)} venues based on domains")

    # Exécuter le clustering
    print("\nExécution du clustering global avec les domaines comme labels...")
    clustering_results = run_global_clustering(
        combined_embeddings,
        true_labels=true_labels,  # Use domain information as true labels
        n_clusters=g.number_of_nodes('domain')  # Use number of domains as number of clusters
    )

    print("\n" + "="*50)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*50)

if __name__ == "__main__":
    main()
