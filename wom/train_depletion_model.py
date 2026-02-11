import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

# Load data
f = '/pscratch/sd/b/bpb/wom_data_cache/cache/exometabolite.parquet'
df = pd.read_parquet(f)

# For depletion model: positive = consumed, negative = everything else
positive_label = "consumed"

# Build mappings
microbes = df["organism_id"].unique()
metabolites = df["chebi_id"].unique()

microbe_to_idx = {m: i for i, m in enumerate(microbes)}
metabolite_to_idx = {m: i for i, m in enumerate(metabolites)}
idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}
idx_to_metabolite = {i: m for m, i in metabolite_to_idx.items()}

n_microbes = len(microbes)
n_metabolites = len(metabolites)

print(f"Microbes: {n_microbes}, Metabolites: {n_metabolites}")

# Build taxonomy lookup for each microbe (take first occurrence)
microbe_taxonomy = {}
for _, row in df.drop_duplicates("organism_id").iterrows():
    microbe_taxonomy[row["organism_id"]] = {
        "kingdom": row["kingdom"],
        "class": row["class"],
        "order": row["order"],
        "family": row["family"],
        "genus": row["genus"]
    }

# Build positive set per microbe
positives = defaultdict(set)

for _, row in df.iterrows():
    if row["assertion"] == positive_label:
        m_idx = microbe_to_idx[row["organism_id"]]
        c_idx = metabolite_to_idx[row["chebi_id"]]
        positives[m_idx].add(c_idx)

# BPR Model
class BPRModel(nn.Module):
    def __init__(self, n_microbes, n_metabolites, embedding_dim):
        super().__init__()
        self.microbe_embeddings = nn.Embedding(n_microbes, embedding_dim)
        self.metabolite_embeddings = nn.Embedding(n_metabolites, embedding_dim)
        
        nn.init.normal_(self.microbe_embeddings.weight, std=0.01)
        nn.init.normal_(self.metabolite_embeddings.weight, std=0.01)
    
    def forward(self, microbe_idx, pos_metabolite_idx, neg_metabolite_idx):
        microbe_emb = self.microbe_embeddings(microbe_idx)
        pos_emb = self.metabolite_embeddings(pos_metabolite_idx)
        neg_emb = self.metabolite_embeddings(neg_metabolite_idx)
        
        pos_score = (microbe_emb * pos_emb).sum(dim=1)
        neg_score = (microbe_emb * neg_emb).sum(dim=1)
        
        return pos_score, neg_score
    
    def predict(self, microbe_idx):
        microbe_emb = self.microbe_embeddings(torch.tensor([microbe_idx]))
        all_metabolite_emb = self.metabolite_embeddings.weight
        scores = torch.matmul(microbe_emb, all_metabolite_emb.T).squeeze()
        return scores

def sample_triplets(positives, n_metabolites, batch_size):
    microbe_indices = []
    pos_indices = []
    neg_indices = []
    
    microbe_list = [m for m in positives.keys() if len(positives[m]) > 0]
    
    for _ in range(batch_size):
        m_idx = np.random.choice(microbe_list)
        pos_idx = np.random.choice(list(positives[m_idx]))
        
        neg_idx = np.random.randint(n_metabolites)
        while neg_idx in positives[m_idx]:
            neg_idx = np.random.randint(n_metabolites)
        
        microbe_indices.append(m_idx)
        pos_indices.append(pos_idx)
        neg_indices.append(neg_idx)
    
    return (torch.tensor(microbe_indices), 
            torch.tensor(pos_indices), 
            torch.tensor(neg_indices))

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

# Training
embedding_dim = 32
learning_rate = 0.01
n_epochs = 1000
batch_size = 256

model = BPRModel(n_microbes, n_metabolites, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(n_epochs):
    microbe_batch, pos_batch, neg_batch = sample_triplets(positives, n_metabolites, batch_size)
    
    pos_scores, neg_scores = model(microbe_batch, pos_batch, neg_batch)
    loss = bpr_loss(pos_scores, neg_scores)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ============================================
# SAVE MODEL AND MAPPINGS
# ============================================

# Save model weights
torch.save(model.state_dict(), "bpr_depletion_model.pt")

# Save mappings and metadata needed for reloading
model_metadata = {
    "microbe_to_idx": microbe_to_idx,
    "metabolite_to_idx": metabolite_to_idx,
    "idx_to_microbe": idx_to_microbe,
    "idx_to_metabolite": idx_to_metabolite,
    "microbe_taxonomy": microbe_taxonomy,
    "positives": dict(positives),
    "n_microbes": n_microbes,
    "n_metabolites": n_metabolites,
    "embedding_dim": embedding_dim
}

with open("bpr_depletion_metadata.pkl", "wb") as f:
    pickle.dump(model_metadata, f)

print("Model saved to bpr_depletion_model.pt")
print("Metadata saved to bpr_depletion_metadata.pkl")

# ============================================
# LOAD MODEL (for future use)
# ============================================

def load_model(model_path, metadata_path):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    model = BPRModel(
        metadata["n_microbes"], 
        metadata["n_metabolites"], 
        metadata["embedding_dim"]
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, metadata

# ============================================
# T-SNE VISUALIZATION
# ============================================

def plot_microbe_embeddings(model, microbe_taxonomy, idx_to_microbe, 
                            color_by="genus", perplexity=30, figsize=(12, 10)):
    
    # Extract embeddings
    embeddings = model.microbe_embeddings.weight.detach().numpy()
    
    # Run t-SNE
    from sklearn import __version__ as sklearn_version

    if int(sklearn_version.split('.')[1]) >= 4:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)

    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Build dataframe for plotting
    plot_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "microbe_id": [idx_to_microbe[i] for i in range(len(embeddings))]
    })
    
    # Add taxonomy
    plot_df["kingdom"] = plot_df["microbe_id"].map(lambda m: microbe_taxonomy[m]["kingdom"])
    plot_df["class"] = plot_df["microbe_id"].map(lambda m: microbe_taxonomy[m]["class"])
    plot_df["order"] = plot_df["microbe_id"].map(lambda m: microbe_taxonomy[m]["order"])
    plot_df["family"] = plot_df["microbe_id"].map(lambda m: microbe_taxonomy[m]["family"])
    plot_df["genus"] = plot_df["microbe_id"].map(lambda m: microbe_taxonomy[m]["genus"])
    
    # Handle NaN in taxonomy
    plot_df[color_by] = plot_df[color_by].fillna("Unknown")
    
    # Get unique categories and assign colors
    categories = plot_df[color_by].unique()
    cmap = plt.cm.get_cmap("tab20", len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for category in categories:
        subset = plot_df[plot_df[color_by] == category]
        ax.scatter(
            subset["x"], 
            subset["y"], 
            c=[color_map[category]], 
            label=category, 
            alpha=0.7,
            s=50
        )
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Microbe Embeddings (BPR Depletion Model)\nColored by {color_by}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    
    plt.savefig(f"microbe_embeddings_tsne_{color_by}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"microbe_embeddings_tsne_{color_by}.pdf", bbox_inches="tight")
    print(f"Saved: microbe_embeddings_tsne_{color_by}.png/pdf")
    
    plt.show()
    
    return plot_df



# Generate plots at different taxonomic levels
plot_df = plot_microbe_embeddings(model, microbe_taxonomy, idx_to_microbe, color_by="order")
plot_microbe_embeddings(model, microbe_taxonomy, idx_to_microbe, color_by="class")
plot_microbe_embeddings(model, microbe_taxonomy, idx_to_microbe, color_by="family")
plot_microbe_embeddings(model, microbe_taxonomy, idx_to_microbe, color_by="genus")

# ============================================
# PLOT TRAINING LOSS
# ============================================

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("BPR Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png", dpi=150)
plt.show()