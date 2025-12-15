import os
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer 
from tqdm import tqdm
from torchmetrics.classification import MultilabelF1Score
import matplotlib.pyplot as plt 
from Bio import SeqIO
from torch.cuda.amp import autocast, GradScaler

# DEVICE
print("================= DEVICE USING =================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FILE PATH
TRAIN_TERMS_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv'
TRAIN_SEQUENCES_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
TEST_SEQUENCES_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta'
OBO_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo'

PROTEIN_EMBEDDINGS = '/kaggle/input/cafa6-protein-embeddings-esm2/protein_embeddings.npy'
PROTEIN_IDS = '/kaggle/input/cafa6-protein-embeddings-esm2/protein_ids.csv'
GOA_PATH = '/kaggle/input/protein-go-annotations/goa_uniprot_all.csv'

# LOAD EMBEDDING
print("================= LOAD EMBEDDING =================")
print("Loading embeddings ...")
protein_ids = pd.read_csv(PROTEIN_IDS)["protein_id"].tolist()
embeddings = np.load(PROTEIN_EMBEDDINGS) # embed sequences 
embeddings_dict = {pid: emb for pid, emb in zip(protein_ids, embeddings)}
print(f"Loaded {len(protein_ids)} embeddings of dimension {embeddings.shape[1]}")

# READ FASTA
def read_fasta(path):
    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append({
            "ID": record.id,
            "Sequence": str(record.seq)
        })
    return pd.DataFrame(records)
    
def clean_id(raw_id):
    try:
        return raw_id.split('|')[1]
    except:
        return str(raw_id).strip()
        
def parse_fasta(fasta_file: str) -> dict[str, str]:
    seqs, seq_id = {}, None
    with open(fasta_file) as f:
        for line in map(str.strip, f):
            if line.startswith(">"):
                seq_id = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
                seqs[seq_id] = ""
            else:
                seqs[seq_id] += line
    return seqs


print("Loading data ...")
train_terms_df = pd.read_csv(TRAIN_TERMS_PATH, sep="\t")
train_sequences_table = read_fasta(TRAIN_SEQUENCES_PATH)
train_sequences = parse_fasta(TRAIN_SEQUENCES_PATH)
test_sequences = parse_fasta(TEST_SEQUENCES_PATH)

print(f"Train size: {len(train_sequences)}, Test size: {len(test_sequences)}")
print(f"Total annotations: {len(train_terms_df)}")

print(f"Columns in train_terms_df: {train_terms_df.columns.tolist()}")
print(f"Unique aspects: {train_terms_df['aspect'].unique()}")
print(f"Annotations per aspect:")
print(train_terms_df['aspect'].value_counts())

# GO 
def parse_obo(obo_file):
    go_parents, go_children = {}, {}
    term, obsolete = None, False

    with open(obo_file) as f:
        for line in map(str.strip, f):
            if line == "[Term]":
                term, obsolete = None, False

            elif line.startswith("id: GO:"):
                term = line.split("id: ")[1]
                go_parents[term] = []
                go_children.setdefault(term, [])

            elif line == "is_obsolete: true":
                obsolete = True

            elif term and not obsolete and (
                line.startswith("is_a:") or line.startswith("relationship: part_of")
            ):
                parent = line.split()[-1]
                go_parents[term].append(parent)
                go_children.setdefault(parent, []).append(term)

    return go_parents, go_children

def get_all_ancestors(term, go_parents, cache={}):
    if term in cache:
        return cache[term]

    anc, stack = set(), [term]
    while stack:
        for p in go_parents.get(stack.pop(), []):
            if p not in anc:
                anc.add(p)
                stack.append(p)

    cache[term] = anc
    return anc

def get_all_descendants(term, go_children, cache=None):
    cache = {} if cache is None else cache
    if term in cache:
        return cache[term]

    desc, stack = set(), [term]
    while stack:
        for c in go_children.get(stack.pop(), []):
            if c not in desc:
                desc.add(c)
                stack.append(c)

    cache[term] = desc
    return desc


def propagate_predictions(pred_df, go_parents):
    cache = {}
    rows = []

    for pid, g in tqdm(pred_df.groupby("pid"), desc="Propagating"):
        scores = {}

        for term, p in zip(g.term, g.p):
            scores[term] = max(scores.get(term, 0), p)
            for anc in get_all_ancestors(term, go_parents, cache):
                scores[anc] = max(scores.get(anc, 0), p)

        rows.extend({'pid': pid, 'term': t, 'p': s} for t, s in scores.items())

    return pd.DataFrame(rows)


go_parents, go_children = parse_obo(OBO_PATH)

def load_goa_and_build_negative_keys(goa_path, go_children):
    if not os.path.exists(goa_path):
        return set(), None

    goa = pd.read_csv(goa_path).drop_duplicates()

    # --- NEGATIVE ---
    neg = goa[goa.qualifier.str.contains("NOT", na=False)]
    neg_map = neg.groupby("protein_id")["go_term"].apply(list)

    neg_keys, cache = set(), {}
    for p, terms in tqdm(neg_map.items(), desc="Propagating negatives"):
        all_terms = set(terms)
        for t in terms:
            all_terms |= get_all_descendants(t, go_children, cache)
        neg_keys |= {f"{p}_{t}" for t in all_terms}

    # --- POSITIVE ---
    pos = goa[~goa.qualifier.str.contains("NOT", na=False)]
    pos = pos[["protein_id", "go_term"]].drop_duplicates()
    pos["score"] = 1.0
    pos["pred_key"] = pos.protein_id.astype(str) + "_" + pos.go_term.astype(str)
    pos = pos[~pos.pred_key.isin(neg_keys)]

    return neg_keys, pos

def apply_negative_propagation(df, negative_keys):
    if not negative_keys:
        return df

    key = df.pid.astype(str) + "_" + df.term.astype(str)
    return df[~key.isin(negative_keys)]


class ProtDataset(Dataset):
    def __init__(self, pids, labels, embeds):
        self.pids = pids
        self.labels = labels
        self.embeds = embeds

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        x = torch.from_numpy(self.embeds[self.pids[i]]).float()

        y = (
            torch.from_numpy(self.labels[i].toarray().ravel())
            if hasattr(self.labels, "toarray")
            else torch.tensor(self.labels[i])
        ).float()

        return x, y


class TestDataSet(Dataset):
    def __init__(self, pids, embeds):
        self.pids = pids
        self.embeds = embeds

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        return self.pids[i], torch.from_numpy(self.embeds[self.pids[i]]).float()


class SimpleMLP(nn.Module):
    def __init__(self, d=1280, n=3000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512,n)
        )
    def forward(self,x): return self.net(x)


def get_improved_model(num_classes):
    return SimpleMLP(d=1280, n=num_classes)

def train(aspect, model, train_dl, val_dl, epochs, n_cls, lr):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 2)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    f1 = MultilabelF1Score(n_cls, threshold=0.05, average='micro').to(device)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                loss = loss_fn(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)

        model.eval()
        preds, labs, vloss = [], [], 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    out = model(x)
                    vloss += loss_fn(out, y).item() * x.size(0)
                preds.append(torch.sigmoid(out))
                labs.append(y)

        preds, labs = torch.vstack(preds), torch.vstack(labs)
        vf1 = f1(preds, labs).item()
        sched.step(vloss / len(val_dl.dataset))

        print(f"[{aspect}] Epoch {ep}: train_loss={tot/len(train_dl.dataset):.4f}, val_f1={vf1:.4f}")

    return model

ASPECTS = ['C', 'F', 'P']
ASPECT_NAMES = {'C': 'Cellular Component (CCO)', 'F': 'Molecular Function (MFO)', 'P': 'Biological Process (BPO)'}
aspect_models = {}
aspect_mlbs = {}


def train_aspect_model(aspect):
    df = train_terms_df[train_terms_df.aspect == aspect]
    if df.empty:
        raise ValueError(f"No data for aspect {aspect}")

    prot2terms = df.groupby('EntryID')['term'].apply(list)
    pids = [p for p in train_sequences if p in embeddings_dict and p in prot2terms]

    mlb = MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform([prot2terms[p] for p in pids])
    aspect_mlbs[aspect] = mlb

    ds = ProtDataset(pids, Y, embeddings_dict)
    n = int(0.9 * len(ds))
    tr, va = random_split(ds, [n, len(ds)-n])

    tr_dl = DataLoader(tr, 128, shuffle=True, num_workers=4)
    va_dl = DataLoader(va, 128, shuffle=False, num_workers=4)

    model = train(
        aspect,
        get_improved_model(len(mlb.classes_)),
        tr_dl, va_dl,
        epochs=30,
        n_cls=len(mlb.classes_),
        lr=1e-3
    )

    aspect_models[aspect] = model
    torch.cuda.empty_cache()
    return model, mlb

ASPECT_CFG = {
    'C': dict(thr=0.02, min_k=20, power=1.1),
    'F': dict(thr=0.02, min_k=15, power=1.1),
    'P': dict(thr=0.04, min_k=10, power=1.2),
}

def predict_ensemble():
    pids = [p for p in test_sequences if p in embeddings_dict]

    loader = DataLoader(
        TestDataSet(pids, embeddings_dict),
        batch_size=128,
        shuffle=False,
        num_workers=3
    )

    rows = []

    for asp in ASPECTS:
        cfg = ASPECT_CFG[asp]
        model, mlb = aspect_models[asp], aspect_mlbs[asp]
        model.eval()

        for pids_batch, x in tqdm(loader, desc=f"Predict {asp}"):
            x = x.to(device)

            with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
                probs = torch.sigmoid(model(x)).cpu().numpy()

            probs = probs ** cfg["power"]

            for pid, s in zip(pids_batch, probs):
                ids = np.where(s >= cfg["thr"])[0]
                if len(ids) < cfg["min_k"]:
                    ids = np.argsort(s)[-cfg["min_k"]:]

                rows.extend(
                    {'pid': pid, 'term': mlb.classes_[j], 'p': float(s[j])}
                    for j in ids
                )

        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


if __name__ == "__main__":

    # GOA negatives
    neg_keys, _ = load_goa_and_build_negative_keys(GOA_PATH, go_children)

    # Train
    for asp in ASPECTS:
        train_aspect_model(asp)

    # Predict (ASPECT-AWARE)
    df = predict_ensemble()

    # OBO + GOA negative
    df = propagate_predictions(df, go_parents)
    df = apply_negative_propagation(df, neg_keys)

    df.sort_values("p", ascending=False)\
      .to_csv("submission.tsv", sep="\t", index=False, header=False)


# ============= Blend GOA =============
# import pandas as pd

# # ======================
# # CONFIG
# # ======================
# MY_SUB = "/kaggle/input/sub0253/submission_0.253.tsv"
# GOA_SUB = "/kaggle/input/cafa-6-blend-goa-negative-propagation/submission.tsv"
# OUT_SUB = "submission.tsv"

# BOOST_GOA = 1.03 

# # ======================
# # LOAD
# # ======================
# cols = ["pid", "term", "p"]

# df_my = pd.read_csv(MY_SUB, sep="\t", names=cols)
# df_goa = pd.read_csv(GOA_SUB, sep="\t", names=cols)

# print("My submission:", len(df_my))
# print("GOA submission:", len(df_goa))

# # ======================
# # BOOST GOA (NHẸ)
# # ======================
# df_goa = df_goa.copy()
# df_goa["p"] = (df_goa["p"] * BOOST_GOA).clip(upper=1.0)

# # ======================
# # BLEND (MAX SCORE)
# # ======================
# blend_df = pd.concat([df_my, df_goa], ignore_index=True)

# blend_df = (
#     blend_df
#     .groupby(["pid", "term"], as_index=False)["p"]
#     .max()
# )

# print("After blend:", len(blend_df))

# # ======================
# # SORT & SAVE
# # ======================
# blend_df = blend_df.sort_values("p", ascending=False)

# blend_df.to_csv(
#     OUT_SUB,
#     sep="\t",
#     index=False,
#     header=False
# )

# print("Saved blended submission to:", OUT_SUB)



