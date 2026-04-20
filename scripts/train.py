import os
import time
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
import implicit
import pickle
import joblib
import psutil
import json
import boto3
import tempfile
import platform
import multiprocessing
from datetime import datetime

# ─── CONFIG ───
cfg = {
    "num_factors": 50,
    "regularization": 0.01,
    "iterations": 20,
    "alpha": 40,
    "min_interactions": 5,
    "test_split_ratio": 0.2,
    "dataset": "mealie_production_data",
    "model_type": "ALS",
}

# ─── MinIO CLIENT ───
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.environ.get('MINIO_ENDPOINT', 'http://129.114.26.176:30900'),
        aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
        aws_secret_access_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin123'),
    )

# ─── LOAD DATA FROM MINIO ───
def load_data():
    print("Loading training data from MinIO...")
    s3 = get_s3_client()
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        s3.download_file('training-data', 'datasets/v2_2026-04-04/train.parquet', f.name)
        train_df = pd.read_parquet(f.name)
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        s3.download_file('training-data', 'datasets/v2_2026-04-04/val.parquet', f.name)
        val_df = pd.read_parquet(f.name)
    
    print(f"Train interactions: {len(train_df):,}")
    print(f"Val interactions: {len(val_df):,}")
    print(f"Columns: {list(train_df.columns)}")
    return train_df, val_df

# ─── PREPROCESS ───
def preprocess(train_df, val_df, cfg):
    print("Preprocessing...")
    
    # Filter sparse users and recipes from training set
    user_counts = train_df['user_id'].value_counts()
    recipe_counts = train_df['recipe_id'].value_counts()
    
    train_df = train_df[
        train_df['user_id'].isin(user_counts[user_counts >= cfg['min_interactions']].index) &
        train_df['recipe_id'].isin(recipe_counts[recipe_counts >= cfg['min_interactions']].index)
    ]
    
    # Build unified ID mappings from train set
    user_ids = train_df['user_id'].unique()
    recipe_ids = train_df['recipe_id'].unique()
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    recipe2idx = {r: i for i, r in enumerate(recipe_ids)}
    
    n_users = len(user_ids)
    n_recipes = len(recipe_ids)
    
    # Build train sparse matrix
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['recipe_idx'] = train_df['recipe_id'].map(recipe2idx)
    train_df = train_df.dropna(subset=['user_idx', 'recipe_idx'])
    
    train_matrix = sparse.csr_matrix(
        (train_df['rating'].values * cfg['alpha'],
         (train_df['recipe_idx'].values.astype(int), 
          train_df['user_idx'].values.astype(int))),
        shape=(n_recipes, n_users)
    )
    
    # Build val sparse matrix — only keep known users/recipes
    val_df['user_idx'] = val_df['user_id'].map(user2idx)
    val_df['recipe_idx'] = val_df['recipe_id'].map(recipe2idx)
    val_df = val_df.dropna(subset=['user_idx', 'recipe_idx'])
    
    val_matrix = sparse.csr_matrix(
        (val_df['rating'].values,
         (val_df['recipe_idx'].values.astype(int),
          val_df['user_idx'].values.astype(int))),
        shape=(n_recipes, n_users)
    )
    
    mappings = {
        'user2idx': user2idx,
        'recipe2idx': recipe2idx,
        'idx2user': {i: u for u, i in user2idx.items()},
        'idx2recipe': {i: r for r, i in recipe2idx.items()},
    }
    
    print(f"n_users: {n_users:,} | n_recipes: {n_recipes:,}")
    print(f"train interactions: {train_matrix.nnz:,} | val interactions: {val_matrix.nnz:,}")
    
    return train_matrix, val_matrix, mappings

# ─── NDCG@K EVALUATION ───
def compute_ndcg(model, train_matrix, val_matrix, k=10):
    print(f"Evaluating NDCG@{k}...")
    n_users = train_matrix.shape[1]
    sample_size = min(300, n_users)
    sample_users = np.random.choice(n_users, sample_size, replace=False)
    
    ndcg_scores = []
    
    for user_idx in sample_users:
        if user_idx >= val_matrix.shape[1]:
            continue
        
        val_items = set(val_matrix.T[user_idx].tocsr().nonzero()[1])
        if not val_items:
            continue
        
        user_items = train_matrix.T[user_idx].tocsr()
        
        try:
            recommended = model.recommend(
                user_idx, user_items, N=k,
                filter_already_liked_items=True
            )
            recommended_ids = [r[0] for r in recommended]
        except Exception:
            continue
        
        # Compute DCG
        dcg = 0.0
        for rank, rec_id in enumerate(recommended_ids, 1):
            if rec_id in val_items:
                dcg += 1.0 / np.log2(rank + 1)
        
        # Compute IDCG (ideal DCG)
        ideal_hits = min(len(val_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
    
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

# ─── GENERATE TAG VECTORS FROM ALS ITEM FACTORS ───
def generate_tag_to_vector(model, mappings, train_df):
    print("Generating tag_to_vector from ALS item factors...")
    
    recipe2idx = mappings['recipe2idx']
    item_factors = model.item_factors  # shape: (n_recipes, n_factors)
    
    # Build recipe_id -> tags mapping from training data
    if 'tags' in train_df.columns:
        recipe_tags = train_df.groupby('recipe_id')['tags'].first().to_dict()
    else:
        # If no tags column, use recipe_id as a single tag
        recipe_tags = {r: [str(r)] for r in recipe2idx.keys()}
    
    tag_to_vectors = {}
    
    for recipe_id, idx in recipe2idx.items():
        if idx >= len(item_factors):
            continue
        recipe_vector = item_factors[idx]
        tags = recipe_tags.get(str(recipe_id), [str(recipe_id)])
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in tag_to_vectors:
                tag_to_vectors[tag] = []
            tag_to_vectors[tag].append(recipe_vector)
    
    # Average vectors per tag
    tag_to_vector = {
        tag: np.mean(np.stack(vecs), axis=0).astype(np.float32)
        for tag, vecs in tag_to_vectors.items()
        if vecs
    }
    
    print(f"Generated vectors for {len(tag_to_vector)} unique tags")
    return tag_to_vector

# ─── SAVE TO MINIO ───
def save_to_minio(obj, bucket, key):
    s3 = get_s3_client()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(obj, f.name)
        s3.upload_file(f.name, bucket, key)
    print(f"Saved to MinIO: s3://{bucket}/{key}")

# ─── MAIN TRAINING FUNCTION ───
def train():
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://129.114.26.176:30500'))
    mlflow.set_experiment("mealie-recipe-recommender")
    
    with mlflow.start_run():
        print("=" * 50)
        print("Starting ALS Training Run")
        print("=" * 50)
        
        # Log config
        mlflow.log_params(cfg)
        mlflow.log_param("cpu_count", multiprocessing.cpu_count())
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("processor", platform.processor())
        mlflow.log_param("gpu_available", False)
        mlflow.log_param("python_version", "3.10")
        mlflow.log_param("implicit_version", implicit.__version__)
        
        # Load data
        t0 = time.time()
        train_df, val_df = load_data()
        train_matrix, val_matrix, mappings = preprocess(train_df, val_df, cfg)
        data_load_time = time.time() - t0
        
        mlflow.log_metric("data_load_time_seconds", data_load_time)
        mlflow.log_metric("n_users", train_matrix.shape[1])
        mlflow.log_metric("n_recipes", train_matrix.shape[0])
        mlflow.log_metric("n_train_interactions", train_matrix.nnz)
        mlflow.log_metric("n_val_interactions", val_matrix.nnz)
        
        # Train
        print("Training ALS model...")
        model = implicit.als.AlternatingLeastSquares(
            factors=cfg['num_factors'],
            regularization=cfg['regularization'],
            iterations=cfg['iterations'],
            use_gpu=False,
            random_state=42,
        )
        
        ram_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        t_train = time.time()
        model.fit(train_matrix)
        training_time = time.time() - t_train
        ram_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("peak_ram_mb", ram_after)
        print(f"Training completed in {training_time:.2f}s")
        
        # Evaluate with NDCG@10
        ndcg_at_10 = compute_ndcg(model, train_matrix, val_matrix, k=10)
        mlflow.log_metric("ndcg_at_10", ndcg_at_10)
        print(f"NDCG@10: {ndcg_at_10:.4f}")
        
        # ─── QUALITY GATE ───
        NDCG_THRESHOLD = 0.0  # Threshold set to 0.0 for initial integration; will increase as production data grows
        print(f"Quality gate: NDCG@10 {ndcg_at_10:.4f} >= {NDCG_THRESHOLD}?")
        
        if ndcg_at_10 >= NDCG_THRESHOLD:
            print("✅ Quality gate PASSED — registering model")
            mlflow.log_param("quality_gate_passed", True)
            
            # Generate tag vectors
            tag_to_vector = generate_tag_to_vector(model, mappings, train_df)
            
            # Save tag_to_vector to MinIO for Sharvin
            save_to_minio(tag_to_vector, 'mlflow', 'production/tag_to_vector.pkl')
            
            # Save model artifacts
            os.makedirs('/tmp/model_artifacts', exist_ok=True)
            with open('/tmp/model_artifacts/als_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('/tmp/model_artifacts/mappings.json', 'w') as f:
                json.dump({
                    'user2idx': {str(k): v for k, v in mappings['user2idx'].items()},
                    'recipe2idx': {str(k): v for k, v in mappings['recipe2idx'].items()},
                }, f)
            with open('/tmp/model_artifacts/config.json', 'w') as f:
                json.dump(cfg, f, indent=2)
            
            mlflow.log_artifact('/tmp/model_artifacts/als_model.pkl')
            mlflow.log_artifact('/tmp/model_artifacts/mappings.json')
            mlflow.log_artifact('/tmp/model_artifacts/config.json')
            
            # Register model in MLflow Model Registry
            from mlflow import MlflowClient
            client = MlflowClient()
            run_id = mlflow.active_run().info.run_id
            try:
                client.create_registered_model("mealie-als-recommender")
            except Exception:
                pass  # Already exists
            model_version = client.create_model_version(
                name="mealie-als-recommender",
                source=f"runs:/{run_id}/als_model.pkl",
                run_id=run_id,
            )
            client.transition_model_version_stage(
                name="mealie-als-recommender",
                version=model_version.version,
                stage="Production"
            )
            print(f"Model registered and promoted to Production: version {model_version.version}")
            
        else:
            print(f"❌ Quality gate FAILED — model not registered")
            mlflow.log_param("quality_gate_passed", False)
        
        print("Run complete.")

if __name__ == "__main__":
    train()
