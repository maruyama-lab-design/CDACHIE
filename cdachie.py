import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from hmmlearn import hmm
from omegaconf import OmegaConf

from src import data_for_clustering, lightning_model, make_bed

# Load configuration using OmegaConf
config = OmegaConf.load("config.yaml")

def main():
    pl.seed_everything(0, workers=True)

    # Data loading
    data_cfg = config.data
    file_cfg = config.file_paths
    
    all_train_loader = data_for_clustering.get_dataloader(
        file_cfg.signals,
        file_cfg.clusters,
        file_cfg.hic_embedding,
        mode='all',
        batch_size=data_cfg.batch_size,
        shuffle=True,
        scaler=True,
        signal_resolution=data_cfg.signal_resolution
    )

    all_loader = data_for_clustering.get_dataloader(
        file_cfg.signals,
        file_cfg.clusters,
        file_cfg.hic_embedding,
        mode='all',
        batch_size=data_cfg.batch_size,
        shuffle=False,
        scaler=True,
        signal_resolution=data_cfg.signal_resolution
    )

    # Model initialization
    model_cfg = config.model
    optim_cfg = config.optimizer
    
    model = lightning_model.ClusteringModel(
        signal_resolution=data_cfg.signal_resolution,
        hic_embedding_dim=data_cfg.hic_embedding_dim,
        share_attn=model_cfg.share_attn,
        share_ffn=model_cfg.share_ffn,
        dropout=model_cfg.dropout,
        d_model=model_cfg.d_model,
        dim_feedforward=model_cfg.dim_feedforward,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        func_lr=optim_cfg.func_lr,
        struc_lr=optim_cfg.struc_lr,
        func_head_lr=optim_cfg.func_head_lr,
        struc_head_lr=optim_cfg.struc_head_lr,
        weight_decay=optim_cfg.weight_decay,
        optimizer=optim_cfg.type
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        enable_checkpointing=config.training.enable_checkpointing,
        callbacks=[lightning_model.OverrideEpochStepCallback()]
    )
    trainer.fit(model, all_train_loader)

    # Testing
    model.test_outputs = {'func_emb': [], 'struc_emb': []}
    trainer.test(model, all_loader)

    # Clustering
    func_emb = normalize(model.test_outputs['func_emb'], axis=1)
    struc_emb = normalize(model.test_outputs['struc_emb'], axis=1)
    concat_emb = np.concatenate([func_emb, struc_emb], axis=1)

    clustering_cfg = config.clustering
    if clustering_cfg.method == 'kmeans':
        cluster = KMeans(n_clusters=clustering_cfg.n_clusters, random_state=0).fit_predict(concat_emb)
    else:
        hmm_model = hmm.GaussianHMM(
            n_components=clustering_cfg.n_clusters,
            covariance_type="full",
            random_state=0
        ).fit(func_emb)
        cluster = hmm_model.predict(model.test_outputs['func_emb'])

    # Save results
    print('Saving results...')
    df = pd.read_csv(file_cfg.clusters)
    df['new_cluster'] = cluster
    df.to_csv(file_cfg.clusters, index=False)
    #make_bed.csv_to_bed(file_cfg.clusters, file_cfg.output_bed, num_clusters=clustering_cfg.n_clusters)
    print('Done!')

if __name__ == "__main__":
    main()