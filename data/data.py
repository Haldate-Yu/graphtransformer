"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.TUs import TUsDataset


def LoadData(DATASET_NAME, pos_enc_dim=10):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)

    # handling for TU datasets
    # specially, we set a min_node_num for TU datasets
    # To filter those graphs with fewer nodes than the setting pos_enc_dim
    TU_DATASETS = ['MUTAG', 'NCI1', 'NCI109', 'DD', 'PROTEINS_full', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY']
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME, pos_enc_dim)

    # Todo:
    # adding ogbn large datasets
