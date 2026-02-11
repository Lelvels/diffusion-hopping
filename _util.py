from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import DATA_ROOT
from diffusion_hopping.data.dataset import CrossDockedDataModule, PDBBindDataModule
from diffusion_hopping.data.featurization import ProteinLigandSimpleFeaturization
from diffusion_hopping.data.filter import QEDThresholdFilter
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.model.enum import Architecture, Parametrization


def get_data_module_choices():
    choices = ["pdbbind", "crossdocked"]
    choices += [f"{c}_filtered" for c in choices]
    choices += [f"{c}_full" for c in choices]
    return choices


def get_datamodule(dataset_name: str, batch_size: int = 32, shuffle: bool = True):
    if dataset_name not in get_data_module_choices():
        raise ValueError(f"Unknown dataset name {dataset_name}")
    """Create dataset with given name, e.g. crossdocked_filtered or pdbbind_filtered_full"""
    dataset_parts = dataset_name.split("_")
    if dataset_parts[0] == "crossdocked":
        dataset_constructor = CrossDockedDataModule
    elif dataset_parts[0] == "pdbbind":
        dataset_constructor = PDBBindDataModule
    else:
        raise ValueError("Unknown dataset name")

    if len(dataset_parts) == 1:
        pre_transform = ProteinLigandSimpleFeaturization(
            c_alpha_only=True, cutoff=8.0, mode="residue"
        )
        pre_filter = None
    elif len(dataset_parts) == 2:
        if dataset_parts[1] == "filtered":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=True, cutoff=8.0, mode="residue"
            )
            pre_filter = QEDThresholdFilter(0.3)
        elif dataset_parts[1] == "full":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=False, cutoff=8.0, mode="residue"
            )
            pre_filter = None
        else:
            raise ValueError("Unknown dataset name")
    elif len(dataset_parts) == 3:
        if dataset_parts[1] == "filtered" and dataset_parts[2] == "full":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=False, cutoff=8.0, mode="residue"
            )
            pre_filter = QEDThresholdFilter(0.3)
        else:
            raise ValueError("Unknown dataset name")
    else:
        raise ValueError("Unknown dataset name")

    dataset = dataset_constructor(
        str(DATA_ROOT / dataset_name),
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        batch_size=batch_size,
        val_batch_size=32,
        test_batch_size=32,
        shuffle=shuffle,
    )
    return dataset


def get_logger(run, **kwargs):
    return WandbLogger(log_model="all", experiment=run, **kwargs)


def get_callbacks(checkpoint_dir=None):
    """
    Get callbacks for training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints. If None, uses default Lightning location.
    """
    val_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-step_{step}-val_loss_{loss/val:.3f}",
        monitor="loss/val",
        mode="min",
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_last=True,
    )
    
    return [val_checkpoint]


def get_model(
    hidden_features=256,
    num_layers=6,
    joint_features=128,
    condition_on_fg=True,
    architecture=Architecture.EGNN,
    lr=1e-4,
    T=500,
    edge_cutoff=(None, 5, 5),
    ligand_features=10,
    protein_features=20,
    attention=False,
    use_lr_scheduler=False,
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.5,
    lr_scheduler_min_lr=1e-6,
):
    return DiffusionHoppingModel(
        T=T,
        parametrization=Parametrization.EPS,
        lr=lr,
        clip_grad=True,
        condition_on_fg=condition_on_fg,
        x_norm=4.0,
        architecture=architecture,
        edge_cutoff=edge_cutoff,
        hidden_features=hidden_features,
        joint_features=joint_features,
        num_layers=num_layers,
        ligand_features=ligand_features,
        protein_features=protein_features,
        attention=attention,
        use_lr_scheduler=use_lr_scheduler,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_min_lr=lr_scheduler_min_lr,
    )
