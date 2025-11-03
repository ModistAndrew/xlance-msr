import argparse
from collections import OrderedDict
import yaml
from pathlib import Path
from typing import Dict, Any, List
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import RawStems, InfiniteSampler
from models import MelRNN, MelRoFormer, UNet, UFormer
from models.bs_roformer import bs_roformer as BSRoformer
from losses.gan_loss import GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss
from losses.reconstruction_loss import MultiMelSpecReconstructionLoss

from modules.discriminator.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from modules.discriminator.MultiScaleDiscriminator import MultiScaleDiscriminator
from modules.discriminator.MultiFrequencyDiscriminator import MultiFrequencyDiscriminator
from modules.discriminator.MultiResolutionDiscriminator import MultiResolutionDiscriminator

class CombinedDiscriminator(nn.Module):
    """A wrapper to combine multiple discriminators into a single module."""    
    def __init__(self, discriminators_config: List[Dict[str, Any]]):
        super().__init__()
        disc_list = []
        for config in discriminators_config:
            name = config['name']
            params = config['params']
            if name == 'MultiPeriodDiscriminator':
                disc_list.append(MultiPeriodDiscriminator(**params))
            elif name == 'MultiScaleDiscriminator':
                disc_list.append(MultiScaleDiscriminator(**params))
            elif name == 'MultiFrequencyDiscriminator':
                disc_list.append(MultiFrequencyDiscriminator(**params))
            elif name == 'MultiResolutionDiscriminator':
                disc_list.append(MultiResolutionDiscriminator(**params))
            else:
                raise ValueError(f"Unknown discriminator type: {name}")
        self.discriminators = nn.ModuleList(disc_list)

    def forward(self, x: torch.Tensor):
        all_scores, all_fmaps = [], []
        for disc in self.discriminators:
            scores, fmaps = disc(x)
            all_scores.extend(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps

class MusicRestorationDataModule(pl.LightningDataModule):
    """Handles data loading for training."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None):
        common_params = {
            "sr": self.config['sample_rate'],
            "clip_duration": self.config['clip_duration'],
        }
        self.train_dataset = RawStems(**self.config['train_dataset'], **common_params)
        self.val_dataset = RawStems(**self.config['val_dataset'], **common_params)
    
    def train_dataloader(self):
        sampler = InfiniteSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            **self.config['dataloader_params']
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.config['dataloader_params']
        )

class MusicRestorationModule(pl.LightningModule):
    """
    PyTorch Lightning module for music source restoration,
    handling model architecture, losses, optimization, and logging.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False # Needed for GANs
        self.use_channel = self.hparams.model['name'] in ['BSRoFormer', 'UFormer']
        self.model_output_loss = self.hparams.model['name'] == 'BSRoFormer'

        # 1. Generator
        self.generator = self._init_generator()
        self.dummy = False
        if hasattr(self.hparams, 'checkpoint'):
            self.load_generator_state_dict()

        # 2. Discriminator
        if hasattr(self.hparams, 'discriminators'):
            self.discriminator = CombinedDiscriminator(self.hparams.discriminators)
        # 3. Losses
        if hasattr(self.hparams, 'losses'):
            loss_cfg = self.hparams.losses
            self.loss_gen_adv = GeneratorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
            self.loss_disc_adv = DiscriminatorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
            self.loss_feat = FeatureMatchingLoss()
            self.loss_recon = MultiMelSpecReconstructionLoss(**loss_cfg['reconstruction_loss'])
        self.l1_loss = nn.L1Loss()
        
    def load_generator_state_dict(self) -> Any:
        path = self.hparams.checkpoint['path']
        type = self.hparams.checkpoint['type']
        full_checkpoint = torch.load(path)
        components = type.split('_')
        if components[0] == 'roformer':
            frozen = ''
            if components[-1] == 'frozen' or components[-1] == 'frozenall':
                frozen = components[-1]
                components.pop()
            if frozen == 'frozenall':
                self.dummy = True
            if len(components) == 1:
                generator_state_dict = OrderedDict()
                filtered_prefix = 'mask_estimators.'
                for key, value in full_checkpoint.items():
                    if key.startswith(filtered_prefix):
                        continue
                    generator_state_dict[key] = value
                missing_keys, unexpected_keys = self.generator.load_state_dict(generator_state_dict, strict=False)
                for key in missing_keys:
                    if key.startswith(filtered_prefix):
                        continue
                    raise ValueError(f"Missing key {key} in state dict")
                for key in unexpected_keys:
                    raise ValueError(f"Unexpected key {key} in state dict")
                print(f"Loaded roformer checkpoint from {path}")
            elif len(components) == 2:
                instrument = components[1]
                generator_state_dict = OrderedDict()
                instrument_index = {'bass': 0, 'drums': 1, 'perc': 1, 'syn': 2, 'orch': 2, 'vox': 3, 'gtr': 4, 'key': 5}
                target_prefix = f'mask_estimators.{instrument_index[instrument]}' 
                new_prefix = 'mask_estimators.0'
                filtered_prefix = 'mask_estimators.'
                for key, value in full_checkpoint.items():
                    if key.startswith(target_prefix):
                        generator_state_dict[new_prefix + key[len(target_prefix):]] = value
                    elif key.startswith(filtered_prefix):
                        continue
                    else:
                        generator_state_dict[key] = value
                self.generator.load_state_dict(generator_state_dict)
                print(f"Loaded roformer vocal checkpoint from {path}")
            else:
                raise ValueError(f"Unknown checkpoint type: {type}")
            if frozen:
                param_dict = dict(self.generator.named_parameters())
                prefix = 'mask_estimators.0'
                frozen_count = 0
                for key in param_dict.keys():
                    if not key.startswith(prefix):
                        param_dict[key].requires_grad = False
                        frozen_count += 1
                        print(f"Frozen: {key}")
                print(f"Loaded roformer checkpoint from {path}")
                print(f"Frozen {frozen_count} parameters from checkpoint")
        else:
            raise ValueError(f"Unknown checkpoint type: {type}")
        
    def _init_generator(self):
        model_cfg = self.hparams.model
        if model_cfg['name'] == 'MelRNN':
            return MelRNN.MelRNN(**model_cfg['params'])
        elif model_cfg['name'] == 'MelRoFormer':
            return MelRoFormer.MelRoFormer(**model_cfg['params'])
        elif model_cfg['name'] == 'MelUNet':
            return UNet.MelUNet(**model_cfg['params'])
        elif model_cfg['name'] == 'UFormer':
            return UFormer.UFormer(UFormer.UFormerConfig(**model_cfg['params']))
        elif model_cfg['name'] == 'BSRoFormer':
            return BSRoformer.BSRoformer(**model_cfg['params'])
        else:
            raise ValueError(f"Unknown model name: {model_cfg['name']}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.dummy:
            return None
        opt_g, opt_d = self.optimizers()
        
        target = batch['target']
        mixture = batch['mixture']
        
        if not self.use_channel:
            # reshape both from (b, c, t) to ((b, c) t)
            target = rearrange(target, 'b c t -> (b c) t')
            mixture = rearrange(mixture, 'b c t -> (b c) t')
        
        # --- Train Discriminator ---
        
        generated = self.generator(mixture)
        if self.use_channel:
            target = rearrange(target, 'b c t -> (b c) t')
            mixture = rearrange(mixture, 'b c t -> (b c) t')
            generated = rearrange(generated, 'b c t -> (b c) t')
        
        real_scores, _ = self.discriminator(target.unsqueeze(1))
        fake_scores, _ = self.discriminator(generated.detach().unsqueeze(1))
        
        d_loss, _, _ = self.loss_disc_adv(real_scores, fake_scores)
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.log('train/d_loss', d_loss, prog_bar=True)

        # --- Train Generator ---
        real_scores, real_fmaps = self.discriminator(target.unsqueeze(1))
        fake_scores, fake_fmaps = self.discriminator(generated.unsqueeze(1))
        
        # Reconstruction Loss
        loss_recon = self.loss_recon(generated, target)
        
        # Adversarial Loss
        loss_adv, _ = self.loss_gen_adv(fake_scores)
        
        # Feature Matching Loss
        loss_feat = self.loss_feat(real_fmaps, fake_fmaps)

        loss_cfg = self.hparams.losses
        g_loss = (
            loss_recon * loss_cfg['lambda_recon'] + 
            loss_adv * loss_cfg['lambda_gan'] + 
            loss_feat * loss_cfg['lambda_feat']
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('train/g_loss', g_loss, prog_bar=True)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss_adv', loss_adv)
        self.log('train/loss_feat', loss_feat)
        
        # Step schedulers
        sch_g, sch_d = self.lr_schedulers()
        if sch_g: sch_g.step()
        if sch_d: sch_d.step()
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        target = batch['target']
        mixture = batch['mixture']
        
        if not self.use_channel:
            # reshape both from (b, c, t) to ((b, c) t)
            target = rearrange(target, 'b c t -> (b c) t')
            mixture = rearrange(mixture, 'b c t -> (b c) t')
        
        with torch.no_grad():
            generated = self.generator(mixture)
            
            if self.use_channel:
                target = rearrange(target, 'b c t -> (b c) t')
                mixture = rearrange(mixture, 'b c t -> (b c) t')
                generated = rearrange(generated, 'b c t -> (b c) t')
        
        loss_recon = self.loss_recon(generated, target)
        
        with torch.no_grad():
            fake_scores, fake_fmaps = self.discriminator(generated.unsqueeze(1))
            real_scores, real_fmaps = self.discriminator(target.unsqueeze(1))
            
            loss_adv, _ = self.loss_gen_adv(fake_scores)
            
            loss_feat = self.loss_feat(real_fmaps, fake_fmaps)
        
        loss_cfg = self.hparams.losses
        g_loss = (
            loss_recon * loss_cfg['lambda_recon'] + 
            loss_adv * loss_cfg['lambda_gan'] + 
            loss_feat * loss_cfg['lambda_feat']
        )
        
        self.log('val/g_loss', g_loss, prog_bar=True)
        self.log('val/loss_recon', loss_recon)
        self.log('val/loss_adv', loss_adv)
        self.log('val/loss_feat', loss_feat)

    def configure_optimizers(self):
        # Generator Optimizer
        opt_g_cfg = self.hparams.optimizer_g
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=opt_g_cfg['lr'], betas=tuple(opt_g_cfg['betas']))
        
        # Discriminator Optimizer
        opt_d_cfg = self.hparams.optimizer_d
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=opt_d_cfg['lr'], betas=tuple(opt_d_cfg['betas']))

        # Schedulers
        if 'warm_up_steps' in self.hparams.scheduler:
            warmup_steps = self.hparams.scheduler['warm_up_steps']
            lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps)
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        
        return [opt_g, opt_d], []

class SimpleMusicRestorationModule(MusicRestorationModule):
    """
    Simplified PyTorch Lightning module for music source restoration.
    Uses only direct loss training without any GAN components.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.automatic_optimization = True # no need to manually optimize
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.dummy:
            return None
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self.common_step(batch, batch_idx, 'val')
    
    def common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, mode: str):
        target = batch['target']
        mixture = batch['mixture']

        if not self.use_channel:
            # reshape both from (b, c, t) to ((b, c) t) for older models
            target = rearrange(target, 'b c t -> (b c) t')
            mixture = rearrange(mixture, 'b c t -> (b c) t')
        
        if self.model_output_loss:
            loss_l1 = self.generator(mixture, target)
        else:
            generated = self.generator(mixture)
            if self.use_channel:
                target = rearrange(target, 'b c t -> (b c) t')
                mixture = rearrange(mixture, 'b c t -> (b c) t')
                generated = rearrange(generated, 'b c t -> (b c) t')
            loss_l1 = self.l1_loss(generated, target)
            
        self.log(f'{mode}/loss_l1', loss_l1, prog_bar=True)
        return loss_l1

    def configure_optimizers(self):
        # Only generator optimizer needed
        opt_g_cfg = self.hparams.optimizer_g
        optimizer = torch.optim.AdamW(
            self.generator.parameters(), 
            lr=opt_g_cfg['lr'], 
            betas=tuple(opt_g_cfg['betas'])
        )
        
        # Schedulers
        if 'warm_up_steps' in self.hparams.scheduler:
            warmup_steps = self.hparams.scheduler['warm_up_steps']
            lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step'
            }
            return [optimizer], [scheduler_config]
        
        return optimizer

def main():
    parser = argparse.ArgumentParser(description="Train a Music Source Restoration Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    pl.seed_everything(42, workers=True)

    data_module = MusicRestorationDataModule(config['data'])
    model_module = MusicRestorationModule(config) if 'discriminators' in config else SimpleMusicRestorationModule(config)

    exp_name = f"{config['exp_name']}"
    exp_name = exp_name.replace(" ", "_")
    save_dir = Path(config['trainer']['save_dir']) / config['project_name'] / exp_name
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{step:08d}",
        every_n_train_steps=config['trainer']['checkpoint_save_interval'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config['trainer']['save_dir'],
        name=config['project_name'],
        version=exp_name
    )
    
    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        limit_train_batches=config['trainer']['limit_train_batches'],
        callbacks=[checkpoint_callback, lr_monitor],
        max_steps=config['trainer']['max_steps'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        accelerator="gpu"
    )
    
    trainer.validate(model_module, datamodule=data_module)
    
    trainer.fit(model_module, datamodule=data_module)

if __name__ == '__main__':
    main()