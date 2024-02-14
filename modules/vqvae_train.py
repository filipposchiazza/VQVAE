import config
from dataset import ImageDataset, prepare_ImageDataset
from vqvae import VQVAE
from vqvae_trainer import VQVAETrainer
from early_stopper import EarlyStopper
from loss import Variance_weighted_loss
from torch import optim
import torch.nn.functional as F

# Load image data
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR, 
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM, 
                                                                                    fraction=config.FRACTION)

# Create VQVAE model
vqvae = VQVAE(input_channels=config.INPUT_CHANNELS,
              output_channels=config.OUT_CHANNELS,
              channels=config.CHANNELS,
              num_resblock=config.NUM_RES_BLOCK,
              num_emb=config.NUM_EMB_VECTORS,
              emb_dim=config.EMB_DIM,
              groups=config.GROUPS,
              beta=config.BETA,
              ema_update=config.EMA_UPDATE)


# Create optimizer, early stopper and learning rate scheduler
optimizer = optim.Adam(vqvae.parameters(), lr=config.LEARNING_RATE)

learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                               T_max=config.NUM_EPOCHS,
                                                               eta_min=1e-6,
                                                               verbose=True)

early_stopper = EarlyStopper(patience=config.PATIENCE,
                             min_delta=config.MIN_DELTA)


# Define the reconstruction loss function
rec_loss_fn = Variance_weighted_loss(device=config.DEVICE)


# Create VQVAE trainer
trainer = VQVAETrainer(vqvae=vqvae,
                       optimizer=optimizer,
                       num_epochs=config.NUM_EPOCHS,
                       learning_rate_scheduler=learning_rate_scheduler,
                       early_stopper=early_stopper,
                       device=config.DEVICE)


# Train VQVAE
history, codebooks = trainer.train(train_dataloader=train_dataloader,
                                   rec_loss_fn=rec_loss_fn,
                                   val_dataloader=val_dataloader,
                                   checkpoint_folder=config.SAVE_FOLDER)


# Save VQVAE model, history and codebooks usage
if early_stopper is not None:
    vqvae.save_model(save_folder=config.SAVE_FOLDER)

vqvae.save_history(history=history, 
                   save_folder=config.SAVE_FOLDER)

vqvae.save_codebook_usage(codebooks=codebooks,
                          save_folder=config.SAVE_FOLDER)

