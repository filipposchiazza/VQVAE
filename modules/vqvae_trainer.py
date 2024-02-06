import torch
import torch.nn.functional as F
from tqdm import tqdm



class VQVAETrainer():

    def __init__(self,
                 vqvae,
                 optimizer,
                 num_epochs,
                 device,
                 learning_rate_scheduler=None,
                 early_stopper=None,
                 verbose=True):
        
        self.vqvae = vqvae.to(device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.learning_rate_scheduler = learning_rate_scheduler
        self.early_stopper = early_stopper
        self.verbose = verbose



    def train(self,
              train_dataloader,
              rec_loss_fn,
              val_dataloader=None,
              checkpoint_folder=None):
        
        # Store the number of codebook vectors used for each epoch by the VQVAE
        self.codebooks = []

        # Store the training and validation history
        self.history = {'loss_rec': [],
                        'loss_quant': [],
                        'val_loss_rec': [],
                        'val_loss_quant': []}
        
        for epoch in range(self.num_epochs):

            # Training mode
            self.vqvae.train()

            # Train one epoch
            loss_rec, loss_quant, loss_tot = self._train_one_epoch(train_dataloader=train_dataloader,
                                                                   rec_loss_fn=rec_loss_fn,
                                                                   epoch=epoch)
            
            # Store training losses
            self.history['loss_rec'].append(loss_rec)
            self.history['loss_quant'].append(loss_quant)

            if val_dataloader is not None:
                # Validation mode
                self.vqvae.eval()

                # Validation step
                val_loss_rec, val_loss_quant, val_loss_tot = self._validate(val_dataloader=val_dataloader,
                                                                            rec_loss_fn=rec_loss_fn)
                
                # Store validation losses
                self.history['val_loss_rec'].append(val_loss_rec)
                self.history['val_loss_quant'].append(val_loss_quant)

            # Early stopping
            if self.early_stopper is not None and val_dataloader is not None:
                if self.early_stopper.early_stopping(monitor=val_loss_tot,
                                                     model=self.vqvae,
                                                     checkpoint_folder=checkpoint_folder):
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            elif self.early_stopper is not None:
                if self.early_stopper.early_stopping(monitor=loss_tot,
                                                     model=self.vqvae,
                                                     checkpoint_folder=checkpoint_folder):
                    print(f'Early stopping at epoch {epoch+1}')
                    break
                
                
            # Update the learning rate
            if self.learning_rate_scheduler is not None:
                self.learning_rate_scheduler.step()

        return self.history, self.codebooks        



    def _train_one_epoch(self,
                         train_dataloader,
                         rec_loss_fn,
                         epoch):
        
        running_loss_rec = 0.
        running_loss_quant = 0.
        running_loss_tot = 0.

        mean_loss_rec = 0.
        mean_loss_quant = 0.
        mean_loss_tot = 0.

        # Iterate over the training batches
        with tqdm(train_dataloader, unit='batches') as tepoch:
            for batch_idx, imgs in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Move the data to the device
                imgs = imgs.to(self.device)

                # Forward pass
                loss_quant, imgs_rec, encoding_indices, _ = self.vqvae(imgs)
                current_codebook_usage = len(torch.unique(encoding_indices.detach().cpu()))
                self.codebooks.append(current_codebook_usage)

                # Compute the total loss
                loss_rec = rec_loss_fn(imgs_rec, imgs)
                loss_tot = loss_rec + loss_quant

                # Backward pass
                self.optimizer.zero_grad()
                loss_tot.backward()
                self.optimizer.step()

                # Update the running and mean losses
                running_loss_rec += loss_rec.item()
                running_loss_quant += loss_quant.item()
                running_loss_tot += loss_tot.item()
                mean_loss_rec = running_loss_rec / (batch_idx+1)
                mean_loss_quant = running_loss_quant / (batch_idx+1)
                mean_loss_tot = running_loss_tot / (batch_idx+1)

                # Update the progress bar
                tepoch.set_postfix({'loss_rec': '{:.6f}'.format(mean_loss_rec),
                                    'loss_quant': '{:.6f}'.format(mean_loss_quant),
                                    'loss_tot': '{:.6f}'.format(mean_loss_tot),
                                    'codebook_usage': f'{current_codebook_usage}/{self.vqvae.num_emb}'})
        
        return mean_loss_rec, mean_loss_quant, mean_loss_tot



    def _validate(self,
                  val_dataloader,
                  rec_loss_fn):
        
        running_val_loss_rec = 0.
        running_val_loss_quant = 0.
        running_val_loss_tot = 0.

        with torch.no_grad():
            for batch_idx, imgs in enumerate(val_dataloader):

                # Move the data to the device
                imgs = imgs.to(self.device)

                # Forward pass
                loss_quant, imgs_rec, _, _ = self.vqvae(imgs)

                # Compute the total loss
                loss_rec = rec_loss_fn(imgs_rec, imgs)
                loss_tot = loss_rec + loss_quant

                # Update the running and mean losses
                running_val_loss_rec += loss_rec.item()
                running_val_loss_quant += loss_quant.item()
                running_val_loss_tot += loss_tot.item()
                
                
        mean_val_loss_rec = running_val_loss_rec / len(val_dataloader)
        mean_val_loss_quant = running_val_loss_quant / len(val_dataloader)
        mean_val_loss_tot = running_val_loss_tot / len(val_dataloader)

        if self.verbose == True:
            print(f'Validation loss: rec={mean_val_loss_rec:.6f}, quant={mean_val_loss_quant:.6f}, tot={mean_val_loss_tot:.6f}')

        return mean_val_loss_rec, mean_val_loss_quant, mean_val_loss_tot


