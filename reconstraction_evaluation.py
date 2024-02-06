import torch
from modules.vqvae import VQVAE
import matplotlib.pyplot as plt
from modules.dataset import ImageDataset, prepare_ImageDataset
import config


# Parameters
MODEL_FOLDER = "./models/01"   # Path to the folder where the model is saved
NUM_IMAGES = 10                # Number of images to compare between real and reconstructed



def plot_dataset(dataset, num_images, title='Dataset sample'):
    
    fig, axes = plt.subplots(2, int(num_images/2), figsize=(12, 6))
    fig.suptitle(title, fontsize=20)
    
    for ax in axes.ravel():
        ax.axis('off')
        
    for i in range(int(num_images/2)):
        img = torch.nn.functional.interpolate(dataset[i].unsqueeze(0), scale_factor=2, mode='bilinear')
        axes[0, i].imshow(img.squeeze(0).permute(1, 2, 0))
        img = torch.nn.functional.interpolate(dataset[i+int(num_images/2)].unsqueeze(0), scale_factor=2, mode='bilinear')
        axes[1, i].imshow(img.squeeze(0).permute(1, 2, 0))
        plt.show



def reconstruction_evaluation(model, dataset, num_images=10):
    
    for i in range(num_images):
        img_real = dataset[i]
        img_real_batched = img_real.unsqueeze(0)
        with torch.no_grad():
            img_rec_batched = model(img_real_batched)[1]
        img_rec = img_rec_batched.squeeze(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.imshow(img_real.permute(1, 2, 0))
        ax1.set_title('Real')
        
        ax2.imshow(img_rec.permute(1, 2, 0))
        ax2.set_title('Reconstructed')
        
        ax1.axis('off')
        ax2.axis('off')
        
        plt.show()




# Load image data
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config.IMG_DIR, 
                                                                                    batch_size=config.BATCH_SIZE,
                                                                                    validation_split=config.VALIDATION_SPLIT,
                                                                                    transform=config.TRANSFORM, 
                                                                                    fraction=config.FRACTION)

# Load model
vqvae_loaded = VQVAE.load_model(save_folder=MODEL_FOLDER).to(config.DEVICE)


# Plot dataset
plot_dataset(train_dataset, num_images=10, title='Training dataset')


# Evaluate reconstruction
reconstruction_evaluation(vqvae_loaded, val_dataset, num_images=2)







