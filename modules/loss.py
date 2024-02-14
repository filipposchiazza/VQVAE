import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.nn as nn
import torch.nn.functional as F




def sqrt_mae(output, target):
    """Square root of the mean absolute error
    
    Parameters
    ----------
    output : torch.Tensor
        Output tensor.
    target : torch.Tensor
        Target tensor.
    
    Returns
    -------
    loss : torch.Tensor
        Square root of the mean absolute error.
    """
    loss =  torch.mean(torch.sqrt(torch.abs(output - target)))
    return loss



class Variance_weighted_loss(nn.Module):

    def __init__(self, local_connectivity=3, loss_fn=F.l1_loss, device='cpu'):
        """Variance weighted loss function.
        
        Parameters
        ----------
        local_connectivity : int, optional
            Size of the local connectivity. The default is 3.
        loss_fn : function, optional
            Loss function. The default is F.l1_loss.
        device : str, optional
            Device. The default is 'cpu'.
        """
        super(Variance_weighted_loss, self).__init__()
        self.local_connectivity = local_connectivity
        self.loss_fn = loss_fn
        self.device = device
        kernel_shape = (1, 1, self.local_connectivity, self.local_connectivity)
        self.kernel = (torch.ones(kernel_shape, requires_grad=False) / (self.local_connectivity ** 2)).to(self.device)



    def forward(self, output, target, epoch):
        # Compute the usual loss (without weighting by variance)
        loss = self.loss_fn(output, target, reduction='none')

        # Warmup according to the epoch
        if epoch > 2:
            # From RGB to GrayScale
            target_gray = torch.mean(target, dim=1, keepdim=True)

            # Compute local mean
            target_mean = F.conv2d(target_gray, self.kernel, padding=self.local_connectivity // 2)

            # Compute local variance
            target_variance = F.conv2d(target_gray ** 2, self.kernel, padding=self.local_connectivity // 2) - target_mean ** 2
            target_variance = (target_variance - target_variance.min()) / (target_variance.max() - target_variance.min()) + 0.5
            #target_variance = torch.clip(target_variance, max=1.0) * 10

            # Weight the loss by the variance
            loss *= target_variance

        # Reduce the loss to a scalar
        loss = torch.mean(loss)
        return loss
    




