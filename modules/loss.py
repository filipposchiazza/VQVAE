import torch 

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

