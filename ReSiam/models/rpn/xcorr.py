import torch
import torch.nn.functional as F

def xcorr_slow(x, kernel):
    """Use loop over Batch dim to calculate cross correlation (slow)
    """
    
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out

def xcorr_fast(x, kernel):
    """Use group conv2d to calculate cross correlation (fast)
    """
    
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    out = F.conv2d(px, pk, groups=batch)
    out = out.view(batch, -1, out.size()[2], out.size()[3])
    return out

def xcorr_depthwise(x, kernel):
    """Use depthwise cross correlation (similar to DW Conv)
    """
    
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

def xcorr_depthwise_new(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel, padding=int((kernel.size(2)-1)/2))
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
