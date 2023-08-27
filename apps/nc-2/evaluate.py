import torch
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for images, true_masks in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            # move images and labels to correct device and type
            images = images.to(device)
            true_masks = true_masks.to(device)

            # predict the mask
            masks_pred = net(images)

            loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(masks_pred[:, :, 1], true_masks.view(images.shape[0],-1))))

    net.train()
    return loss
