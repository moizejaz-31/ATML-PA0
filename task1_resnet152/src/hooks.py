import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class ActivationHookManager:
    """
    Manages PyTorch forward hooks to capture activations across early, mid, and late layers.
    Applies spatial global average pooling to 4D feature maps for clean 2D representation.
    """
    def __init__(self, model, layer_names=['layer1', 'layer2', 'layer4']):
        self.model = model
        self.layer_names = layer_names
        self.hooks = []
        self.activations = {}
        self._register_hooks()

    def _get_hook(self, name):
        def hook(module, input, output):
            # If 4D spatial feature map (B, C, H, W), apply global average pooling
            if output.dim() == 4:
                pooled = torch.mean(output, dim=[2, 3])
                self.activations[name] = pooled.detach().cpu()
            else:
                self.activations[name] = output.detach().cpu().view(output.size(0), -1)
        return hook

    def _register_hooks(self):
        for name in self.layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                h = layer.register_forward_hook(self._get_hook(name))
                self.hooks.append(h)

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

def extract_layer_activations(model, loader, layer_names, num_samples=1000, device="cpu"):
    """
    Extracts layer activations across specified layer_names for up to num_samples from loader.
    Applies spatial global average pooling to 4D tensors to return (N, D) matrices per layer.
    """
    model.eval()
    model.to(device)

    collected_activations = {name: [] for name in layer_names}
    collected_labels = []

    def get_hook(name):
        def hook(module, input, output):
            if output.dim() == 4:
                pooled = torch.mean(output, dim=[2, 3])
                collected_activations[name].append(pooled.detach().cpu())
            else:
                collected_activations[name].append(output.detach().cpu().view(output.size(0), -1))
        return hook

    hooks = []
    for name in layer_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            hooks.append(layer.register_forward_hook(get_hook(name)))

    total_collected = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if total_collected >= num_samples:
                break
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            if inputs.shape[-2:] != (224, 224):
                inputs = torch.nn.functional.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            
            model(inputs)
            collected_labels.append(targets[:min(batch_size, num_samples - total_collected)].cpu())
            total_collected += batch_size

    for h in hooks:
        h.remove()

    final_activations = {
        name: torch.cat(collected_activations[name], dim=0)[:num_samples]
        for name in layer_names
    }
    final_labels = torch.cat(collected_labels, dim=0)[:num_samples]

    return final_activations, final_labels

