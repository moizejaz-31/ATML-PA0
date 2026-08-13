import types
import torch
import torch.nn as nn

class ResidualAblator:
    """
    Disables skip connections (+ identity) in specified Bottleneck blocks.
    """
    def __init__(self, model):
        self.model = model
        self.original_forwards = {}

    def disable_skip_connections(self, block_names):
        """
        block_names: list of tuples (layer_name, block_index), e.g., [('layer1', 0), ('layer2', 0)]
        """
        for layer_name, block_idx in block_names:
            layer = getattr(self.model, layer_name)
            block = layer[block_idx]
            
            key = f"{layer_name}_{block_idx}"
            if key not in self.original_forwards:
                self.original_forwards[key] = block.forward

            # Define new forward without identity addition
            def ablated_forward(self_block, x):
                identity = x
                out = self_block.conv1(x)
                out = self_block.bn1(out)
                out = self_block.relu(out)

                out = self_block.conv2(out)
                out = self_block.bn2(out)
                out = self_block.relu(out)

                out = self_block.conv3(out)
                out = self_block.bn3(out)

                # Skip connection identity shortcut removed!
                out = self_block.relu(out)
                return out

            block.forward = types.MethodType(ablated_forward, block)
            print(f"Disabled skip connection in {key}")

    def restore_skip_connections(self):
        for key, original_forward in self.original_forwards.items():
            layer_name, block_idx_str = key.split('_')
            block_idx = int(block_idx_str)
            layer = getattr(self.model, layer_name)
            layer[block_idx].forward = original_forward
            print(f"Restored skip connection in {key}")
        self.original_forwards.clear()
