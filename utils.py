from collections import defaultdict


def get_target_module(model, module, layer_idx):
    """Get the target module corresponding to the specified module name and layer index.

    Args:
        model (torch.nn.Module): target model
        module (str): module name (e.g., 'mlp.up_proj', 'self_attn.q_proj', etc.)
        layer_idx (int): layer index

    Returns:
        torch.nn.Module: target module
    """
    text_model = get_text_model(model)
    if model.name_or_path in ['google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'Qwen/Qwen3-14B', 'meta-llama/Llama-3.1-8B-Instruct', 'tiiuae/Falcon3-10B-Instruct']:
        if module == 'mlp.up_proj':
            return text_model.layers[layer_idx].mlp.up_proj
        elif module == 'mlp.gate_proj':
            return text_model.layers[layer_idx].mlp.gate_proj
        elif module == 'mlp.down_proj':
            return text_model.layers[layer_idx].mlp.down_proj
        elif module == 'self_attn.q_proj':
            return text_model.layers[layer_idx].self_attn.q_proj
        elif module == 'self_attn.k_proj':
            return text_model.layers[layer_idx].self_attn.k_proj
        elif module == 'self_attn.v_proj':
            return text_model.layers[layer_idx].self_attn.v_proj
        elif module == 'self_attn.o_proj':
            return text_model.layers[layer_idx].self_attn.o_proj
        else:
            raise ValueError(f"Unknown module: {module}. Supported modules are 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'.")
    elif model.name_or_path in ['microsoft/phi-4']:
        if module == 'mlp.up_proj' or module == 'mlp.gate_proj':
            return text_model.layers[layer_idx].mlp.gate_up_proj
        elif module == 'mlp.down_proj':
            return text_model.layers[layer_idx].mlp.down_proj
        elif module == 'self_attn.q_proj' or module == 'self_attn.k_proj' or module == 'self_attn.v_proj':
            return text_model.layers[layer_idx].self_attn.qkv_proj
        elif module == 'self_attn.o_proj':
            return text_model.layers[layer_idx].self_attn.o_proj
        else:
            raise ValueError(f"Unknown module: {module}. Supported modules are 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'.")
    else:
        raise ValueError(f"Unsupported model type: {model.name_or_path}")
            


def manipulate_neuron(module, input_tensor, output_tensor, neuron_idx, operation='suppress'):
    """hook function to manipulate specified neurons during the forward pass.

    Args:
        module (torch.nn.Module): target module
        input_tensor (torch.Tensor): input tensor
        output_tensor (torch.Tensor): output tensor
        neuron_idx (List[int]): indices of the neurons to manipulate
        operation (str): type of operation ('suppress' or 'enhance')

    Returns:
        torch.Tensor: output tensor after manipulation
    """
    modified_output = output_tensor.clone()
    if operation == 'suppress':
        # set the specified activations to zero
        modified_output[:, :, neuron_idx] = 0.0
    elif operation == 'enhance':
        # double the specified activations
        modified_output[:, :, neuron_idx] *= 2.0
    else:
        raise ValueError("Invalid operation. Use 'suppress' or 'enhance'.")
    return modified_output


def register_hook(model, neurons, operation='suppress'):
    """Register hooks to manipulate specified neurons during the forward pass.

    Args:
        model (torch.nn.Module): target model
        neurons (List[Dict]): information about the neurons to manipulate
        operation (str): type of operation ('suppress' or 'enhance')
    """
    indices_module = defaultdict(list)  # (module_name, layer_idx) -> List[neuron_idx]
    for neuron in neurons:
        module_name = neuron['module_name']
        layer_idx = neuron['layer_idx']
        neuron_idx = neuron['neuron_idx']
        indices_module[(module_name, layer_idx)].append(neuron_idx)

    hooks = []
    for (module_name, layer_idx), neuron_indices in indices_module.items():
        neuron_indices = sorted(neuron_indices)
        if len(neuron_indices) == 0:
            continue
        target_module = get_target_module(model, module_name, layer_idx)
        if model.name_or_path in ['google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'Qwen/Qwen3-14B', 'meta-llama/Llama-3.1-8B-Instruct', 'tiiuae/Falcon3-10B-Instruct']:
            hook = target_module.register_forward_hook(
                lambda module, input_tensor, output_tensor, 
                    # to capture loop variable, use default argument
                    n_indices=neuron_indices: manipulate_neuron(
                    module, input_tensor, output_tensor, n_indices, operation
                )
            )
            hooks.append(hook)
        elif model.name_or_path in ['microsoft/phi-4']:
            text_model = get_text_model(model)
            config = text_model.layers[layer_idx].config
            intermed_size = config.intermediate_size
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            key_size = config.num_key_value_heads * head_dim
            query_size = config.num_attention_heads * head_dim
            if module_name == 'mlp.up_proj':
                # adjust neuron_indices for mlp.gate_up_proj
                adjusted_indices = [idx + intermed_size for idx in neuron_indices]
                hook = target_module.register_forward_hook(
                    lambda module, input_tensor, output_tensor,
                    n_indices=adjusted_indices: manipulate_neuron(
                        module, input_tensor, output_tensor, n_indices, operation
                    )
                )
            elif module_name == 'self_attn.k_proj':
                # adjust neuron_indices for self_attn.qkv_proj
                adjusted_indices = [idx + query_size for idx in neuron_indices]
                hook = target_module.register_forward_hook(
                    lambda module, input_tensor, output_tensor,
                    n_indices=adjusted_indices: manipulate_neuron(
                        module, input_tensor, output_tensor, n_indices, operation
                    )
                )
            elif module_name == 'self_attn.v_proj':
                # adjust neuron_indices for self_attn.qkv_proj
                adjusted_indices = [idx + query_size + key_size for idx in neuron_indices]
                hook = target_module.register_forward_hook(
                    lambda module, input_tensor, output_tensor,
                    n_indices=adjusted_indices: manipulate_neuron(
                        module, input_tensor, output_tensor, n_indices, operation
                    )
                )
            else:
                hook = target_module.register_forward_hook(
                    lambda module, input_tensor, output_tensor,
                    n_indices=neuron_indices: manipulate_neuron(
                        module, input_tensor, output_tensor, n_indices, operation
                    )
                )
            hooks.append(hook)
        else:
            raise ValueError(f"Unsupported model type: {model.name_or_path}")

    return hooks


def get_text_model(model):
    if 'Qwen3' in model.name_or_path or 'Llama-3' in model.name_or_path or 'phi-4' in model.name_or_path or 'Falcon3' in model.name_or_path:
        text_model = model.model
    elif 'gemma-3' in model.name_or_path:
        text_model = model.language_model.model
    else:
        raise ValueError(f"Unsupported model type: {model.name_or_path}")

    return text_model
