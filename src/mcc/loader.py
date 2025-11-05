
def _load_checkpoint(path_or_url: str):
    import torch
    
    if str(path_or_url).startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu", check_hash=True)
    return torch.load(path_or_url, map_location="cpu", weights_only=False)

def _extract_state_dict(ckpt: dict) -> dict:
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    if any(k.startswith("module.") for k in state.keys()):
        # strip "module." prefix
        state = {k[len("module."):]: v for k, v in state.items()}

    return state

def load_model(model_config_path: str,
               model_checkpoint_path: str,
               device: str = None):
    """
    Load MCC model from config and checkpoint.
    
    Args:
        model_config_path (str): Path to the model configuration file.
        model_checkpoint_path (str): Path to the model checkpoint file.
        device (str, optional): Device to load the model onto. Defaults to None.
    Returns: 
        nn.Module: Loaded MCC model.
    """
    import torch
    from mcc.config import MCCConfig, load_config
    from mcc.model import get_mcc_model

    # Load YAML
    cfg: MCCConfig = load_config(model_config_path)
    params = cfg.model

    # Get device
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build model architecture
    print("Building model...")
    model = get_mcc_model(**params.__dict__)

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = _load_checkpoint(model_checkpoint_path)
    state = _extract_state_dict(ckpt)

    print("Loading state dict...")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"MCC load_state_dict\n missing: {missing}\n unexpected: {unexpected}")

    print("Moving model to device...")
    model.to(device).eval()

    # Compile the model 
    #print("Compiling model...")
    #model = torch.compile(model, mode='reduce-overhead', fullgraph=False)

  
    print(f"MCC model loaded on {device}")
    print(f" - config: {model_config_path}")
    print(f" - checkpoint: {model_checkpoint_path}")

    return model