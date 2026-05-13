import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_language_model(name="distilbert-base-uncased", dtype=torch.float16, **kwargs):
    """
    name: bert-base-uncased, roberta-base
    """
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(name, torch_dtype=dtype, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name)
    def encode_fn(text: str, device='cuda'):
        model.to(device)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # use CLS token
        embeddings = outputs.last_hidden_state.sum(1).squeeze().to(torch.float32).cpu().numpy()
        return embeddings

    return encode_fn
