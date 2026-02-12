"""Helper code to instantiate various models."""

import math
import logging
import torch
import torchvision
import torch.nn as nn

from collections import OrderedDict

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

from .language_models import RNNModel, TransformerModel, LinearModel
from .losses import CausalLoss, MLMLoss, MostlyCausalLoss

log = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


def _model_name(cfg_model):
    if isinstance(cfg_model, str):
        return cfg_model
    name = _cfg_get(cfg_model, "name", None)
    if name is None:
        raise ValueError("Model config must provide a model name.")
    return name


class LoRALinear(nn.Module):
    """LoRA wrapper for a linear layer."""

    def __init__(self, base_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects a torch.nn.Linear layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.base_layer(x)
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + delta * self.scaling

    def shared_parameters(self):
        return [self.lora_A, self.lora_B]


class LoRAQKVLinear(nn.Module):
    """LoRA wrapper for fused qkv linear layers in timm ViT blocks."""

    def __init__(self, base_layer, rank=4, alpha=1.0, dropout=0.0, apply_q=True, apply_k=True, apply_v=True):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRAQKVLinear expects a torch.nn.Linear layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if base_layer.out_features % 3 != 0:
            raise ValueError("Fused qkv layer must have out_features divisible by 3.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.chunk = base_layer.out_features // 3
        self.has_any_adapter = bool(apply_q or apply_k or apply_v)

        def make_pair(enabled):
            if not enabled:
                return None, None
            A = nn.Parameter(torch.zeros(self.rank, base_layer.in_features))
            B = nn.Parameter(torch.zeros(self.chunk, self.rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            return A, B

        self.lora_A_q, self.lora_B_q = make_pair(apply_q)
        self.lora_A_k, self.lora_B_k = make_pair(apply_k)
        self.lora_A_v, self.lora_B_v = make_pair(apply_v)

    def _delta(self, x, A, B):
        if A is None or B is None:
            return None
        return ((self.dropout(x) @ A.t()) @ B.t()) * self.scaling

    def forward(self, x):
        out = self.base_layer(x)
        if not self.has_any_adapter:
            return out
        chunks = list(out.split(self.chunk, dim=-1))
        delta_q = self._delta(x, self.lora_A_q, self.lora_B_q)
        delta_k = self._delta(x, self.lora_A_k, self.lora_B_k)
        delta_v = self._delta(x, self.lora_A_v, self.lora_B_v)
        if delta_q is not None:
            chunks[0] = chunks[0] + delta_q
        if delta_k is not None:
            chunks[1] = chunks[1] + delta_k
        if delta_v is not None:
            chunks[2] = chunks[2] + delta_v
        return torch.cat(chunks, dim=-1)

    def shared_parameters(self):
        params = []
        for p in [self.lora_A_q, self.lora_B_q, self.lora_A_k, self.lora_B_k, self.lora_A_v, self.lora_B_v]:
            if p is not None:
                params.append(p)
        return params


class TimmLoRAAdapter(nn.Module):
    """Wrapper around a ViT model patched with LoRA modules."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def shared_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class TimmVPTAdapter(nn.Module):
    """Visual Prompt Tuning wrapper for timm ViT models."""

    def __init__(
        self,
        base_model,
        num_tokens=5,
        prompt_dropout=0.0,
        prompt_proj_dim=-1,
        deep=False,
        num_deep_layers=None,
        prompt_mlp_layers=0,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_tokens = int(num_tokens)
        self.deep = bool(deep)
        self.num_prefix_tokens = int(getattr(base_model, "num_prefix_tokens", 1))

        for p in self.base_model.parameters():
            p.requires_grad = False

        embed_dim = int(getattr(base_model, "embed_dim", getattr(base_model, "num_features")))
        prompt_dim = int(prompt_proj_dim) if prompt_proj_dim is not None and int(prompt_proj_dim) > 0 else embed_dim

        patch_size = getattr(getattr(base_model, "patch_embed", None), "patch_size", (16, 16))
        if isinstance(patch_size, int):
            patch_area = patch_size * patch_size
        else:
            patch_area = int(patch_size[0]) * int(patch_size[1])
        val = math.sqrt(6.0 / float(3 * patch_area + prompt_dim))

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        self.prompt_proj = nn.Linear(prompt_dim, embed_dim) if prompt_dim != embed_dim else nn.Identity()

        mlp_layers = []
        for _ in range(int(prompt_mlp_layers)):
            mlp_layers.append(nn.Linear(embed_dim, embed_dim))
            mlp_layers.append(nn.GELU())
        self.prompt_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.prompt_dropout = nn.Dropout(float(prompt_dropout)) if prompt_dropout and prompt_dropout > 0 else nn.Identity()

        if self.deep:
            total_layers = len(self.base_model.blocks) - 1
            self.num_deep_layers = int(num_deep_layers) if num_deep_layers is not None else total_layers
            self.num_deep_layers = max(0, min(self.num_deep_layers, total_layers))
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(self.num_deep_layers, self.num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings, -val, val)
        else:
            self.deep_prompt_embeddings = None

    def _transform_prompt(self, prompt_emb):
        prompt = self.prompt_proj(prompt_emb)
        prompt = self.prompt_mlp(prompt)
        prompt = self.prompt_dropout(prompt)
        return prompt

    def _insert_prompts(self, x, prompt_tokens):
        return torch.cat(
            [
                x[:, : self.num_prefix_tokens, :],
                prompt_tokens,
                x[:, self.num_prefix_tokens :, :],
            ],
            dim=1,
        )

    def _strip_prompts(self, x):
        return torch.cat(
            [
                x[:, : self.num_prefix_tokens, :],
                x[:, self.num_prefix_tokens + self.num_tokens :, :],
            ],
            dim=1,
        )

    def _pos_embed(self, x):
        if hasattr(self.base_model, "_pos_embed"):
            return self.base_model._pos_embed(x)

        B = x.shape[0]
        prefix = []
        if hasattr(self.base_model, "cls_token") and self.base_model.cls_token is not None:
            prefix.append(self.base_model.cls_token.expand(B, -1, -1))
        if hasattr(self.base_model, "reg_token") and self.base_model.reg_token is not None:
            prefix.append(self.base_model.reg_token.expand(B, -1, -1))
        if len(prefix) > 0:
            x = torch.cat([*prefix, x], dim=1)
        if hasattr(self.base_model, "pos_embed") and self.base_model.pos_embed is not None:
            if self.base_model.pos_embed.shape[1] == x.shape[1]:
                x = x + self.base_model.pos_embed
        if hasattr(self.base_model, "pos_drop"):
            x = self.base_model.pos_drop(x)
        return x

    def _forward_features(self, x):
        x = self.base_model.patch_embed(x)
        x = self._pos_embed(x)

        if hasattr(self.base_model, "patch_drop"):
            x = self.base_model.patch_drop(x)
        if hasattr(self.base_model, "norm_pre"):
            x = self.base_model.norm_pre(x)

        B = x.shape[0]
        shallow_prompt = self._transform_prompt(self.prompt_embeddings).expand(B, -1, -1)
        x = self._insert_prompts(x, shallow_prompt)

        for idx, block in enumerate(self.base_model.blocks):
            if self.deep_prompt_embeddings is not None and idx > 0 and (idx - 1) < self.deep_prompt_embeddings.shape[0]:
                deep_prompt = self._transform_prompt(self.deep_prompt_embeddings[idx - 1]).expand(B, -1, -1)
                x = self._insert_prompts(self._strip_prompts(x), deep_prompt)
            x = block(x)

        x = self._strip_prompts(x)
        if hasattr(self.base_model, "norm"):
            x = self.base_model.norm(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        if hasattr(self.base_model, "forward_head"):
            return self.base_model.forward_head(x)
        if x.ndim == 3:
            if getattr(self.base_model, "global_pool", "") == "avg":
                x = x[:, self.num_prefix_tokens :, :].mean(dim=1)
            else:
                x = x[:, 0]
        if hasattr(self.base_model, "fc_norm") and self.base_model.fc_norm is not None:
            x = self.base_model.fc_norm(x)
        if hasattr(self.base_model, "head_drop"):
            x = self.base_model.head_drop(x)
        if hasattr(self.base_model, "head"):
            x = self.base_model.head(x)
        return x

    def shared_parameters(self):
        params = [self.prompt_embeddings]
        params += list(self.prompt_proj.parameters()) if isinstance(self.prompt_proj, nn.Module) else []
        params += list(self.prompt_mlp.parameters()) if isinstance(self.prompt_mlp, nn.Module) else []
        if self.deep_prompt_embeddings is not None:
            params.append(self.deep_prompt_embeddings)
        return params


def _is_vit_backbone(model):
    return hasattr(model, "patch_embed") and hasattr(model, "blocks") and hasattr(model, "forward")


def _find_vit_backbone(model):
    if _is_vit_backbone(model):
        return model, None
    if hasattr(model, "model") and _is_vit_backbone(model.model):
        return model.model, model
    return None, None


def _parse_peft_cfg(cfg_model, kwargs):
    peft_cfg = _cfg_get(cfg_model, "peft", None) if not isinstance(cfg_model, str) else None
    if peft_cfg is None and "cfg_case" in kwargs:
        peft_cfg = _cfg_get(kwargs["cfg_case"], "peft", None)
    return peft_cfg


def _replace_linear_with_lora(module_parent, attr_name, rank, alpha, dropout):
    old_layer = getattr(module_parent, attr_name, None)
    if isinstance(old_layer, nn.Linear):
        setattr(module_parent, attr_name, LoRALinear(old_layer, rank=rank, alpha=alpha, dropout=dropout))
        return 1
    return 0


def _apply_lora_to_vit(backbone, peft_cfg):
    lora_cfg = _cfg_get(peft_cfg, "lora", peft_cfg)
    rank = int(_cfg_get(lora_cfg, "rank", 4))
    alpha = float(_cfg_get(lora_cfg, "alpha", rank))
    dropout = float(_cfg_get(lora_cfg, "dropout", 0.0))
    targets = _cfg_get(lora_cfg, "target_modules", _cfg_get(lora_cfg, "targets", ["q", "k", "v"]))
    if isinstance(targets, str):
        targets = [target.strip() for target in targets.split(",")]
    target_set = {target.lower() for target in targets}

    for p in backbone.parameters():
        p.requires_grad = False

    adapters = 0
    for block in backbone.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                qkv_layer = LoRAQKVLinear(
                    attn.qkv,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    apply_q=("q" in target_set),
                    apply_k=("k" in target_set),
                    apply_v=("v" in target_set),
                )
                if qkv_layer.has_any_adapter:
                    attn.qkv = qkv_layer
                    adapters += 1
            else:
                if "q" in target_set:
                    adapters += _replace_linear_with_lora(attn, "q_proj", rank, alpha, dropout)
                if "k" in target_set:
                    adapters += _replace_linear_with_lora(attn, "k_proj", rank, alpha, dropout)
                if "v" in target_set:
                    adapters += _replace_linear_with_lora(attn, "v_proj", rank, alpha, dropout)

            if "proj" in target_set:
                adapters += _replace_linear_with_lora(attn, "proj", rank, alpha, dropout)

        if "mlp" in target_set:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                adapters += _replace_linear_with_lora(mlp, "fc1", rank, alpha, dropout)
                adapters += _replace_linear_with_lora(mlp, "fc2", rank, alpha, dropout)

    if adapters == 0:
        log.warning("LoRA PEFT requested but no target layers were patched.")
    return TimmLoRAAdapter(backbone)


def _apply_vpt_to_vit(backbone, peft_cfg):
    vpt_cfg = _cfg_get(peft_cfg, "vpt", peft_cfg)
    return TimmVPTAdapter(
        backbone,
        num_tokens=int(_cfg_get(vpt_cfg, "num_tokens", 5)),
        prompt_dropout=float(_cfg_get(vpt_cfg, "dropout", 0.0)),
        prompt_proj_dim=int(_cfg_get(vpt_cfg, "project", -1)),
        deep=bool(_cfg_get(vpt_cfg, "deep", False)),
        num_deep_layers=_cfg_get(vpt_cfg, "num_deep_layers", None),
        prompt_mlp_layers=int(_cfg_get(vpt_cfg, "prompt_mlp_layers", _cfg_get(vpt_cfg, "prompt_mlp_num", 0))),
    )


def _adapt_model_with_peft(model, cfg_model, modality="vision", **kwargs):
    if modality != "vision":
        return model
    model_name = _model_name(cfg_model).lower()
    if "vit" not in model_name:
        return model

    peft_cfg = _parse_peft_cfg(cfg_model, kwargs)
    if peft_cfg is None or not bool(_cfg_get(peft_cfg, "enabled", True)):
        return model

    peft_type = _cfg_get(peft_cfg, "type", None)
    if peft_type is None:
        if _cfg_get(peft_cfg, "vpt", None) is not None:
            peft_type = "vpt"
        elif _cfg_get(peft_cfg, "lora", None) is not None:
            peft_type = "lora"
    if peft_type is None:
        raise ValueError("PEFT config requires `type` to be one of `vpt` or `lora`.")

    backbone, parent = _find_vit_backbone(model)
    if backbone is None:
        log.warning("PEFT requested for %s but no compatible ViT backbone was found.", model_name)
        return model

    peft_type = str(peft_type).lower()
    log.info("Applying %s PEFT adaptation to ViT model %s.", peft_type, model_name)
    if peft_type == "vpt":
        adapted_backbone = _apply_vpt_to_vit(backbone, peft_cfg)
    elif peft_type == "lora":
        adapted_backbone = _apply_lora_to_vit(backbone, peft_cfg)
    else:
        raise ValueError(f"Unknown PEFT type {peft_type}.")

    if parent is None:
        return adapted_backbone
    parent.model = adapted_backbone
    return model


def construct_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    if cfg_data.modality == "vision":
        model = _construct_vision_model(cfg_model, cfg_data, pretrained, **kwargs)
    elif cfg_data.modality == "text":
        model = _construct_text_model(cfg_model, cfg_data, pretrained, **kwargs)
    else:
        raise ValueError(f"Invalid data modality {cfg_data.modality}")
    # Save nametag for printouts later:
    model.name = _model_name(cfg_model)

    # Choose loss function according to data and model:
    if "classification" in cfg_data.task:
        loss_fn = torch.nn.CrossEntropyLoss()
    elif "causal-lm-sanity" in cfg_data.task:
        loss_fn = MostlyCausalLoss()
    elif "causal-lm" in cfg_data.task:
        loss_fn = CausalLoss()
    elif "masked-lm" in cfg_data.task:
        loss_fn = MLMLoss(vocab_size=cfg_data.vocab_size)
    else:
        raise ValueError(f"No loss function registered for task {cfg_data.task}.")
    loss_fn = torch.jit.script(loss_fn)
    return model, loss_fn


def _construct_text_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    raw_cfg_model = cfg_model
    cfg_model = _model_name(cfg_model)
    if cfg_model == "transformer3f":
        # This is the transformer from "A field guide to federated learning"
        """
        we train a modified 3-layer Transformer model [250],
        where the dimension of the token embeddings is 96, and the hidden dimension of the feed-forward
        network (FFN) block is 1536. We use 8 heads for the multi-head attention, where each head is based
        on 12-dimensional (query, key, value) vectors. We use ReLU activation and set dropout rate to 0.1.
        """
        # For simplicity the dropout is disabled for now
        # the 12-dim query is 96/8 = 12
        model = TransformerModel(
            ntokens=cfg_data.vocab_size, ninp=96, nhead=8, nhid=1536, nlayers=3, dropout=0, positional_embedding="fixed"
        )
    elif cfg_model == "transformer3":
        # Same as above but with learnable positional embeddings
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
        )
    elif cfg_model == "transformer3t":
        # Same as above but with learnable positional embeddings and tied weights
        model = TransformerModel(
            ntokens=cfg_data.vocab_size,
            ninp=96,
            nhead=8,
            nhid=1536,
            nlayers=3,
            dropout=0,
            positional_embedding="learnable",
            tie_weights=True,
        )
    elif cfg_model == "transformer1":
        # This is our initial sanity check transformer:
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=200, nhead=1, nhid=200, nlayers=1, dropout=0)
    elif cfg_model == "transformerS":
        # A wide sanity-check transformer
        model = TransformerModel(ntokens=cfg_data.vocab_size, ninp=512, nhead=1, nhid=512, nlayers=1, dropout=0)
    elif cfg_model == "LSTM":
        # This is the LSTM from "LEARNING DIFFERENTIALLY PRIVATE RECURRENT LANGUAGE MODELS"
        """
        word s t is mapped to an embedding vector e t ∈ R 96
        by looking up the word in the model’s vocabulary. The e t is composed with the state emitted by
        the model in the previous time step s t−1 ∈ R 256 to emit a new state vector s t and an “output
        embedding” o t ∈ R 96 .
        """
        model = RNNModel("LSTM", cfg_data.vocab_size, ninp=96, nhid=96, nlayers=1, dropout=0.0, tie_weights=True)
    elif cfg_model == "linear":
        model = LinearModel(cfg_data.vocab_size, embedding_size=200)
    else:
        try:
            from transformers import (
                AutoModelForMaskedLM,
                AutoModelForPreTraining,
                AutoModelForSequenceClassification,
                AutoConfig,
            )

            if cfg_data.task == "masked-lm":
                auto_class = AutoModelForMaskedLM
            elif cfg_data.task == "classification":
                auto_class = AutoModelForSequenceClassification
            else:
                auto_class = AutoModelForPreTraining
            # Make sure to use the matching tokenizer and vocab_size!
            if cfg_model == "gpt2S":
                cfg_model = "gpt2"
                extra_args = dict(activation_function="relu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
            elif cfg_model == "bert-sanity-check":
                cfg_model = "bert-base-uncased"
                extra_args = dict(hidden_act="relu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            else:
                extra_args = dict()
            if pretrained:
                model = auto_class.from_pretrained(cfg_model, **extra_args)
            else:
                hf_cfg = AutoConfig.from_pretrained(cfg_model, **extra_args)
                model = auto_class.from_config(hf_cfg)
            # model.transformer.h[0].attn.scale_attn_weights = False
            if model.config.vocab_size != cfg_data.vocab_size:
                model.resize_token_embeddings(new_num_tokens=cfg_data.vocab_size)
            model = HuggingFaceContainer(model)
        except OSError as error_msg:
            raise ValueError(f"Invalid huggingface model {cfg_model} given: {error_msg}")
    return _adapt_model_with_peft(model, raw_cfg_model, modality="text", **kwargs)


class HuggingFaceContainer(torch.nn.Module):
    """Wrap huggingface models for a unified interface. Ugh."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if "inputs" in kwargs:
            kwargs["input_ids"] = kwargs.pop("inputs")
        if "input_ids" not in kwargs:
            kwargs["input_ids"] = args[0]
        if kwargs["input_ids"].dtype != torch.long:
            kwargs["inputs_embeds"] = kwargs.pop("input_ids")
        outputs = self.model(**kwargs)
        return outputs["logits"] if "logits" in outputs else outputs["prediction_logits"]

    def shared_parameters(self):
        return self.model.parameters()


class VisionContainer(torch.nn.Module):
    """We'll use a container to catch extra attributes and allow for usage with model(**data)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, **kwargs):
        return self.model(inputs)

    def shared_parameters(self):
        if hasattr(self.model, "shared_parameters"):
            return self.model.shared_parameters()
        return self.model.parameters()


def _construct_vision_model(cfg_model, cfg_data, pretrained=True, **kwargs):
    """Construct the neural net that is used."""
    raw_cfg_model = cfg_model
    cfg_model = _model_name(cfg_model)
    channels = cfg_data.shape[0]
    classes = cfg_data.classes

    if "ImageNet" in cfg_data.name:
        try:
            model = getattr(torchvision.models, cfg_model.lower())(pretrained=pretrained)
            try:
                # Try to adjust the linear layer and fill with previous data
                fc = torch.nn.Linear(model.fc.in_features, classes)
                if pretrained:
                    fc.weight.data = model.fc.weight[:classes]
                    fc.bias.data = model.fc.bias[:classes]
                model.fc = fc
            except AttributeError:
                pass
        except AttributeError:
            if "nfnet" in cfg_model:
                model = NFNet(
                    channels,
                    classes,
                    variant="F0",
                    stochdepth_rate=0.25,
                    alpha=0.2,
                    se_ratio=0.5,
                    activation="ReLU",
                    stem="ImageNet",
                    use_dropout=True,
                )
            elif "resnet101wsl" in cfg_model:
                model = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
            elif "resnet50swsl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl")
            elif "resnet50ssl" in cfg_model:
                model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl")
            elif "resnetmoco" in cfg_model:
                model = torchvision.models.resnet50(pretrained=False)
                if pretrained:
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
                    # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
                    url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar"
                    state_dict = torch.hub.load_state_dict_from_url(
                        url, progress=True, map_location=torch.device("cpu")
                    )["state_dict"]
                    for key in list(state_dict.keys()):
                        val = state_dict.pop(key)
                        # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                        sanitized_key = key.replace("module.", "")
                        state_dict[sanitized_key] = val

                    model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
            elif "vit_base_april" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_small_april" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)
                model.blocks[0] = ModifiedBlock(model.blocks[0])
            elif "vit_base" in cfg_model:
                import timm  # lazily import

                # timm models are listed at https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
                model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
            elif "vit_small" in cfg_model:
                import timm

                model = timm.create_model("vit_small_patch16_224", pretrained=pretrained)

            elif "linear" == cfg_model:
                input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
            elif "none" == cfg_model:
                model = torch.nn.Sequential(torch.nn.Flatten(), _Select(classes))
            else:
                raise ValueError(f"Could not find ImageNet model {cfg_model} in torchvision.models or custom models.")
    else:
        # CIFAR Model from here:
        if "resnetgn" in cfg_model.lower():
            block, layers = resnet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="groupnorm4th",
                downsample="B",
                width_per_group=16 if len(layers) < 4 else 64,
                zero_init_residual=False,
            )
        elif "resnet" in cfg_model.lower():
            if "-" in cfg_model.lower():  # Hacky way to separate ResNets from wide ResNets which are e.g. 28-10
                depth = int("".join(filter(str.isdigit, cfg_model.split("-")[0])))
                width = int("".join(filter(str.isdigit, cfg_model.split("-")[1])))
            else:
                depth = int("".join(filter(str.isdigit, cfg_model)))
                width = 1
            block, layers = resnet_depths_to_config(depth)
            model = ResNet(
                block,
                layers,
                channels,
                classes,
                stem="CIFAR",
                convolution_type="Standard",
                nonlin="ReLU",
                norm="BatchNorm2d",
                downsample="B",
                width_per_group=(16 if len(layers) < 4 else 64) * width,
                zero_init_residual=False,
            )
        elif "densenet" in cfg_model.lower():
            growth_rate, block_config, num_init_features = densenet_depths_to_config(
                int("".join(filter(str.isdigit, cfg_model)))
            )
            model = DenseNet(
                growth_rate=growth_rate,
                block_config=block_config,
                num_init_features=num_init_features,
                bn_size=4,
                drop_rate=0,
                channels=channels,
                num_classes=classes,
                memory_efficient=False,
                norm="BatchNorm2d",
                nonlin="ReLU",
                stem="CIFAR",
                convolution_type="Standard",
            )
        elif "vgg" in cfg_model.lower():
            model = VGG(
                cfg_model,
                in_channels=channels,
                num_classes=classes,
                norm="BatchNorm2d",
                nonlin="ReLU",
                head="CIFAR",
                convolution_type="Standard",
                drop_rate=0,
                classical_weight_init=True,
            )
        elif "linear" in cfg_model:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
        elif "nfnet" in cfg_model:
            model = NFNet(
                channels,
                classes,
                variant="F0",
                stochdepth_rate=0.25,
                alpha=0.2,
                se_ratio=0.5,
                activation="ReLU",
                stem="CIFAR",
                use_dropout=True,
            )
        elif "convnet-trivial" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv", torch.nn.Conv2d(channels, 3072, 3, stride=1, padding=1)),
                        ("relu", torch.nn.ReLU(inplace=True)),
                        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
                        ("flatten", torch.nn.Flatten()),
                        ("linear", torch.nn.Linear(3072, classes)),
                    ]
                )
            )
        elif "convnetsmall" == cfg_model.lower():
            model = ConvNetSmall(width=256, num_channels=channels, num_classes=classes)
        elif "convnet" == cfg_model.lower():
            model = ConvNet(width=64, num_channels=channels, num_classes=classes)
        elif "convnet_beyond" == cfg_model.lower():
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", torch.nn.Conv2d(channels, 32, 3, stride=2, padding=1)),
                        ("relu0", torch.nn.LeakyReLU()),
                        ("conv2", torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                        ("relu1", torch.nn.LeakyReLU()),
                        ("conv3", torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                        ("relu2", torch.nn.LeakyReLU()),
                        ("conv4", torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                        ("relu3", torch.nn.LeakyReLU()),
                        ("flatt", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(12544, 12544)),
                        ("relu4", torch.nn.LeakyReLU()),
                        ("linear1", torch.nn.Linear(12544, classes)),
                        ("softmax", torch.nn.Softmax(dim=1)),
                    ]
                )
            )
        elif "lenet_zhu" == cfg_model.lower():
            model = LeNetZhu(num_channels=channels, num_classes=classes)
        elif "cnn6" == cfg_model.lower():
            # This is the model from R-GAP:
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("layer0", torch.nn.Conv2d(channels, 12, kernel_size=4, padding=2, stride=2, bias=False)),
                        ("act0", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer1", torch.nn.Conv2d(12, 36, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act1", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer2", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act2", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer3", torch.nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("act3", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer4", torch.nn.Conv2d(36, 64, kernel_size=3, padding=1, stride=2, bias=False)),
                        ("act4", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("layer5", torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False)),
                        ("flatten", torch.nn.Flatten()),
                        ("act5", torch.nn.LeakyReLU(negative_slope=0.2)),
                        ("fc", torch.nn.Linear(3200, classes, bias=True)),
                    ]
                )
            )
        elif cfg_model == "MLP":
            width = 1024
            model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", torch.nn.Flatten()),
                        ("linear0", torch.nn.Linear(3072, width)),
                        ("relu0", torch.nn.ReLU()),
                        ("linear1", torch.nn.Linear(width, width)),
                        ("relu1", torch.nn.ReLU()),
                        ("linear2", torch.nn.Linear(width, width)),
                        ("relu2", torch.nn.ReLU()),
                        ("linear3", torch.nn.Linear(width, classes)),
                    ]
                )
            )
        elif "vit_small" in cfg_model:
            import timm
            img_size = (cfg_data.shape[1], cfg_data.shape[2])
            in_chans = cfg_data.shape[0]
            if "MNIST" in cfg_data.name:
                p = 14
            else:
                p = 16
            model = timm.create_model("vit_small_patch16_224", pretrained=False if p==14 else pretrained, 
                                        img_size=img_size, 
                                        in_chans=in_chans, 
                                        num_classes=classes,
                                        patch_size=p)
            if "april" in cfg_model:
                model.model.blocks[0] = ModifiedBlock(model.model.blocks[0])
        elif "custom" in cfg_model:
            # Provision for custom ViT or other models
            if "vit" in cfg_model:
                img_size = (cfg_data.shape[1], cfg_data.shape[2])
                in_chans = cfg_data.shape[0]
                if "MNIST" in cfg_data.name:
                    p = 14
                else:
                    p = 16
                model = SmallViT(img_size=img_size, in_chans=in_chans, num_classes=classes, 
                             patch_size=p, embed_dim=384, depth=4, num_heads=3)
                if "april" in cfg_model:
                    model.model.blocks[0] = ModifiedBlock(model.model.blocks[0])
            else:
                raise ValueError(f"Custom model type not recognized in {cfg_model}")
        else:
            raise ValueError("Model could not be found.")

    model = _adapt_model_with_peft(model, raw_cfg_model, modality="vision", **kwargs)
    return VisionContainer(model)


class ConvNetSmall(torch.nn.Module):
    """ConvNet without BN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu2", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv3", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, stride=2, padding=1)),
                    ("relu3", torch.nn.ReLU()),
                    ("pool1", torch.nn.AdaptiveAvgPool2d(1)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(4 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                    ("bn0", torch.nn.BatchNorm2d(1 * width)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn1", torch.nn.BatchNorm2d(2 * width)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                    ("bn2", torch.nn.BatchNorm2d(2 * width)),
                    ("relu2", torch.nn.ReLU()),
                    ("conv3", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn3", torch.nn.BatchNorm2d(4 * width)),
                    ("relu3", torch.nn.ReLU()),
                    ("conv4", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn4", torch.nn.BatchNorm2d(4 * width)),
                    ("relu4", torch.nn.ReLU()),
                    ("conv5", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn5", torch.nn.BatchNorm2d(4 * width)),
                    ("relu5", torch.nn.ReLU()),
                    ("pool0", torch.nn.MaxPool2d(3)),
                    ("conv6", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn6", torch.nn.BatchNorm2d(4 * width)),
                    ("relu6", torch.nn.ReLU()),
                    ("conv7", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                    ("bn7", torch.nn.BatchNorm2d(4 * width)),
                    ("relu7", torch.nn.ReLU()),
                    ("pool1", torch.nn.MaxPool2d(3)),
                    ("flatten", torch.nn.Flatten()),
                    ("linear", torch.nn.Linear(36 * width, num_classes)),
                ]
            )
        )

    def forward(self, input):
        return self.model(input)


class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(768, num_classes))
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class ModifiedBlock(torch.nn.Module):
    def __init__(self, old_Block):
        super().__init__()
        self.attn = old_Block.attn
        self.drop_path = old_Block.drop_path1
        self.norm2 = old_Block.norm2
        self.mlp = old_Block.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.drop_path(self.mlp((self.norm2(x))))
        return x


class SmallViT(torch.nn.Module):
    """A small ViT variant for low-resolution images like MNIST/CIFAR."""

    def __init__(self, img_size=(32, 32), patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        import timm.models.vision_transformer
        self.model = timm.models.vision_transformer.VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
        )

    def forward(self, x):
        return self.model(x)

    def shared_parameters(self):
        if hasattr(self.model, "shared_parameters"):
            return self.model.shared_parameters()
        return self.model.parameters()
