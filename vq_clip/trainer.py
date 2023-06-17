import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from transformers import (
    CLIPConfig,
    CLIPModel,
    AutoTokenizer,
    AutoProcessor,
    CLIPProcessor,
)
from transformers.models.clip.modeling_clip import clip_loss

from .modeling_vq_clip import VQCLIPConfig, VQCLIPModel, VQCLIPOutput
from .modeling_vq_adapter import VQAdapterModel, VQAdapterConfig
from .cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from .eval import zero_shot_eval


def clip_loss_from_embeds(text_embeds, image_embeds, logit_scale):
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    return clip_loss(logits_per_text)


class LightningVQCLIPTrainer(pl.LightningModule):
    """
    Trainer for a VQ adapter on top of a CLIP vision tower
    """

    def __init__(
        self,
        vq_clip_config_path: str = "./model_conf/vq-ViT-L-14/config.json",
        # pretrained clip args
        pretrained_clip_url: str = "openai/clip-vit-base-patch32",
        # training_specific args
        warmup_steps: int = 100,
        max_lr: float = 8e-4,
        min_lr: float = 5e-5,
        lr_gamma: float = 0.4,
        lr_cycle_steps: int = 500,
        torch_compile: bool = False,
        # eval
        imagenet_path: str = "",
        validation_batch_size: int = 512,
    ):
        super().__init__()
        self.vq_clip_config_path = vq_clip_config_path
        self.vq_clip_config = VQCLIPConfig.from_pretrained(vq_clip_config_path)

        self.clip_url = pretrained_clip_url
        self.imagenet_path = imagenet_path
        self.validation_batch_size = validation_batch_size

        self.vision_vq_adapter = VQAdapterModel(
            VQAdapterConfig.from_dict(self.vq_clip_config.vision_vq_adapter_config_dict)
        )

        if torch_compile:
            self.vision_vq_adapter = torch.compile(self.vision_vq_adapter)

        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_gamma = lr_gamma
        self.lr_cycle_steps = lr_cycle_steps

    def __get_vq_clip_model(self) -> VQCLIPModel:
        vq_clip = VQCLIPModel(self.vq_clip_config)
        vq_clip.vision_vq_adapter = self.vision_vq_adapter
        vq_clip.clip_model = CLIPModel.from_pretrained(self.clip_url)
        return vq_clip.to(self.device)

    def on_save_checkpoint(self, _):
        self.save_hf(self.logger.log_dir + "/hf/")

    def save_hf(self, path: str = ""):
        vq_clip = self.__get_vq_clip_model()
        vq_clip.save_adapter(save_directory=path)
        print("Saved HF format", path)

    def step(self, img_emb, text_emb):
        """
        img_emb normalized image embedding tensor batch from CLIP
        text_emb normalized text embedding tensor batch from CLIP
        """
        with torch.no_grad():
            # TODO Assumes logit_scale.exp() = 100.
            pre_quant_contrastive_loss = clip_loss_from_embeds(img_emb, text_emb, 100.0)

        res = self.vision_vq_adapter(img_emb, return_perplexity=True)
        img_emb = res["z"]
        quant_loss = res["loss"]
        perplexity = res["perplexity"]

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        contrastive_loss = clip_loss_from_embeds(img_emb, text_emb, 100.0)

        loss = contrastive_loss + quant_loss

        logs = dict(
            quant_loss=quant_loss,
            contrastive_loss=contrastive_loss,
            loss=loss,
            perplexity=perplexity,
            pre_quant_contrastive_loss=pre_quant_contrastive_loss,
        )

        return loss, logs

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Builds temporary clip model
            vq_clip = self.__get_vq_clip_model()
            # uncomment to see how performance w/o adapters is the same as normal pretrained CLIP
            # vq_clip.vision_vq_adapter = None
            # vq_clip.text_vq_adapter = None

            processor = CLIPProcessor.from_pretrained(self.clip_url)

            if self.imagenet_path is not None:
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        top1, top5 = zero_shot_eval(
                            vq_clip,
                            processor,
                            self.imagenet_path,
                            self.validation_batch_size,
                        )
                self.log_dict(dict(imagenet_top1=top1, imagenet_top5=top5))

        img_emb, text_emb = batch
        loss, logs = self.step(img_emb, text_emb)

        self.log_dict({"v_" + k: v for k, v in logs.items()})
        return loss

    def training_step(self, batch, _):
        img_emb, text_emb = batch
        loss, logs = self.step(img_emb, text_emb)
        self.log_dict({"t_" + k: v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.vision_vq_adapter.parameters())
        learning_rate_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.lr_cycle_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            gamma=self.lr_gamma,
        )
        return [optimizer], [
            {"scheduler": learning_rate_scheduler, "interval": "step", "frequency": 1}
        ]