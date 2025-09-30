# TRAINING SCRIPT FOR ParaDiffusion
#
# 1. ステージ1：ベースモデルの事前学習
#     * これは、SDXLやLlama-2のような巨大なモデルを、Webから収集した大規模なデータセットでゼロから学習する段階です
#         。これには膨大な計算資源と時間が必要で、通常はモデルの開発元（Stability AIやMetaなど）が行います。
#     * 私たちのtrain.pyは、この段階を実装していません。代わりに、すでに学習済みの公開モデル（stabilityai/stable-d
#         iffusion-xl-base-1.0など）をロードして利用します。これはこの種のプロジェクトでは標準的なアプローチです。
#
# 2. ステージ2：段落と画像の連携学習 (Alignment Learning)
#     * README.mdによると、この段階では「ParaImage-Big」(約400万ペア: CogVLMを用いてLAION-5Bの画像からキャプションを合成して作成)
#         という、より大規模で自動生成されたキャプションを持つデータセットを使い、テキストエンコーダ（Llama）と画像生成モデル（UNet）の連携を強化します。
#     * README.mdには「ParaImage-Big」のダウンロードリンクが提供されていないため、この段階を直接実行することはでき
#         ません。ただし、train.pyの学習ロジック自体はこのステージ2にも応用可能で、もしデータセットがあればパスを切
#         り替えることで実行できます。
#     * Llama V2 にLoRAを、 UNet は全パラメータを学習対象に適用
#
# 3. ステージ3：品質チューニング (Quality-Tuning)
#     * これは、最終的な画質と、詳細なプロンプトへの忠実度をさらに向上させるための最終仕上げの段階です。README.md
#         でダウンロード可能とされている、小規模で非常に高品質な「ParaImage-Small」(約85,000ペア)データセットを使用します。
#     * 現在の`train.py`は、まさにこのステージ3を実装するために作られています。 事前学習済みの強力なモデルをベース
#         に、高品質なデータセットでLoRAを用いて効率的にファインチューニングを行います。
#     * Frozen LLAMA V2 で テキストエンコーダを固定し、UNetにLoRAを適用


import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import math

# Import the custom pipeline
from pipeline_stable_diffusion_llama import StableDiffusionLlamaPipeline

# --- Dynamic Path Configuration ---
# Get the absolute path of the directory where this script is located (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SD_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
LLAMA_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights/Llama-2-7b-hf")
DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "ParaPrompts-400/ParaPrompts_400.csv")
DATA_ROOT_PATH = os.path.join(PROJECT_ROOT, "laion_aesthetic_sample/00000")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paraimage-model-lora")
# --------------------------------


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train ParaDiffusion model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help='The scheduler type to use.')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA projection matrix.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--image_size", type=int, default=512, help="The size of the images for training.")

    args = parser.parse_args()
    return args


class ParaImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, image_transform=None, image_size=512):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert("RGB")
        except (IOError, FileNotFoundError):
            print(f"Warning: Image not found at {img_name}. Skipping.")
            return None

        prompt = self.data.iloc[idx, 1]

        image = self.image_transform(image)
        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {"image": image, "input_ids": tokenized_prompt.input_ids.squeeze(0), "attention_mask": tokenized_prompt.attention_mask.squeeze(0)}

def collate_fn(examples):
    examples = [e for e in examples if e is not None]
    if not examples:
        return None

    images = torch.stack([example["image"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])

    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

def main(args):
    set_seed(args.seed)

    project_configs = ProjectConfiguration(project_dir=OUTPUT_DIR)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project_configs,
    )

    # Load models from fixed paths
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
    )
    vae = AutoencoderKL.from_pretrained(SD_MODEL_PATH, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(SD_MODEL_PATH, subfolder="unet", torch_dtype=torch.float16)
    unet.enable_gradient_checkpointing()
    noise_scheduler = DDPMScheduler.from_pretrained(SD_MODEL_PATH, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA to UNet and Text Encoder
    unet_lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    text_encoder_lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["q_proj", "v_proj"])

    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)

    # Optimizer
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and Dataloader
    train_dataset = ParaImageDataset(
        csv_file=DATA_CSV_PATH,
        root_dir=DATA_ROOT_PATH,
        tokenizer=tokenizer,
        image_size=args.image_size
    )

    print(f"Number of training examples: {len(train_dataset)}")

    print(f"Data Loader num_workers: {args.dataloader_num_workers}")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device)


    # Training loop
    total_batch_size = args.train_batch_size * accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    # max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    max_train_steps = 2

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            with accelerator.accumulate(unet, text_encoder):
                latents = vae.encode(batch["pixel_values"].to(torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True).hidden_states[-1]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(list(unet.parameters()) + list(text_encoder.parameters()), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                logs = {"loss": train_loss}
                progress_bar.set_postfix(**logs)
                train_loss = 0.0

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if (epoch + 1) % 10 == 0 or epoch == args.num_train_epochs - 1:
                save_path = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)

                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)

                unwrapped_unet.save_pretrained(os.path.join(save_path, "unet_lora"))
                unwrapped_text_encoder.save_pretrained(os.path.join(save_path, "text_encoder_lora"))

                print(f"Saved LoRA weights for epoch {epoch+1} to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
