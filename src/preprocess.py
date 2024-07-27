from PIL import Image
import os
from glob import glob
from transformers import CLIPProcessor, CLIPModel
import decord
from tqdm import tqdm
import argparse
import numpy as np


if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", "-r", type=int, default=0)

    args = parser.parse_args()

    cache_dir = "/131_data/jihwan/data/huggingface/hub/"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir).to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)


    data_root = "/cvdata1/jihwan/minecraft"
    # output_root = "/cvdata1/jihwan/minecraft_clip"

    path_train = os.path.join(data_root, "train")
    # path_test = os.path.join(data_root, "test")

    paths = [path_train]
    # paths = [path_train, path_val]

    v_decoder = decord.VideoReader

    for path in paths:
        video_files = sorted(glob(os.path.join(path, f"{args.rank+1}_*/*.mp4")))
        print(f"Rank {args.rank}: Processing {len(video_files)} video")

        for video_file in tqdm(video_files):
            output_dir = os.path.dirname(video_file).replace("minecraft", "minecraft_clip")
            os.makedirs(output_dir, exist_ok=True)

            vr = v_decoder(video_file)
            n_frames = len(vr)
            assert n_frames == 301

            video_data = vr.get_batch(range(1, n_frames)).asnumpy().transpose(0, 3, 1, 2)
            inputs = processor(images=video_data, return_tensors='pt')
            input = inputs['pixel_values'].to('cuda')

            for i in range(20):
                pixel_values = input[15*i:15*(i+1)]

                image_embeds = model.get_image_features(pixel_values=pixel_values, output_hidden_states=True)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                image_embeds = image_embeds.cpu().detach().numpy()

                for j in range(15):
                    idx = i*15+j
                    output_path = video_file.replace("minecraft", "minecraft_clip").replace(".mp4", f"_{idx}.npy")
                    np.save(output_path, image_embeds[j])



            
            