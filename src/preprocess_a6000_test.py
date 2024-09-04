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
    parser.add_argument("--batch_size", "-b", type=int, default=15)

    args = parser.parse_args()
    bsz = args.batch_size

    cache_dir = "/131_data/jihwan/data/huggingface/hub/"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir).to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)


    data_root = "/cvdata1/jihwan/minecraft"
    # output_root = "/cvdata1/jihwan/minecraft_clip"

    path_test = os.path.join(data_root, "test")

    paths = [path_test]

    for path in paths:
        folders = sorted(glob(os.path.join(path, f"{args.rank}")))

        v_decoder = decord.VideoReader
 
        for folder in folders:
            output_path = folder.replace("minecraft", "minecraft_lmdb").replace("/cvdata1/jihwan/", "/131_data/jihwan/data/")
            output_path = output_path + ".npy"
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)

            video_files = sorted(glob(os.path.join(folder, "*.mp4")))
            # video_files = video_files[:10]
            print(f"Rank {args.rank}: Processing {len(video_files)} video")

            image_embedss = []

            for video_file in tqdm(video_files):
                vr = v_decoder(video_file)
                n_frames = len(vr)
                assert n_frames == 301

                video_data = vr.get_batch(range(1, n_frames)).asnumpy().transpose(0, 3, 1, 2)
                inputs = processor(images=video_data, return_tensors='pt')
                input = inputs['pixel_values'].to('cuda')

                assert 300 % bsz == 0
                num_it = 300//bsz

                for i in range(num_it):
                    pixel_values = input[bsz*i:bsz*(i+1)]

                    image_embeds = model.get_image_features(pixel_values=pixel_values)
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    image_embeds = image_embeds.cpu().detach().numpy()

                    image_embedss.append(image_embeds)
            
            image_embedss = np.concatenate(image_embedss, axis=0)

            print(f"Saving as {output_path}, {image_embedss.shape}")
            np.save(output_path, image_embedss)


                # for j in range(bsz):
                #     idx = i*bsz+j
                #     output_path = video_file.replace("minecraft", "minecraft_clip").replace(".mp4", f"_{idx:04d}.npy")
                #     output_path = output_path.replace("/cvdata1/jihwan/", "/131_data/jihwan/data/")
                #     np.save(output_path, image_embeds[j])



            
            