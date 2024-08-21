from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights("models/loras/iu_v35.safetensors", adapter_name="iu")
# pipeline.fuse_lora(lora_scale=0.7)

prompt = "nikon RAW photo,8 k,Fujifilm XT3,masterpiece, best quality, 1girl,solo,realistic, photorealistic,ultra detailed, diamond stud earrings, long straight black hair, hazel eyes, serious expression, slender figure, wearing a black blazer and white blouse, standing against a city skyline at night, iu1, <lora:iu_v35:1>"
image = pipe(prompt).images[0]
image.save('output/temp.png')