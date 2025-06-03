import torch 
from torch import nn 
import gradio as gr 
import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 
from src.models.srdiffusion_module import SuperResolutionDiffusionModule 
from torchvision import transforms 
from PIL import Image
import numpy as np 



path = "/mnt/apple/k66/hanh/diffusion/logs/train/runs/sr_diff_final/sr_diffusion/2zfwg25j/checkpoints/epoch=209-step=1470000.ckpt"
model = SuperResolutionDiffusionModule.load_from_checkpoint(path).to("cuda") 
model.eval() 


def process_image(input_image):
    with torch.no_grad(): 
        input_image = transforms.ToTensor()(input_image)
        input_image = transforms.Resize((64, 64))(input_image)
        input_image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(input_image)
        input_image = input_image.unsqueeze(0) 
        input_image = input_image.to("cuda") 
        model.sampler.denoise_net = model.diffusion_model.denoise_net 
        minus_pred, _ = model.sampler.reverse_process(c=input_image, batch_size=1) 
        minus_pred = minus_pred.clamp(-1, 1)  
        up = torch.nn.functional.interpolate(input_image, scale_factor=4, mode="bilinear")
        hr_pred = up + minus_pred 
        hr_pred = hr_pred.clamp(-1, 1) 
        hr_pred = model.rescale(hr_pred) 
        output_image = hr_pred.squeeze(0) 
        output_image = output_image.permute(1, 2, 0)
        output_image = (output_image.cpu().numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image) 
        return output_image 


with gr.Blocks() as demo:
    gr.Markdown("# Application for Image Super Resolution")
    gr.Markdown("**Drop your image and enjoy this moment. Warning: Only accept 64x64x3 image.**")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Output Image", type="pil")
    
    process_button = gr.Button("Process...")
    
    process_button.click(process_image, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    demo.launch()
