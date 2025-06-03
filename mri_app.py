import torch 
from torch import nn 
import torch.optim as optim
import gradio as gr 
import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 
from PIL import Image
import numpy as np

from src.models.flowmatching_module import FlowMatchingModule 
from src.models.components.UNet import UNet 
from src.models.flow_matching.flow_matching import VelocityModel

num_steps = 1000
unet = UNet(
    in_ch = 1, 
    t_emb_dim = 256, 
    base_channel = 32, 
    multiplier = [1, 2, 4, 8], 
    type_condition = "continuous_label",
    use_discrete_time = False,
    use_attention = False
).to("cuda").eval()

velocity_model = VelocityModel(image_size=128, channel=1, net=unet).to("cuda").eval()

checkpoint_path = "/mnt/apple/k66/hanh/generative_model/logs/train/runs/axial_mri_flow_matching/conditional_flow_matching_mri/9lnrq9rz/checkpoints/epoch=399-step=1017200.ckpt"

extras = {
    "num_steps": num_steps,
    "velocity_model": velocity_model,
    "use_condition": True,
    "optimizer": None,
    "w": 4.0
}

model = FlowMatchingModule.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    **extras
).to("cuda")


def generate_image(age: float): 
    with torch.no_grad(): 
        age = float(age)/100.0
        age_tensor = torch.tensor([age], dtype=torch.float32, device="cuda") 
        age_tensor = age_tensor.unsqueeze(0) 

        image = model.sample(input_size=torch.Size([1, 1, 128, 128]), c=age_tensor)  # shape: (1, 1, 128, 128)
        
        image = image.squeeze(0).squeeze(0)
        image = image * 0.5 + 0.5

        image = image.clamp(0, 1)
        image = (image * 255).cpu().numpy().astype(np.uint8)

        image = Image.fromarray(image) 
        image.save("/mnt/apple/k66/hanh/generative_model/sample.png")

        return image



with gr.Blocks() as demo:
    gr.Markdown("# Application for generating MRI image from age!")
    gr.Markdown("Input age and enjoy!!")

    with gr.Row():
        age_input = gr.Number(label="Age (float)", value=23.0)
        output_image = gr.Image(label="MRI Image (128x128)", type="pil")

    generate_button = gr.Button("Generate")

    generate_button.click(fn=generate_image, inputs=age_input, outputs=output_image)

if __name__ == "__main__":
    demo.launch()


