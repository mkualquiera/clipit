
from DrawingInterface import DrawingInterface

import torch
import torch.nn
import random
import numpy as np
import PIL.Image

class PixelDrawer(DrawingInterface):
    num_rows = 45
    num_cols = 80
    do_mono = False
    pixels = []

    def __init__(self, width, height, do_mono, scale):
        super(DrawingInterface, self).__init__()

        self.num_cols = width
        self.num_rows = height
        self.do_mono = do_mono
        self.upsampler = False
        self.scale = scale


    def load_model(self, config_path, checkpoint_path, device):
        # gamma = 1.0

        self.device = device

        num_rows, num_cols = self.num_rows, self.num_cols

        # Initialize Random Pixels
        self.all_colors = torch.randn((1,3,num_rows,num_cols),
            device=self.device,requires_grad=True)

        self.color_vars = []

        self.color_vars.append(self.all_colors)

        # Optimizers
        # points_optim = torch.optim.Adam(points_vars, lr=1.0)
        # width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(self.color_vars, lr=0.02)

        self.opts = [color_optim]

    def get_opts(self):
        print("Obtained opt")
        return self.opts

    def rand_init(self, toksX, toksY):
        # TODO
        pass

    @torch.no_grad()
    def init_from_tensor(self, init_tensor):
        print("WORKING WITH INIT IMAGE!")
        new_vars = []
        newim = init_tensor.resize((self.num_cols,self.num_rows),resample=PIL.Image.NEAREST)
        newim = newim.convert('RGBA')
        pixels = list(newim.getdata())
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                index = r * self.num_cols + c
                pixel = pixels[index]
                
                self.all_colors[0,0,r,c] = pixel[0]/255.0
                self.all_colors[0,1,r,c] = pixel[0]/255.0
                self.all_colors[0,2,r,c] = pixel[0]/255.0
                ## TODO masking?
                ##if pixel[3]/255.0 > 0.8:
                ##    new_vars.append(self.color_vars[index])


    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        # TODO
        return 5

    def synth(self, cur_iteration):

        img = self.all_colors
        if not self.upsampler:
            self.upsampler = torch.nn.Upsample(scale_factor=self.scale,mode='nearest')
        img = self.upsampler(img)
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 255)
        # img = np.repeat(img, 4, axis=0)
        # img = np.repeat(img, 4, axis=1)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            torch.clamp(self.all_colors,min=0.0,max=1.0)
            

    def get_z(self):
        return None

    def get_z_copy(self):
        return None