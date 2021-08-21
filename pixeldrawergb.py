
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

    def __init__(self, width, height, do_mono, shape=None):
        super(DrawingInterface, self).__init__()

        self.canvas_width = width
        self.canvas_height = height
        self.do_mono = do_mono
        self.upsampler = False
        if shape is not None:
            self.num_rows, self.num_cols = shape


    def load_model(self, config_path, checkpoint_path, device):
        # gamma = 1.0

        self.device = torch.device('cuda')

        canvas_width, canvas_height = self.canvas_width, self.canvas_height
        num_rows, num_cols = self.num_rows, self.num_cols
        
        cell_width = canvas_width / num_cols
        cell_height = canvas_height / num_rows

        # Initialize Random Pixels
        colors = []
        for r in range(num_rows):
            for c in range(num_cols):
                if self.do_mono:
                    mono_color = random.random()
                    cell_color = torch.tensor([mono_color, mono_color, mono_color])
                else:
                    cell_color = torch.tensor([random.random(), random.random(), random.random()])
                cell_color = cell_color.to(self.device)
                colors.append(cell_color)

        self.all_colors = colors

        self.color_vars = []
        for color in colors:
            color.requires_grad = True
            self.color_vars.append(color)

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
                self.color_vars[index][0] = pixel[0]/255.0
                self.color_vars[index][1] = pixel[1]/255.0
                self.color_vars[index][2] = pixel[2]/255.0
                self.all_colors[index][0] = pixel[0]/255.0
                self.all_colors[index][1] = pixel[1]/255.0
                self.all_colors[index][2] = pixel[2]/255.0
                if pixel[3]/255.0 > 0.8:
                    new_vars.append(self.color_vars[index])
        self.color_vars = new_vars
        color_optim = torch.optim.Adam(self.color_vars, lr=0.02)
        self.opts = [color_optim]
        print("Updated opt")


    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        # TODO
        return 5

    def synth(self, cur_iteration):

        colorstensor = torch.stack(self.all_colors)
        img = colorstensor.reshape((self.num_rows,self.num_cols,3))
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if not self.upsampler:
            self.upsampler = torch.nn.Upsample(scale_factor=6,mode='nearest')
        img = self.upsampler(img)
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 254)
        # img = np.repeat(img, 4, axis=0)
        # img = np.repeat(img, 4, axis=1)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            for color in self.all_colors:
                color.data.clamp_(0.0, 1.0)
                if self.do_mono:
                    avg_amount = torch.mean(color)
                    color.data[:3] = avg_amount

    def get_z(self):
        return None

    def get_z_copy(self):
        return None