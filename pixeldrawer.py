from DrawingInterface import DrawingInterface

import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

pydiffvg.set_print_timing(False)

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
        self.init_image = None
        if shape is not None:
            self.num_rows, self.num_cols = shape


    def load_model(self, config_path, checkpoint_path, device):
        # gamma = 1.0

        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = torch.device('cuda')
        pydiffvg.set_device(device)

        canvas_width, canvas_height = self.canvas_width, self.canvas_height
        num_rows, num_cols = self.num_rows, self.num_cols
        cell_width = canvas_width / num_cols
        cell_height = canvas_height / num_rows

        shapes = []
        shape_groups = []
        colors = []
        
        if self.init_image:
            newim = self.init_image.resize((num_cols,num_rows),resample=PIL.image.NEAREST)
            newim = newim.convert('RGB')
            pixels = list(newim.getdata())
            for r in range(num_rows):
                cur_y = r * cell_height
                for c in range(num_cols):
                    cur_x = c * cell_width
                    index = r * num_cols + c
                    pixel = pixels[index]
                    cell_color = torch.tensor([pixel[0], pixel[1], pixel[2], 1.0])
                    colors.append(cell_color)
                    p0 = [cur_x, cur_y]
                    p1 = [cur_x+cell_width, cur_y+cell_height]
                    path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
                    shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), stroke_color = None, fill_color = cell_color)
                    shape_groups.append(path_group)
        else:
            # Initialize Random Pixels
            for r in range(num_rows):
                cur_y = r * cell_height
                for c in range(num_cols):
                    cur_x = c * cell_width
                    if self.do_mono:
                        mono_color = random.random()
                        cell_color = torch.tensor([mono_color, mono_color, mono_color, 1.0])
                    else:
                        cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])
                    colors.append(cell_color)
                    p0 = [cur_x, cur_y]
                    p1 = [cur_x+cell_width, cur_y+cell_height]
                    path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
                    shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), stroke_color = None, fill_color = cell_color)
                    shape_groups.append(path_group)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        color_vars = []
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        # Optimizers
        # points_optim = torch.optim.Adam(points_vars, lr=1.0)
        # width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(color_vars, lr=0.02)

        self.img = img
        self.shapes = shapes 
        self.shape_groups  = shape_groups
        self.opts = [color_optim]

    def get_opts(self):
        return self.opts

    def rand_init(self, toksX, toksY):
        # TODO
        pass

    def init_from_tensor(self, init_tensor):
        self.init_image = init_tensor
        pass

    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        # TODO
        return 5

    def synth(self, cur_iteration):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, cur_iteration, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
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
            for group in self.shape_groups:
                group.fill_color.data[:3].clamp_(0.0, 1.0)
                group.fill_color.data[3].clamp_(1.0, 1.0)
                if self.do_mono:
                    avg_amount = torch.mean(group.fill_color.data[:3])
                    group.fill_color.data[:3] = avg_amount

    def get_z(self):
        return None

    def get_z_copy(self):
        return None
