import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import torch
import numpy as np

class ImageModule():
    # Upper left corner (0, 0)
    # option: aug & input_size
    def __init__(self, opt):
        self.opt = opt
        self.input_size = opt.input_size

        self.to_torch_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.bri_para = 0
        self.con_para = 0
        self.sat_para = 0
        self.hue_para = 0
        self.color_perm = torch.randperm(4)
        
    def auto_center(self, img, intrinsic=None):
        pass

    def crop(self, img, crop_rect, intrinsic=None):
        w, h = img.size
        x1, y1, x2, y2 = crop_rect
        img = img.crop((x1, y1, x2, y2))
        if intrinsic is not None:
            intrinsic[0, 2] += -x1
            intrinsic[1, 2] += -y1
        return img, intrinsic
        
    def resize(self, img, new_size, intrinsic=None):
        img = img.resize(new_size, Image.BILINEAR)
        if intrinsic is not None:
            intrinsic[0, :] *= new_size[0] / img.size[0]
            intrinsic[1, :] *= new_size[1] / img.size[1]
        return img, intrinsic
    
    def square(self, img, intrinsic=None):
        w, h = img.size
        np_img = np.array(img)
        s = max(w, h)
        square_img = np.zeros((s, s, 3), dtype=np_img.dtype)
        if h > w:
            square_img[:, s//2-w//2:s//2-w//2+w, :3] = np_img[:, :, :3]
            if intrinsic is not None:
                intrinsic[0, 2] += s//2 - w//2
        else:
            square_img[s//2-h//2:s//2-h//2+h, :, :3] = np_img[:, :, :3]
            if intrinsic is not None:
                intrinsic[1, 2] += s//2 - h//2
        img = Image.fromarray(square_img)
        return img, intrinsic

    def to_tensor(self, img):
        return self.to_torch_tensor(img)

    def flip_x(self, img, intrinsic=None):
        img = transforms.RandomHorizontalFlip(p=1.0)(img)
        if intrinsic is not None:
            intrinsic[0, :] *= -1.0
            intrinsic[0, 2] += img.width
        return img, intrinsic
    
    def transform_init(self, img):
        self.rand_scale = (torch.rand(1).item() - 0.5)*2*self.opt.aug.scale_ratio + 1
        self.rand_crop = []
        for i in range(4):
            self.rand_crop.append(torch.rand(1)*self.opt.aug.crop_ratio)

    def random_transform(self, img, intrinsic=None):
        w, h = img.size
        if self.opt.aug.random_scale:
            sw, sh = int(w*self.rand_scale), int(h*self.rand_scale)
            img, intrinsic = self.resize(img, (sw, sh), intrinsic)
            img, intrinsic = self.crop(img, (sw//2-w//2, sh//2-h//2, sw//2-w//2+w, sh//2-h//2+h), intrinsic)
        if self.opt.aug.random_trans:
            x1 = int(self.rand_crop[0]*w)
            y1 = int(self.rand_crop[1]*h)
            x2 = w - int(self.rand_crop[2]*w)
            y2 = h - int(self.rand_crop[3]*h)
            img, intrinsic = self.crop(img, (x1, y1, x2, y2), intrinsic)
        return img, intrinsic

    def color_aug_init(self):
        self.bri_para = (torch.rand(1).item() - 0.5)*2 * self.opt.aug.bri + 1
        self.sat_para = (torch.rand(1).item() - 0.5)*2 * self.opt.aug.sat + 1
        self.con_para = (torch.rand(1).item() - 0.5)*2 * self.opt.aug.con + 1
        self.hue_para = (torch.rand(1).item() - 0.5)*2 * self.opt.aug.hue
        
        self.color_perm = torch.randperm(4)

    def random_color_aug(self, img):
        for i in range(4):
            if self.color_perm[i] == 0:
                img = transforms.functional.adjust_brightness(img, self.bri_para)
            if self.color_perm[i] == 1:
                img = transforms.functional.adjust_contrast(img, self.con_para)
            if self.color_perm[i] == 2:
                img = transforms.functional.adjust_saturation(img, self.sat_para)
            if self.color_perm[i] == 3:
                img = transforms.functional.adjust_hue(img, self.hue_para)
        return img

    def random_mask(self, mask):
        mask_draw = ImageDraw.Draw(mask)
        rand_num = np.random.rand()
        # np.random.seed(3322)
        # if vid == 0:
        #     rand_num = 0.5
        # else:
        #     rand_num = 0
        if rand_num > 0.75:
            mask_num = 8
        elif rand_num > 0.25:
            mask_num = 4
        else:
            mask_num = 0
        for i in range(mask_num):
            x, y = np.random.rand() * 512, np.random.rand() * 512
            w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
            mask_draw.rectangle([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
        for i in range(mask_num):
            x, y = np.random.rand() * 512, np.random.rand() * 512
            w, h = np.random.rand() * 75 + 25, np.random.rand() * 75 + 25
            mask_draw.ellipse([x, y, x + w, h + y], fill=(0, 0, 0), outline=(0, 0, 0))
        return mask