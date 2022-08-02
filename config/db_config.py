import os
from yacs.config import CfgNode as CN
 
class config_db():

    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.logdir = ''
        self.cfg.load_db_net_checkpoint = ''
        self.cfg.checkpoint_path = ''
        self.cfg.result_path = ''
        self.cfg.lr = 0.0
        self.cfg.train_mask = False

        self.cfg.dataset = CN()
        self.cfg.dataset.serial_batches = False
        self.cfg.dataset.pin_memory = False
        self.cfg.dataset.dataroot = ''
        self.cfg.dataset.img_path = ''
        self.cfg.dataset.mask_path = ''
        self.cfg.dataset.depth_path = ''
        self.cfg.dataset.depth_db_path = ''
        self.cfg.dataset.param_ex_path = ''
        self.cfg.dataset.param_in_path = ''
        self.cfg.dataset.obj_path = ''
        self.cfg.dataset.taichi_intrinsic = False
        self.cfg.dataset.num_views = 0
        self.cfg.dataset.yaw_list = []
        self.cfg.dataset.input_size = 512
        self.cfg.dataset.angle = 30
        self.cfg.dataset.aug = CN()
        self.cfg.dataset.aug.bri = 0.0
        self.cfg.dataset.aug.con = 0.0
        self.cfg.dataset.aug.sat = 0.0
        self.cfg.dataset.aug.hue = 0.0
        self.cfg.dataset.aug.scale_ratio = 0.0
        self.cfg.dataset.aug.crop_ratio = 0.0
        self.cfg.dataset.aug.random_color = False
        self.cfg.dataset.aug.random_transform = False
        self.cfg.dataset.aug.flip_x = False
        self.cfg.dataset.model = CN()
        self.cfg.dataset.model.sampling_sigma = 0.0
        self.cfg.dataset.model.num_sample_inout = 0

        self.cfg.ne_camera = CN()
        self.cfg.ne_camera.samples = 0
        self.cfg.ne_camera.ray_batch = 0

        self.cfg.mvs_model = CN()
        self.cfg.mvs_model.noise = False
        self.cfg.mvs_model.no_sobel = False
        self.cfg.mvs_model.no_initial = False
        self.cfg.mvs_model.no_global = False
        self.cfg.mvs_model.pos_embedding = False
        self.cfg.mvs_model.time = False
        self.cfg.mvs_model.is_2k = False
        self.cfg.mvs_model.steps = 30

        self.cfg.record = CN()
        self.cfg.record.save_freq = 1
        self.cfg.record.show_freq = 1
        self.cfg.record.print_freq = 1
        
        self.cfg.num_threads = 0
        self.cfg.gpus = ''
        self.cfg.batch_size = 0
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()

if __name__ == '__main__':
    cc=config_db()
    cc.load("test.yaml")
    print(cc.cfg)
    print(cc.get_defalut_cfg())