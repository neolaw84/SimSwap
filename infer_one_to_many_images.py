'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:26
Description: 
'''

import os
import glob
import pathlib

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage

from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def main(opt, model, app, spNorm):
    
    # start_epoch, epoch_iter = 1, 0
    
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    with torch.no_grad():
        pic_a = opt.pic_a_path

        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)

        # change it to work with new insightface
        assert len(img_a_align_crop) > 0
        
        # img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        # img_a = transformer_Arcface(img_a_align_crop_pil)
                
        # img_id_np = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        # img_id = img_id_np.cuda()

        # create latent id
        # img_id_downsample = F.interpolate(img_id, size=(112,112))
        # latend_id = model.netArc(img_id_downsample)

        img_a_downsample = cv2.resize(img_a_align_crop[0], (112, 112))
        latend_id = model.netArc.get(img_a_downsample)
        latend_id = torch.tensor([lid["embedding"] for lid in latend_id]).to(device="cuda")

        latend_id = F.normalize(latend_id, p=2, dim=1)
        print ("here here", latend_id.shape)


        ############## Forward Pass ######################

        pic_b = opt.pic_b_path
        img_b_whole = cv2.imread(pic_b)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        # detect_results = None
        swap_result_list = []
        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)


        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None

        # print ("outputing to ... {}".format(opt.output_path))
        reverse2wholeimage(b_align_crop_tenor_list,swap_result_list, b_mat_list, crop_size, img_b_whole, logoclass, \
            opt.output_path,opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)
        print(' ')

        print('************ Done ! ************')

if __name__ == "__main__":
    opt = TestOptions().parse()

    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    
    model = create_model(opt)
    model.eval()
    spNorm =SpecificNorm()

    # update insightface
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    # added for image dir
    
    if opt.dir_to_dir:
        target_images = glob.glob("{}/*.jpg".format(opt.pic_b_dir_path))
        out_image_names = [pathlib.Path(ti).name for ti in target_images]
        out_image_names = ["{}/{ti}".format(opt.output_dir_path, ti=ti) for ti in out_image_names]
        for ti, oin in zip (target_images, out_image_names):
            opt.pic_b_path =str(ti)
            opt.output_path = str(oin)
            main(opt, model, app, spNorm)

    # main(opt, model, app, spNorm)