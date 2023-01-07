import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
from utils.utils import boolean_string, Params
import argparse
import pickle
from utils.constants import *


parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')

def init_model():

    args = parser.parse_args()
    params = Params(f'projects/{args.project}.yml')
    weight = args.load_weights
    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales
    obj_list = params.obj_list
    seg_list = params.seg_list
    cudnn.fastest = True
    cudnn.benchmark = True
    weight = torch.load(weight, 'cpu')
    weight_last_layer_seg = weight['segmentation_head.0.weight']
    if weight_last_layer_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        if params.seg_multilabel:
            seg_mode = MULTILABEL_MODE
        else:
            seg_mode = MULTICLASS_MODE

    model = HybridNetsBackbone(compound_coef=3, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                               scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=None,
                               seg_mode=seg_mode)
    model.load_state_dict(weight)
    model.requires_grad_(False)
    model.eval()
    return model

if __name__ == '__main__':
    model = init_model()
    model_bytes = pickle.dumps(model)
    with open('/home/hty/model.pkl', 'wb') as f:
        pickle.dump(model_bytes, f, protocol=2)
