import torch
import  pickle
import cv2
import rospy
import numpy as np
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from torchvision import transforms
import argparse
import time

parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
parser.add_argument('--conf_thresh', type=restricted_float, default='0.55')
parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
parser.add_argument('--cuda', type=boolean_string, default=False)
args = parser.parse_args()
params = Params(f'/home/hty/PycharmProjects/HybridNets/projects/{args.project}.yml')

def process_image(image=None):
    model_bytes = None
    # 模拟输入， 实际输入由外部传入
    if image is None:
        image = cv2.imread("/home/hty/PycharmProjects/HybridNets/demo/image/4.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        print("read image from local")
    else:
        image_np = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imread(image_np)

    start_time = time.perf_counter()
    if model_bytes is None:
        print("model_bytes is None, load from local.")
        with open('/home/hty/model.pkl', 'rb') as f:
            model_bytes = pickle.load(f)
    model = pickle.loads(model_bytes)
    end_time = time.perf_counter()
    print("load time: ", end_time - start_time)

    # 初始化一些变量和参数
    use_cuda = args.cuda
    color_list_seg = {'road': [43, 255, 153], "lane": [255, 7, 205]} # 分割区域的颜色
    shapes = []
    input_imgs = []
    obj_list = params.obj_list
    threshold = args.conf_thresh # for detection
    iou_threshold = args.iou_thresh
    color_list = standard_to_bgr(STANDARD_COLORS) # transform to list of 3-tuples

    # 初始化原始图像列表，转换颜色，调整大小
    ori_imgs = [image]
    ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
    ori_imgs = [cv2.resize(img, (1280, 720)) for img in ori_imgs]

    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)
        print("max resized shape: ", resized_shape)

    normalize = transforms.Normalize(
        mean=params.mean, std=params.std
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    for ori_img in ori_imgs:
        # redundant ?
        h0, w0 = ori_img.shape[:2] # h, w, c
        r = resized_shape / max(h0, w0)  # resize image to img_size
        input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)

        h, w = input_img.shape[:2]
        # 缩放并在图片顶部、底部添加灰边
        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)
        input_imgs.append(input_img)

        # 用于还原 ？
        shapes.append(((h0, w0), ((h / h0, w / w0), pad)))

    if use_cuda:
        model = model.cuda()
        x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
    else:
        x = torch.stack([transform(fi) for fi in input_imgs], 0)

    with torch.no_grad():
        # prediction
        features, regression, classification, anchors, seg = model(x)
        seg_mask_list = []
        _, seg_mask = torch.max(seg, 1) # dim = 1 ?
        seg_mask_list.append(seg_mask)

        for i in range(seg.size(0)):
            for seg_class_index, seg_mask in enumerate(seg_mask_list):
                seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                pad_h = int(shapes[i][1][1][1])
                pad_w = int(shapes[i][1][1][0])
                seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0] - pad_h, pad_w:seg_mask_.shape[1] - pad_w]
                seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
                color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                for index, seg_class in enumerate(params.seg_list):
                    color_seg[seg_mask_ == index + 1] = color_list_seg[seg_class]
                color_seg = color_seg[..., ::-1]
                color_mask = np.mean(color_seg, 2)
                seg_img = ori_imgs[i]
                seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                seg_img = seg_img.astype(np.uint8)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

        for i in range(len(ori_imgs)):
            out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])
                plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                             color=color_list[get_index_label(obj, obj_list)])

    cv2.imwrite('/home/hty/PycharmProjects/HybridNets/ros.jpg', cv2.cvtColor(ori_imgs[0], cv2.COLOR_RGB2BGR))
    # print(ori_imgs[0].tobytes())
    return ori_imgs[0].tobytes()

if __name__ == '__main__':
    start_time = time.perf_counter()
    input_bytes = sys.stdin.buffer.read()
    process_image(pickle.loads(input_bytes))  # one second on cpu, 4 seconds on cuda
    end_time = time.perf_counter()
    print("total time:", end_time - start_time)

