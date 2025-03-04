# import warnings
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# import torch, yaml, cv2, os, shutil, sys
# import numpy as np
#
# np.random.seed(0)
# import matplotlib.pyplot as plt
# from tqdm import trange
# from PIL import Image
# from ultralytics.nn.tasks import attempt_load_weights
# from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
#
#
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)
#
#
# class ActivationsAndGradients:
#     """ Class for extracting activations and
#     registering gradients from targetted intermediate layers """
#
#     def __init__(self, model, target_layers, reshape_transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.reshape_transform = reshape_transform
#         self.handles = []
#         for target_layer in target_layers:
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_activation))
#             # Because of https://github.com/pytorch/pytorch/issues/61519,
#             # we don't use backward hook to record gradients.
#             self.handles.append(
#                 target_layer.register_forward_hook(self.save_gradient))
#
#     def save_activation(self, module, input, output):
#         activation = output
#
#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         self.activations.append(activation.cpu().detach())
#
#     def save_gradient(self, module, input, output):
#         if not hasattr(output, "requires_grad") or not output.requires_grad:
#             # You can only register hooks on tensor requires grad.
#             return
#
#         # Gradients are computed in reverse order
#         def _store_grad(grad):
#             if self.reshape_transform is not None:
#                 grad = self.reshape_transform(grad)
#             self.gradients = [grad.cpu().detach()] + self.gradients
#
#         output.register_hook(_store_grad)
#
#     def post_process(self, result):
#         logits_ = result[:, 4:]
#         boxes_ = result[:, :4]
#         sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#         return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
#             indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
#
#     def __call__(self, x):
#         self.gradients = []
#         self.activations = []
#         model_output = self.model(x)
#         post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
#         return [[post_result, pre_post_boxes]]
#
#     def release(self):
#         for handle in self.handles:
#             handle.remove()
#
#
# class yolov8_target(torch.nn.Module):
#     def __init__(self, ouput_type, conf, ratio) -> None:
#         super().__init__()
#         self.ouput_type = ouput_type
#         self.conf = conf
#         self.ratio = ratio
#
#     def forward(self, data):
#         post_result, pre_post_boxes = data
#         result = []
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf:
#                 break
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#         return sum(result)
#
#
# class yolov8_heatmap:
#     def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
#         device = torch.device(device)
#         ckpt = torch.load(weight)
#         model_names = ckpt['model'].names
#         model = attempt_load_weights(weight, device)
#         model.info()
#         for p in model.parameters():
#             p.requires_grad_(True)
#         model.eval()
#
#         target = yolov8_target(backward_type, conf_threshold, ratio)
#         target_layers = [model.model[l] for l in layer]
#         method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
#         method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
#
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int)
#         self.__dict__.update(locals())
#
#     def post_process(self, result):
#         result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
#         return result
#
#     def draw_detections(self, box, color, name, img):
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
#                     lineType=cv2.LINE_AA)
#         return img
#
#     def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
#         """Normalize the CAM to be in the range [0, 1]
#         inside every bounding boxes, and zero outside of the bounding boxes. """
#         renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
#         for x1, y1, x2, y2 in boxes:
#             x1, y1 = max(x1, 0), max(y1, 0)
#             x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
#             renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
#         renormalized_cam = scale_cam_image(renormalized_cam)
#         eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
#         return eigencam_image_renormalized
#
#     def process(self, img_path, save_path):
#         # img process
#         img = cv2.imread(img_path)
#         img = letterbox(img)[0]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#
#         try:
#             grayscale_cam = self.method(tensor, [self.target])
#         except AttributeError as e:
#             return
#
#         grayscale_cam = grayscale_cam[0, :]
#         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#
#         pred = self.model(tensor)[0]
#         pred = self.post_process(pred)
#         if self.renormalize:
#             cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
#                                                                grayscale_cam)
#         if self.show_box:
#             for data in pred:
#                 data = data.cpu().detach().numpy()
#                 cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())],
#                                                  f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
#                                                  cam_image)
#
#         cam_image = Image.fromarray(cam_image)
#         cam_image.save(save_path)
#
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)
#
#         if os.path.isdir(img_path):
#             for img_path_ in os.listdir(img_path):
#                 self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
#         else:
#             self.process(img_path, f'{save_path}/result.png')
#
#
# def get_params():
#     params = {
#         'weight': './yolov8n.pt',  # 现在只需要指定权重即可,不需要指定cfg
#         'device': 'cuda:0',
#         'method': 'EigenGradCAM',
#         # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
#         'layer': [10, 12, 14, 16, 18], #指定要使用的特征层的索引列表，这些层将用于提取特征并生成热力图。不同的层可能会提供不同的特征信息。
#         'backward_type': 'all',  # class, box, all  #指定回传的类型，可以是 class（仅针对分类）、box（仅针对边框）或 all（两者均使用）。
#         'conf_threshold': 0.2,  # 0.2  #置信度阈值，只有置信度高于该值的检测结果才会被考虑。例如，0.65 表示只有置信度超过 65% 的检测结果会被处理。
#         'ratio': 0.02,  # 0.02-0.1  #处理的预测结果的比例，0.02 表示只处理前 2% 的预测结果（按置信度排序）。可以根据需要调整，以控制要可视化的检测数量。
#         'show_box': True,  #是否在生成的热力图上绘制检测框。如果为 True，则会在热力图上显示检测到的对象的边框。
#         'renormalize': True   #是否将热力图归一化到每个边框内的范围 [0, 1]。如果为 True，热力图将在边框内归一化，以更好地突出显示目标区域。
#     }
#     return params
#
#
# if __name__ == '__main__':
#     model = yolov8_heatmap(**get_params())
#     # model(r'/home/hjj/Desktop/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/images/9999947_00000_d_0000026.jpg', 'result')
#     model(r'D:\xhf\biye\st-yolo\ST-YOLO\ultralytics\assets', 'result')











import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_dir, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # process each image in directory
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            # img process
            img = cv2.imread(img_path)
            img = letterbox(img)[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0
            tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

            # init ActivationsAndGradients
            grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

            # get ActivationsAndResult
            result = grads(tensor)
            activations = grads.activations[0].cpu().detach().numpy()

            # postprocess to yolo output
            post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
            for i in trange(int(post_result.size(0) * self.ratio), desc=f'Processing {img_name}'):
                if float(post_result[i].max()) < self.conf_threshold:
                    break

                self.model.zero_grad()
                # get max probability for this prediction
                if self.backward_type == 'class' or self.backward_type == 'all':
                    score = post_result[i].max()
                    score.backward(retain_graph=True)

                if self.backward_type == 'box' or self.backward_type == 'all':
                    for j in range(4):
                        score = pre_post_boxes[i, j]
                        score.backward(retain_graph=True)

                # process heatmap
                if self.backward_type == 'class':
                    gradients = grads.gradients[0]
                elif self.backward_type == 'box':
                    gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
                else:
                    gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                                grads.gradients[4]
                b, k, u, v = gradients.size()
                weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                      gradients.detach().numpy())
                weights = weights.reshape((b, k, 1, 1))
                saliency_map = np.sum(weights * activations, axis=1)
                saliency_map = np.squeeze(np.maximum(saliency_map, 0))
                saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                if (saliency_map_max - saliency_map_min) == 0:
                    continue
                saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

                # add heatmap and box to image
                cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
                cam_image = Image.fromarray(cam_image)
                cam_image.save(f'{save_path}/{img_name}_{i}.png')

def get_params():
    params = {
        'weight': './our_best.pt', # 这选择想要热力可视化的模型权重路径
        'cfg': './ultralytics/cfg/models/11/yolo11_SPDConv+SE.yaml', # 这里选择与训练上面模型权重相对应的.yaml文件路径
        'device': 'cuda:0', # 选择设备，其中0表示0号显卡。如果使用CPU可视化 # 'device': 'cpu'
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[23]',   # 选择特征层
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.65, # 置信度阈值默认0.65， 可根据情况调节
        'ratio': 0.02 # 取前多少数据，默认是0.02，可根据情况调节
    }
    return params

if __name__ == '__main__':
    model = yolov8_heatmap(**get_params()) # 初始化
    model(r'D:/w/2O/jyz\val', './result5.5') # 第一个参数是图片文件夹的路径，第二个参数是保存路径，比如是result的话，其会创建一个名字为result的文件夹，如果result文件夹不为空，其会先清空文件夹。
