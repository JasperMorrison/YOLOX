
import numpy as np
from skimage import io
import os
import cv2
from pathlib import Path
from loguru import logger

class ConvCamExtractor():
    def __init__(self, layer):
        self.layer = layer

    def register_hook(self):
        self.features = []
        self.gradients = []
        self.handlers = []
        self.handlers.append(self.layer.register_forward_hook(self._get_features_hook))
        self.handlers.append(self.layer.register_full_backward_hook(self._get_grads_hook))

    def _get_features_hook(self, module, input, output):
        self.features.append(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradients.append(output_grad[0])

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # resize to image size
    h,w,c = image.shape
    mask = cv2.resize(mask, (w,h))
    # print("gen_cam, image and mask shape:", image.shape, mask.shape)
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    heatmap = heatmap[..., ::-1] # bgr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return [norm_image(cam), (heatmap).astype(np.uint8)]

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def conv_grad_cam(gradients, features):
    cams = []
    normal = len(features)
    for i,gradient in enumerate(gradients):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = features[i%normal][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)
    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def conv_grad_camplusplus(gradients, features):
    cams = []
    normal = len(features)
    for i,gradient in enumerate(gradients):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = features[i%normal][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)

    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def gen_input_grad(gradients):
    total_gradient = None
    for i,gradient in enumerate(gradients):
        if total_gradient is not None:
            total_gradient += gradient
        else:
            total_gradient = gradient
    return total_gradient

def preproc_from_data_augment(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    return padded_img

class Grad_CAM():
    def __init__(self) -> None:
        self.extrators = {}

    def grad_cam_register(self, model = None, cam_layer=[]):
        print(model)
        for layer_name in cam_layer:
            layer = model.get_submodule(layer_name)
            layer_cam = ConvCamExtractor(layer)
            self.extrators[layer_name] = layer_cam

        for cam in self.extrators.values():
            cam.register_hook()

        logger.info(self.extrators.keys())

    def grad_cam_remove(self):
        for cam in self.extrators.values():
            cam.remove_handlers()
        self.extrators.clear()

    def save_cam(self, save_folder, img_info):
        r = img_info['ratio']
        raw_image = img_info['raw_img']
        img_name = img_info['file_name']
        input_size = img_info['input_size']
        os.makedirs(save_folder, exist_ok=True)
        input_image = preproc_from_data_augment(raw_image, input_size) 

        for layer_name, cam_ext in self.extrators.items():
            if len(cam_ext.gradients) <= 0 or len(cam_ext.features) <= 0:
                logger.warning(layer_name + ", empty gradients")
                continue

            cam_img = gen_cam(input_image, conv_grad_cam(cam_ext.gradients, cam_ext.features))
            cam_img[0] = cam_img[0][: int(raw_image.shape[0] * r), : int(raw_image.shape[1] * r)]
            cam_img[1] = cam_img[1][: int(raw_image.shape[0] * r), : int(raw_image.shape[1] * r)]
            camplusplus_img = gen_cam(input_image, conv_grad_camplusplus(cam_ext.gradients, cam_ext.features))
            camplusplus_img[0] = camplusplus_img[0][: int(raw_image.shape[0] * r), : int(raw_image.shape[1] * r)]
            camplusplus_img[1] = camplusplus_img[1][: int(raw_image.shape[0] * r), : int(raw_image.shape[1] * r)]

            cam_img_name = img_name + "_" + layer_name + "_cam.jpg"
            cam_path = str(Path(save_folder) / cam_img_name)
            heatmap_name = img_name + "_" + layer_name + "_heatmap.jpg"
            heatmap_path = str(Path(save_folder) / heatmap_name)
            io.imsave(cam_path, cam_img[0])
            io.imsave(heatmap_path, cam_img[1])

            cam_img_name = img_name + "_" + layer_name + "_cam++.jpg"
            cam_path = str(Path(save_folder) / cam_img_name)
            heatmap_name = img_name + "_" + layer_name + "_heatmap++.jpg"
            heatmap_path = str(Path(save_folder) / heatmap_name)
            io.imsave(cam_path, camplusplus_img[0])
            io.imsave(heatmap_path, camplusplus_img[1])


