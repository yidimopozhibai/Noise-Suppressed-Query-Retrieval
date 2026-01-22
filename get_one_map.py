import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
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


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return logits_[indices], boxes_[indices], xywh2xyxy(boxes_[indices]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0][0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class rtdetr_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class rtdetr_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = rtdetr_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.__dict__.update(locals())

    def post_process(self, result, shape):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]

        # filter
        score, cls = logits_.max(1, keepdim=True)
        idx = (score > self.conf_threshold).squeeze()
        logits_, boxes_ = logits_[idx], boxes_[idx]

        # xywh -> xyxy
        h, w = shape
        boxes_ = xywh2xyxy(boxes_)
        boxes_[:, 0] *= w
        boxes_[:, 2] *= w
        boxes_[:, 1] *= w
        boxes_[:, 3] *= w

        return torch.cat([boxes_, logits_], dim=1)

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        h, w, _ = image_float_np.shape
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        img = letterbox(img, auto=False, scaleFill=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        pred = self.model(tensor)[0][0]
        pred = self.post_process(pred, img.shape[:2])
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())],
                                                 f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                                                 cam_image)
        cam_image = cv2.resize(cam_image, (ori_w, ori_h))
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)


        ######################################

    # def Gan_For_New_img(self,img):
    #     import cv2
    #     """图片风格迁移"""
    #
    #     #img = cv2.resize(img, (500, 500))
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    #     # 图片预处理
    #     (h, w) = img.shape[:2]
    #     blob = cv2.dnn.blobFromImage(img, 1, (w, h), (0, 0, 0), swapRB=True)  # 改变图像格式 B C H W
    #     # blob = cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None)
    #     # 参数:
    #     #       image:表示输入图像。
    #     #       scalefactor:表示对图像内的数据进行缩放的比例因子。具体运算是每个像素值*scalefactor,该值默认为 1。
    #     #       size:用于控制blob的宽度、高度。
    #     #       mean:表示从每个通道减去的均值。 (0,0,0):表示不进行均值减法。即不对图像的B、G、R通道进行任何减法操作。
    #     #               若输入图像本身是B、G、R通道顺序的,并且下一个参数swapRB值为True,
    #     #               则mean值对应的通道顺序为R、G、B。:opencv BGR RGB
    #     #       swapRB:表示在必要时交换通道的R通道和B通道。一般情况下使用的是RGB通道。而openCV通常采用的是BGR通道
    #     #               因此可以根据需要交换第1个和第3个通道。该值默认为 False。
    #     #       crop:布尔值,如果为True:则在调整大小后进行居中裁剪
    #     # 返回值:blob: 表示在经过缩放、裁剪、减均值后得到的符合人工神经网络输入的数据。该数据是一个四维数据,
    #     #           布局通常使用N(表示batch size)、C(图像通道数,如RGB图像具有三个通道)、H(图像高度)、W(图像宽度)表示
    #
    # # 加载模型
    #     net = cv2.dnn.readNet(r'la_muse.t7')  # readNet 通用读取模型 不论后缀
    #     net.setInput(blob)  # 将图片传入模型
    #     out = net.forward()  # 向前传播
    #     # 重塑形状(忽略第1维),4维变3维
    #     # 调整输出out的形状,模型推理输出out是四维BCHW形式的,调整为三维CHW形式
    #     out_new = out.reshape(out.shape[1], out.shape[2], out.shape[3])
    #     # 对输入的数组(或图像)进行归一化处理,使其数值范围在指定的范国内
    #     cv2.normalize(out_new, out_new, norm_type=cv2.NORM_MINMAX)  # 输入数组 输出数组  归一化类型
    #     # 转置输出结果的维度
    #     # result = out_new
    #     result = out_new.transpose(1, 2, 0)  # 将图像转置成 高 宽 通道数 符合opencv的格式
    #     # 显示转换后的图像
    #     cv2.imshow('stylized Image', result)
    #     cv2.imwrite('headmap\stylized.jpg', result)  # 保存带有检测框的原图
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     return result

    def Gan_For_New_img(self, img):
        import cv2
        """图片风格迁移"""
        cv2.imshow('img', img)
        #cv2.waitKey(0)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1, (w, h), (0, 0, 0), swapRB=True)
        net = cv2.dnn.readNet(r'la_muse.t7')
        net.setInput(blob)
        out = net.forward()
        out_new = out.reshape(out.shape[1], out.shape[2], out.shape[3])
        cv2.normalize(out_new, out_new, norm_type=cv2.NORM_MINMAX)
        result = out_new.transpose(1, 2, 0)
        cv2.imshow('stylized Image', result)
        cv2.imwrite('headmap/stylized.jpg', result)
        return result


    def __call__1(self, img_path):
        # img process
        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        img, _, _ = letterbox(img, auto=False, scaleFill=True)  # 使用letterbox函数进行尺寸调整和填充
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 生成热图
        grayscale_cam = self.method(tensor, [self.target])[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)


        # 生成热图掩码
        cam_mask = grayscale_cam.copy()
        cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())  # 归一化
        cam_mask = (cam_mask * 255).astype('uint8')  # 转换为8位灰度图


#=====================
        # img process
        img1 = cv2.imread(img_path)
        ori_h, ori_w = img1.shape[:2]
        img1, _, _ = letterbox(img1, auto=False, scaleFill=True)  # 使用letterbox函数进行尺寸调整和填充
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = np.float32(img1) / 255.0
        tensor = torch.from_numpy(np.transpose(img1, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 生成热图
        grayscale_cam1 = self.method(tensor, [self.target])[0, :]
        cam_image = show_cam_on_image(img1, grayscale_cam1, use_rgb=True)

        # 生成热图掩码
        cam_mask1 = grayscale_cam1.copy()
        cam_mask1 = (cam_mask1 - cam_mask1.min()) / (cam_mask1.max() - cam_mask1.min())  # 归一化
        cam_mask1 = (cam_mask1 * 255).astype('uint8')  # 转换为8位灰度图

        # 二值化处理
        _, binary_mask = cv2.threshold(cam_mask1, 160, 255, cv2.THRESH_BINARY)  # 使用127作为阈值
        plt.imshow(cam_image)
        plt.title("Heatmap")
        plt.show()

        # 显示热图掩码
        plt.imshow(cam_mask, cmap='gray')
        plt.title("Heatmap Mask")
        plt.show()
        # 显示二值化热图掩码
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Binary Heatmap Mask")
        plt.show()
        # 调用函数

    def __call__(self, img_path):
        # 读取图片
        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        # 图片预处理
        img, _, _ = letterbox(img, auto=False, scaleFill=True)  # 使用letterbox函数进行尺寸调整和填充
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0  # 归一化到[0, 1]
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 生成热图
        grayscale_cam = self.method(tensor, [self.target])[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # 生成热图掩码
        cam_mask = grayscale_cam.copy()
        cam_mask = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min())  # 归一化
        cam_mask = (cam_mask * 255).astype('uint8')  # 转换为8位灰度图

        # 二值化处理
        _, binary_mask = cv2.threshold(cam_mask, 160, 255, cv2.THRESH_BINARY)  # 使用200作为阈值

        # 将二值化热图掩码贴在原图上
        binary_mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        overlay_img = cv2.addWeighted(binary_mask_3channel, 0.5, img, 0.5, 0, dtype = cv2.CV_32F)
        # 显示原图检测框
        pred = self.model(tensor)[0][0]
        pred = self.post_process(pred, img.shape[:2])
        detected_img = img.copy()  # 复制一份原图用于绘制检测框
        for data in pred:
            data = data.cpu().detach().numpy()
            x1, y1, x2, y2 = data[:4].astype(int)
            # 确保坐标不越界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            self.draw_detections([x1, y1, x2, y2], self.colors[int(data[4:].argmax())],
                                 f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                                 detected_img)

        # 将归一化的图像转换回[0, 255]范围并保存
        original_img = (img * 255).astype('uint8')
        # cv2.imwrite('headmap\original_img.jpg', original_img)  # 保存原图
        detected_img = (detected_img * 255).astype('uint8')  # 将归一化的检测图转换回[0, 255]范围
        cv2.imwrite('headmap\detected_img.jpg', detected_img)  # 保存带有检测框的原图
        cv2.imwrite('headmap\heatmap.jpg', cam_image)  # 保存热图

        overlay_img_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Overlay Image', overlay_img_bgr)

        #这段代码将生成四个图像文件：original_img.jpg（原图）、detected_img.jpg（带有检测框的原图）、heatmap.jpg（热图），以及overlay_img.jpg（使用二值化热图作为掩码贴在原图上的图像）。同时，这些图像也会在屏幕上显示。
############################################################
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        # 应用风格迁移
        Gan_img = self.Gan_For_New_img(img)
        print("Gan_img min:", Gan_img.min()) #cao tm de 傻逼网络
        print("Gan_img max:", Gan_img.max())
        # 将风格迁移后的图像转换为BGR格式以正确显示

        Gan_img_bgr = cv2.cvtColor(Gan_img, cv2.COLOR_RGB2BGR)
        #Gan_img_bgr =Gan_img
        # 获取原图的BGR格式
        original_img_bgr = cv2.imread(img_path)
        # cv2.imshow("Gan_img_bgr",Gan_img_bgr)
        # 确保original_img_bgr与Gan_img_bgr尺寸一致
        if original_img_bgr.shape[:2] != Gan_img_bgr.shape[:2]:
            original_img_bgr = cv2.resize(original_img_bgr, (Gan_img_bgr.shape[1], Gan_img_bgr.shape[0]))

        # 将二值化热图掩码调整到与Gan_img_bgr相同的尺寸
        binary_mask_resized = cv2.resize(binary_mask, (Gan_img_bgr.shape[1], Gan_img_bgr.shape[0]))
        # cv2.imshow("binary_mask_resized",binary_mask_resized) okkkkkkkkkkkkkkkkkkkkkkk
        # 将原图和风格迁移后的图像都转换为浮点数，以便进行加权混合
        original_img_float = original_img_bgr.astype(np.float32) / 255.0
        # Gan_img_float = Gan_img_bgr.astype(np.float32) / 255.0
        # 如果网络输出的是0到255的值，将其归一化到0到1的范围
        if Gan_img.max() > 1:
            Gan_img_float = Gan_img.astype(np.float32) / 255.0
        else:
            Gan_img_float = Gan_img
        # 确保像素值在0-1范围内


        print("Gan_img_float min:", Gan_img_float.min())
        print("Gan_img_float max:", Gan_img_float.max())

        Gan_img_float = np.clip(Gan_img_float, 0, 1)
        # 将二值化掩码转换为灰度图，其中掩码区域为1，非掩码区域为0
        mask_weights = binary_mask_resized.astype(np.float32) / 255.0

        # 创建一个与mask_weights相同尺寸的三通道数组，用于存储1 - mask_weights
        inverse_mask_weights = 1.0 - mask_weights
        # test3=(inverse_mask_weights * 255).astype(np.uint8)
        # cv2.imshow("test3",test3) okkkkkkkkkkkkkkkkkkkkkkkkkkkkk
        # 扩展mask_weights和inverse_mask_weights到三通道
        mask_weights_3channel = np.stack((mask_weights, mask_weights, mask_weights), axis=-1)
        inverse_mask_weights_3channel = np.stack((inverse_mask_weights, inverse_mask_weights, inverse_mask_weights),
                                                 axis=-1)

        # 使用numpy的逐元素乘法和加法将原图和风格迁移后的图像混合
        combined_img_float = original_img_float * inverse_mask_weights_3channel+Gan_img_float * mask_weights_3channel

        # 确保像素值在0-1范围内
        combined_img_float = np.clip(combined_img_float, 0, 1)

        # 将混合后的图像转换回8位整数
        combined_img = (combined_img_float * 255).astype(np.uint8)





        # test1=Gan_img_float
        # test2 = (test1 * 255).astype(np.uint8) #test2 全黑 有问题



        # cv2.imshow("test2",test2)
        # 显示组合后的图像
        cv2.imshow("Combined Image", combined_img)

        # 保存组合后的图像
        cv2.imwrite('headmap/GanAttack.jpg', combined_img)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()







        ############gan图片

# #1 白色攻击 ####
#         # 将overlay_img转换回[0, 255]范围并转换为uint8
#         overlay_img = (overlay_img * 255).astype('uint8')
#         overlay_img_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以正确显示
#
#         # 将增加了掩码的图片重新传入网络得到检测结果
#         # 将overlay_img_bgr转换回RGB格式
#         overlay_img_rgb = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGB)
#         # 归一化到[0, 1]
#         overlay_img_rgb = np.float32(overlay_img_rgb) / 255.0
#         # 调整尺寸和填充
#         overlay_img_rgb, _, _ = letterbox(overlay_img_rgb, new_shape=(640, 640), scaleFill=False, scaleup=True)
#         # 转换为模型输入格式
#         overlay_tensor = torch.from_numpy(np.transpose(overlay_img_rgb, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#
#         # 进行预测
#         with torch.no_grad():  # 确保不计算梯度
#             pred = self.model(overlay_tensor)[0][0]
#             pred = self.post_process(pred, overlay_img_rgb.shape[:2])
#
#         # 将预测结果转换回原始图像尺寸
#         attack_result_img = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGB)
#         attack_result_img = cv2.resize(attack_result_img, (ori_w, ori_h))
#
#         # 绘制检测框
#         for data in pred:
#             data = data.cpu().detach().numpy()
#             x1, y1, x2, y2 = data[:4].astype(int)
#             # 确保坐标不越界
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(attack_result_img.shape[1], x2), min(attack_result_img.shape[0], y2)
#             self.draw_detections([x1, y2, x2, y1], self.colors[int(data[4:].argmax())],  # 注意y1, y2顺序
#                                  f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
#                                  attack_result_img)
#
#         # 保存预测结果图像
#         attack_result_img = cv2.cvtColor(attack_result_img, cv2.COLOR_RGB2BGR)  # 转换回BGR格式

#2  风格迁移攻击####
        gan_overimg=combined_img
        # 将gan_overimg转换回[0, 255]范围并转换为uint8
        gan_overimg_bgr = cv2.cvtColor(gan_overimg, cv2.COLOR_RGB2BGR)  # 转换为BGR格式以正确显示

        # 将gan_overimg_bgr转换回RGB格式
        gan_overimg_rgb = cv2.cvtColor(gan_overimg_bgr, cv2.COLOR_BGR2RGB)
        # 归一化到[0, 1]
        gan_overimg_rgb = np.float32(gan_overimg_rgb) / 255.0
        # 调整尺寸和填充
        gan_overimg_rgb, _, _ = letterbox(gan_overimg_rgb, new_shape=(640, 640), scaleFill=False, scaleup=True)
        # 转换为模型输入格式
        gan_overimg_tensor = torch.from_numpy(np.transpose(gan_overimg_rgb, axes=[2, 0, 1])).unsqueeze(0).to(
            self.device)

        # 进行预测
        with torch.no_grad():  # 确保不计算梯度
            pred = self.model(gan_overimg_tensor)[0][0]
            pred = self.post_process(pred, gan_overimg_rgb.shape[:2])

        # 将预测结果转换回原始图像尺寸
        gan_attack_result_img = cv2.cvtColor(gan_overimg_bgr, cv2.COLOR_BGR2RGB)
        # gan_attack_result_img = cv2.resize(gan_attack_result_img, (ori_w, ori_h))

        # 绘制检测框
        for data in pred:
            data = data.cpu().detach().numpy()
            x1, y1, x2, y2 = data[:4].astype(int)
            # 确保坐标不越界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(gan_attack_result_img.shape[1], x2), min(gan_attack_result_img.shape[0], y2)
            self.draw_detections([x1, y1, x2, y2], self.colors[int(data[4:].argmax())],
                                 f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                                 gan_attack_result_img)

        # 保存预测结果图像
        gan_attack_result_img = cv2.cvtColor(gan_attack_result_img, cv2.COLOR_RGB2BGR)  # 转换回BGR格式
        cv2.imwrite('headmap/GanAttack.jpg', combined_img)

        # 显示结果图像
        cv2.imshow('Gan Attack Result Image', gan_attack_result_img)


        ### headmap
        # cv2.imshow('Original Image', original_img)
        cv2.imshow('Detected Image', detected_img)
        cv2.imshow('Heatmap', cam_image)

        ### attack
        cv2.imwrite('headmap/attack_result.jpg', gan_attack_result_img)
        cv2.imshow('Attack Result Image', gan_attack_result_img)





        cv2.waitKey(0)
        cv2.destroyAllWindows()
def get_params():
    params = {
        'weight': 'runs/train/best.pt',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [15, 19, 22, 25],
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.2,  # 0.2
        'ratio': 1.0,  # 0.02-0.1
        'show_box': False,  # 不需要绘制框请设置为False
        'renormalize': False  # 需要把热力图限制在框内请设置为True
    }
    return params



# 需要安装grad-cam==1.4.8


# 调用函数
if __name__ == '__main__':
    model = rtdetr_heatmap(**get_params())
    img_path = 'dataset/images/val/000614.jpg'  # 替换为您的图片路径
    model(img_path)