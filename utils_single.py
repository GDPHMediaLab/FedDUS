import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
sigmoid = nn.Sigmoid()
softmax = nn.Softmax()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    def print_numclass(self):
        print(self.n_classes)
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(imagec2, label, net, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    # image_C2,label =  imagec2.squeeze(0).cpu().detach().numpy(),label.squeeze(0).cpu().detach().numpy()
    image_C2,label=imagec2,label
    # weight = net.outc.conv.weight.cuda()
    if len(image_C2.shape) == 3:
        prediction = np.zeros_like(label)
        # feature_maps = np.zeros_like(label)
        for ind in range(image_C2.shape[0]):
            slice_C2 = image_C2[ind, :, :]
            x, y = slice_C2.shape[0], slice_C2.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice_C2 = zoom(slice_C2, (patch_size[0] / x, patch_size[1] / y), order=0)  # previous using 0
            inputc2 = torch.from_numpy(slice_C2).unsqueeze(0).unsqueeze(0).float().cuda()
            # print(inputc2.shape)
            net.eval()
            with torch.no_grad():
                # outputs,feature_map = net(inputc2)
                outputs = net(inputc2)
                # feature_map = weight * feature_map
                # feature_map = torch.sum(feature_map,dim=1)
                # feature_map = feature_map.squeeze(0)
                # feature_map[feature_map < 0] =0
                # print("*"*20)

                outputs = sigmoid(outputs.squeeze(1))
                # outputs[outputs < 0.9] = 0
                outputs =  (outputs>0.5).float()
                out = outputs.squeeze(0)
                # out = out.squeeze(0)
                out = out.cpu().detach().numpy()
                # feature_map = feature_map.cpu().detach().numpy()
                
                # plt.imsave("/home/dongxy/data_2T/project/Pytorch-UNet-master/model_out/0830_single/output_test/heatmap_ZS2"+'/'+case +"_"+str(ind)+".jpg",feature_map,cmap = "Blues_r",format="jpg")
                
                # feature_map = show_cam_on_image(out,feature_map)
                # feature_map = feature_map[:,:,0] + feature_map[:,:,1] + feature_map[:,:,2]
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    # feature_map = zoom(feature_map, (x / patch_size[0], y / patch_size[1]), order=0)
                    slice_C2 = zoom(slice_C2, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # feature_map_ = Image.fromarray(feature_map).convert("RGB")
                # feature_map_.save("/home/dongxy/data_2T/project/Pytorch-UNet-master/model_out/0830_single/output_test/heatmap_ZS2"+'/'+case +"_"+str(ind)+".jpg")
                # feature_map[feature_map<0]=0
                # feature_map = (feature_map-np.min(feature_map))/(np.max(feature_map)-np.min(feature_map))
                # feature_map_ = show_cam_on_image(slice_C2,feature_map)
                # feature_map_ = cv2.cvtColor(feature_map_, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("/home/dongxy/data_2T/project/Pytorch-UNet-master/model_out/0830_single/output_test/111"+'/'+case +"_"+str(ind)+".jpg",feature_map_)
                # # print(feature_map.shape)
                # feature_map = Image.fromarray(feature_map).convert("RGB")
                # plt.show(feature_map)
                # plt.imsave("/home/dongxy/data_2T/project/Pytorch-UNet-master/model_out/0830_single/output_test/heatmap_ZS2"+'/'+case +"_"+str(ind)+"jpg",feature_map,cmap = "Blues_r",format="jpg")
                # cv2.imencode(".jpg",feature_map)[1].tofile("/home/dongxy/data_2T/project/Pytorch-UNet-master/model_out/0830_single/output_test/heatmap_ZS2"+'/'+case +"_"+str(ind)+"jpg")

                # feature_maps[ind] = feature_map

    metric_list = []
    for i in range(1,2):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        imgC2_itk = sitk.GetImageFromArray(image_C2.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # feature_itk = sitk.GetImageFromArray(feature_maps.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        imgC2_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        # feature_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        # # # sitk.WriteImage(feature_itk, test_save_path + '/'+case + "_feature.nii.gz")
        sitk.WriteImage(imgC2_itk, test_save_path + '/'+ case + "_img2.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
def show_cam_on_image(img, mask):
    # mask = 
    # img = Image.fromarray(img.astype("uint8"))
    # img = img.convert("RGB")

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_BONE)
    heatmap = np.float32(heatmap) / 255
    # img = np.float32(img) / 255
    

    cam = 1000*heatmap + np.float32(img)
    # cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
def Norm2d(array):
    xlength,ylength = array.shape
    ymax = 255
    ymin = 0
    xmax = max(map(max,array))
    xmin = min(map(min,array))
    for i in range(xlength):
        for j in range(ylength):
            if((xmax-xmin) != 0):
                array[i][j] = round(((ymax-ymin)*(array[i][j]-xmin)/(xmax-xmin))+ymin)
    return array
def Norm3d(array):
    array1 = 255 * ((array - array.min()) / (array.max()-array.min()))
    return array1

