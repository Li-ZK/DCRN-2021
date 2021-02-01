import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

import spectral
from utils import find_max
from sklearn.preprocessing import scale
import copy

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')



def draw_feature_map(handle, model = None, dataObj = None, groundtruth = None, iscomposite = False, input_val = None):
    
    # API to draw composite_image/color map, only provide supportive color and label for KSC, UP, and Salinas.
    ##########################################################################################################
    # handle: Clarify exact which data set is used, i.e., ksc, pavia_U, salians.
    # model: Used only when predicting color map, clarify exact model instance to use for prediction.
    # dataObj: Used only when predicting color map, clarify exact dataObj instance to use for prediction.
    # groundtruth: Used for generating color map for groundtruth, clarify ground truth matrix to plot.
    # iscomposite: Used for indicating whether a composite color image will be generated. Bool instance.
    # input_val: Used only when iscomposite == True, clarify the input 3-d cubes used for plot.
    ##########################################################################################################
    # Sample Usage:
    # Plot composite image for KSC: 
    #       draw_feature_map('ksc', iscomposite = True, input_val = ksc_matrix)
    # Plot ground truth map for KSC:
    #       draw_feature_map('ksc', groundtruth = ksc_gt_matrix)
    # Plot prediction color map for KSC:
    #       draw_feature_map('ksc', model = model, dataObj = dataObj)
    ##########################################################################################################
    
    color_dict = {'ksc': [[0, 0, 0], [140, 67, 46], [0, 0, 255], [255, 100, 0], [0, 255, 123], 
                          [164, 75, 155], [101, 174, 255], [118, 254, 172], [60, 91, 112], 
                          [255, 255, 0], [101, 193, 60], [255, 0, 255], [100, 0, 255], [0, 172, 254]],
                  'pavia_U': [[0, 0, 0], [192, 192, 192], [0, 255, 0], [0, 255, 255], [0, 128, 0],
                              [255, 0, 255], [165, 82, 41], [128, 0, 128], [255, 0, 0], [255, 255, 0]],
                  'salinas': [[0, 0, 0], [0, 0, 255], [255, 100, 0], [0, 255, 134], [150, 70, 150],
                              [100, 150, 255], [60, 90, 114], [255, 255, 125], [255, 0, 255], 
                              [100, 0, 255], [1, 170, 255], [0, 255, 0], [175, 175, 82], 
                              [100, 190, 56], [140, 67, 46], [115, 255, 172], [255, 255, 0]]}
    tick_dict = {'ksc': ('Background', 'Scrub', 'Willow_S', 'Cabbage_P', 
                    'Cabbage_O', 'Slash_P', 'Oak_H', 'Hardwood_S',
                    'Graminoid_M', 'Spartina_M', 'Cattail_M', 'Salt_M',
                    'Mud_F', 'Water'),
                 'pavia_U': ('Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                             'Metal sheets', 'Bare soil', 'Bitumen', 
                             'Self-Blocking Bricks', 'Shadows'),
                 'salinas': ('Background', 'Weeds_1', 'Weeds_2', 'Fallow', 'Fallow_P',
                             'Fallow_S', 'Stubble', 'Celery', 'Grapes',
                             'Soil', 'Corn', 'Lettuce_4wk', 'Lettuce_5wk',
                             'Lettuce_6wk', 'Lettuce_7wk','Vinyard_U', 'Vinyard_T')}
    band_dict = {'ksc':(50, 28, 25), 'pavia_U': (46, 27, 10), 'salinas': (25, 19, 8)}
    
    if iscomposite:
        band = band_dict[handle]
        image = np.array([input_val[:, :, band[0]], input_val[:, :, band[1]], input_val[:, :, band[2]]]).transpose(1, 2, 0)
        plt.figure(figsize=(9, 9))
        spectral.imshow(image)
        plt.axis('off')
        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(handle+'_composite', dpi = 300)

    if groundtruth is not None:
        color = color_dict[handle]
        color = np.array(color)
        spectral.imshow(classes = groundtruth.astype(int), figsize = (9, 9), colors = color)
        if iscomposite:
            tick = tick_dict[handle]
            bar = plt.colorbar()
            bar.set_ticks(np.linspace(0, len(tick) - 1, len(tick)))
            bar.set_ticklabels(tick)
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(handle+'_gt', dpi = 300)
    else:
        color = color_dict[handle]
        temp = copy.deepcopy(dataObj.data)
        temp = temp.reshape((-1, temp.shape[-1]))
        temp = scale(temp)
        info = temp.reshape((dataObj.data.shape[0], dataObj.data.shape[1], -1))
        paddingData = np.zeros((info.shape[0] + 2 * dataObj.hwz + 1, info.shape[1] + 2 * dataObj.hwz + 1, info.shape[2]))
        paddingData[dataObj.hwz: info.shape[0] + dataObj.hwz, dataObj.hwz: info.shape[1] + dataObj.hwz] = info[:, :]
        y_out = np.zeros_like(info)
        y_out = np.zeros_like(dataObj.groundtruth)
        try:
            with tqdm(range(dataObj.groundtruth.shape[0])) as t:
                for x in t:
                    for y in range(dataObj.groundtruth.shape[1]):
                        if dataObj.groundtruth[x, y] == 0:
                            continue
                        X_in = torch.from_numpy(paddingData[x: x + 2 * dataObj.hwz + 1, y : y + 2 * dataObj.hwz + 1, :].transpose(2,0,1).astype(np.float32)).to(device)
                        X_in = X_in.reshape(1, 1, X_in.shape[0], X_in.shape[1], X_in.shape[2])
                        y_temp = model(X_in)
                        y_out[x,y] = int(find_max(y_temp)) + 1
                spectral.imshow(classes = y_out, colors = color, figsize=(9, 9))
                plt.axis('off')
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        pass
