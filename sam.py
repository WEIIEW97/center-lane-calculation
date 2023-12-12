from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os
import sys
# if sys.platform == "linux":
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def generate_sam_masks(ckpt_path, img, device):
    sam = sam_model_registry["vit_h"](ckpt_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    return masks

def use_sam_predictor(input_point, input_label, ckpt_path, img, device):
    sam = sam_model_registry["vit_h"](ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    return masks, scores, logits


def main():
    image = cv2.imread('/home/william/extdisk/Data/extract_footpath/image_rgb/1702370792.414842.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = '/home/william/Downloads/sam_vit_h_4b8939.pth'
    device = "cuda"

    # masks = generate_sam_masks(sam_checkpoint, image, device)
    # print(len(masks))
    # print(masks[0].keys())

    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 

    # input_points = np.array([[293, 122], [278, 373]])
    # input_label = np.array([1 ,1])

    input_point = np.array([[323, 353], [338, 283], [372, 207], [300, 159]])
    input_label = np.array([1, 1, 1, 1])
    masks, scores, logits = use_sam_predictor(input_point, input_label, sam_checkpoint, image, device)

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()  
    seg_mask = masks[2].astype(np.uint8)
    seg_mask *= 255
    cv2.imwrite("output/sam_footpath_1702370792.414842.jpg", seg_mask)
    


if __name__ == "__main__":
    main()
