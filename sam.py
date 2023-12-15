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

def cv_click_event(image):
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', click_event)
    

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return points

def build_sam_cv_pipeline(image, sam_checkpoint, device="cuda"):
    points = cv_click_event(image)
    num_points = len(points)
    if num_points is not None or num_points != 0:
        input_points = np.array(points)
        input_labels = np.ones(num_points, dtype=np.uint8)

        masks, scores, logits = use_sam_predictor(input_points, input_labels, sam_checkpoint, image, device)

        best_score = 0
        selected_idx = -1
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score > best_score:
                best_score = score
                seletced_idx = i
        seg_mask = masks[selected_idx].astype(np.uint8)
        seg_mask *= 255
        return seg_mask

def main():
    sam_checkpoint = '/home/william/Downloads/sam_vit_h_4b8939.pth'
    device = "cuda"

    rgb_dir = '/home/william/extdisk/Data/extract_footpath/image_rgb'
    save_dir = '/home/william/extdisk/Data/extract_footpath/seg_mask'
    os.makedirs(save_dir, exist_ok=True)
    all_rgb_files = [f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_root_dir, f))]

    for f in all_rgb_files:
        image = cv2.imread(os.path.join(rgb_dir, f))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_mask = build_sam_cv_pipeline(image, sam_checkpoint, device)
        if seg_mask is not None:
            write_name = f"sam_seg_{f}"
            cv2.imwrite(os.path.join(save_dir, write_name), seg_mask)
    
    print("done!")

def test():
    image = cv2.imread('/home/william/extdisk/Data/extract_footpath/image_rgb/1702370798.032749.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = '/home/william/Downloads/sam_vit_h_4b8939.pth'
    device = "cuda"

    seg_mask = build_sam_cv_pipeline(image, sam_checkpoint, device)
    cv2.imshow("highest score segmentation", seg_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    test()
