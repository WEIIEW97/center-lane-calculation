import cv2
import numpy as np
import skimage.morphology as skm
import os
from tqdm import tqdm
from main import find_middle_lane_rowwise
# img_path = "/home/william/extdisk/data/cybathlon/footpath_bag_data/sam_seg_mask/sam_seg_1703138027.273827.png"
# mask = cv2.imread(img_path)[:,:,0]

# selem = skm.disk(7)
# dilated_mask = skm.dilation(mask, selem)
# erosion_mask = skm.erosion(dilated_mask, selem)
# # dilated_mask = skm.dilation(dilated_mask, selem)

# cv2.imshow("di", erosion_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def iterate():
    rootdir = "/home/william/extdisk/data/cybathlon/footpath_bag_data/"
    closing_folder = "sam_seg_mask_close"
    dilation_folder = "sam_seg_mask_dilate"
    source_folder = "sam_seg_mask"

    closing_folder_vis = "sam_seg_mask_close_vis"
    dilation_folder_vis = "sam_seg_mask_dilate_vis"

    cf = rootdir+closing_folder
    df = rootdir+dilation_folder

    cfv = rootdir+closing_folder_vis
    dfv = rootdir+dilation_folder_vis


    os.makedirs(cf, exist_ok=True)
    os.makedirs(df, exist_ok=True)

    os.makedirs(cfv, exist_ok=True)
    os.makedirs(dfv, exist_ok=True)

    all_files = [f for f in os.listdir(rootdir+source_folder) if os.path.isfile(rootdir+source_folder+f"/{f}")]
    # selem = skm.disk(7)

    for file in tqdm(all_files, desc="Processing closing operations"):
        img_path = cf+f"/{file}"
        mask = cv2.imread(img_path)
        coords = find_middle_lane_rowwise(mask[:,:,0])
        for x, y in coords:
            cv2.circle(mask, (x, y), 1, (255, 192, 203), 1)
        cv2.imwrite(cfv+f"/{file}", mask)


    for file in tqdm(all_files, desc="Processing dilation operations"):
        img_path = df+f"/{file}"
        mask = cv2.imread(img_path)
        coords = find_middle_lane_rowwise(mask[:,:,0])
        for x, y in coords:
            cv2.circle(mask, (x, y), 1, (255, 192, 203), 1)
        cv2.imwrite(dfv+f"/{file}", mask)





