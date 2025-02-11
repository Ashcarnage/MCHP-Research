import nibabel as nib
import numpy as np
import nrrd
import glob 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
test_image = nib.load("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/P1_segmented Data/P1_T1W Labelled/P1_T1WImage.nii.gz").get_fdata()

t1_list = sorted(glob.glob("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/*/*T1*/*.nii.gz"))
t2_list = sorted(glob.glob("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/*/*T2*/*.nii.gz"))
dw_list = sorted(glob.glob("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/*/*DW*/*.nii.gz"))
mask_list = sorted(glob.glob("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/*/*T2*/*.nrrd"))

for img,d in enumerate(t2_list):
    print("prepairing image and masks... for  : ",img)

    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1=nib.load(t1_list[img]).get_fdata()
    temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
   
    temp_image_dw=nib.load(dw_list[img]).get_fdata()
    temp_image_dw=scaler.fit_transform(temp_image_dw.reshape(-1, temp_image_dw.shape[-1])).reshape(temp_image_dw.shape)

    temp_mask,_ = nrrd.read(mask_list[img])
    temp_mask=temp_mask.astype(np.uint8)
    # print(temp_image_t1.shape)
    # print(np.unique(temp_mask))
    # temp_image_t2 = temp_image_t2[88:344, 88:344, 3:23]#[103:328,103:328,3:23]
    if img==2:
       temp_image_t2 = temp_image_t2[64:344, 64:344, 3:23] 
    else:
       temp_image_t2 = temp_image_t2[88:344, 88:344, 3:23]
    # temp_combined_images = np.stack([temp_image_dw, temp_image_t1, temp_image_t2], axis=3)
    # temp_combined_images=temp_combined_images[52:308, 88:344, 3:23]
    # temp_mask = temp_mask[88:344, 88:344, 3:23]
    if img==2:
       temp_mask = temp_mask[64:344, 64:344, 3:23] 
    else:
       temp_mask = temp_mask[88:344, 88:344, 3:23] 
    print(temp_image_t2.shape)
   #  print(temp_image_t2)
    print(temp_mask.shape)
    print('/Users/ayushbhakat/Documents/Eureka/prostrate/data/image_'+str(d[-18:-12])+'.npy')
    np.save('/Users/ayushbhakat/Documents/Eureka/prostrate/data/image_'+str(d[-18:-12])+'.npy', temp_image_t2)
    np.save('/Users/ayushbhakat/Documents/Eureka/prostrate/mask/mask_'+str(d[-18:-12])+'.npy', temp_mask)





      