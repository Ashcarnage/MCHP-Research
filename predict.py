import tensorflow as tf 
import os 
import numpy as np 
import imageio
import nrrd
from IPython.display import Image
# from test2 import BinaryIoU,weighted_binary_crossentropy
def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):    
        if image_name.endswith('.npy'):
            # print(image_name)
            image = np.load(os.path.join(img_dir, image_name))
            
            # Ensure all images have the same shape
            if images and image.shape != images[0].shape:
                print(f"Skipping {image_name} due to shape mismatch: {image.shape}")
                continue
            
            images.append(image)
    
    # Only convert to array if there are compatible shapes
    if images:
        images = np.array(images)
    
    return images


def combine_data_with_mask(data, mask, mask_intensity=255):
    """
    Combines grayscale brain scan data with a mask by enhancing or highlighting 
    the intensity of masked regions without affecting the original data.
    
    Parameters:
        data (np.ndarray): 3D numpy array of brain scan data (grayscale).
        mask (np.ndarray): 3D numpy array of mask data (same shape as data).
        mask_intensity (int): Intensity to apply to masked regions.
        
    Returns:
        combined_data (np.ndarray): 3D numpy array with overlay applied.
    """
    # Ensure the data and mask are in the correct shape
    if data.shape != mask.shape:
        raise ValueError("The data and mask volumes must have the same shape.")
    
    # Create a copy of the original data to avoid modifying it directly
    combined_data = data.copy()
    
    # Overlay the mask by setting masked regions to the specified intensity
    combined_data[mask > 0] = mask_intensity

    return combined_data
from skimage.transform import resize 

def visualize_data_gif(data):
    images = []
    # print(data.shape)
    for i in range(data.shape[0]):
        x = data[min(i,data.shape[0]-1),:,:]
        y = data[:,min(i,data.shape[1]-1),:]
        z = data[:,:,min(i,data.shape[2]-1)]

        # Resize each slice to have the same dimensions
        target_shape = z.shape
        x_resized = resize(x, target_shape, preserve_range=True).astype(data.dtype)
        y_resized = resize(y, target_shape, preserve_range=True).astype(data.dtype)
        # img = np.concatenate((x,y,z),axis = 1)
        # img = np.concatenate((x,y_resized,z_resized),axis = 1)
        # img = np.concatenate((x_resized,y,z_resized),axis = 1)
        img = np.concatenate((x_resized,y_resized,z),axis = 1)
        images.append(img)
    imageio.mimsave("./gif/overlayed_brain_scanReal.gif",images,duration=0.01)
    return Image(filename = './gif/overlayed_brain_scanReal.gif',format = 'png')


imgs = os.listdir("/Users/ayushbhakat/Documents/Eureka/prostrate/data")
mask = os.listdir("/Users/ayushbhakat/Documents/Eureka/prostrate/mask")

images = load_img("/Users/ayushbhakat/Documents/Eureka/prostrate/data/", imgs)
masks = load_img("/Users/ayushbhakat/Documents/Eureka/prostrate/mask/",mask)

data, header = nrrd.read("/Users/ayushbhakat/Documents/Eureka/prostrate/PROSTATE DATA/SEGMENTED PROSTATE/P1_segmented Data/P1_T1W Labelled/P1_T1WLabel.nrrd")

from tensorflow.keras.metrics import MeanIoU

# Define custom metric for IoU with threshold
class BinaryIoU(MeanIoU):
    def __init__(self, num_classes=2, threshold=0.5):
        super().__init__(num_classes=num_classes)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Apply threshold to predictions for binary segmentation
        y_pred = tf.cast(y_pred > self.threshold, tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)

def weighted_binary_crossentropy():
    class_weights = {0: 10.0, 1: 40.0}
    # Convert class_weights dictionary to a tensor
    class_weights_tensor = tf.convert_to_tensor(list(class_weights.values()), dtype=tf.float32)
    
    # def loss(y_true, y_pred):
    #     # Flatten the tensors to make it easier to apply the weights
    #     y_true = tf.reshape(y_true, [-1])
    #     y_pred = tf.reshape(y_pred, [-1])

    #     # Compute the binary crossentropy loss
    #     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    #     # Use class weights to adjust the loss
    #     weight_map = tf.gather(class_weights_tensor, tf.cast(y_true, tf.int32))  # Use tensor as lookup
    #     weighted_bce = bce * weight_map

    #     return tf.reduce_mean(weighted_bce)
    def dice_loss(y_true, y_pred, smooth=1e-6):
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Compute Dice coefficient
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        # Return Dice loss
        return 1 - dice

    def custom_loss(y_true, y_pred):
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Weighted Binary Cross-Entropy
        bce = tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)
        weight_map = tf.gather(class_weights_tensor, tf.cast(y_true_flat, tf.int32))
        weighted_bce = bce * weight_map
        weighted_bce_mean = tf.reduce_mean(weighted_bce)

        # Dice Loss
        dice = dice_loss(y_true, y_pred)

        # Combined loss
        combined = weighted_bce_mean + dice

        return combined

    return custom_loss

def create_nrrd_header(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), space="left-posterior-superior"):
    """
    Create an NRRD header dynamically based on a NumPy array and metadata.
    
    Parameters:
        array (numpy.ndarray): Input 3D or 4D array.
        spacing (tuple): Voxel spacing for each dimension (default: isotropic spacing of 1.0).
        origin (tuple): Physical coordinates of the origin (default: (0.0, 0.0, 0.0)).
        space (str): Coordinate system (default: "left-posterior-superior").
    
    Returns:
        dict: A dictionary representing the NRRD header.
    """
    # Verify array dimensionality
    if len(array.shape) not in [3, 4]:
        raise ValueError("NRRD format typically supports 3D or 4D arrays.")

    # Determine the data type
    dtype_to_nrrd = {
        'uint8': 'unsigned char',
        'int8': 'char',
        'uint16': 'unsigned short',
        'int16': 'short',
        'uint32': 'unsigned int',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
    }
    
    nrrd_type = dtype_to_nrrd.get(array.dtype.name)
    if nrrd_type is None:
        raise ValueError(f"Unsupported data type for NRRD: {array.dtype}")

    # Create header
    header = {
        'type': nrrd_type,
        'dimension': len(array.shape),
        'sizes': array.shape,
        'space': space,
        'space directions': [
            [spacing[0], 0, 0],
            [0, spacing[1], 0],
            [0, 0, spacing[2]]
        ],
        'space origin': origin,
        'kinds': ['domain'] * len(array.shape),  # For segmentation, all dimensions are 'domain'
    }
    
    return header


model = tf.keras.models.load_model(
    "/Users/ayushbhakat/Documents/Eureka/prostrate/prostrate4_CNN.h5", 
    compile=False  # Load without compiling the model
)
# LR = 5e-5
# optim = tf.keras.optimizers.Adam(LR)
# loss_fn = weighted_binary_crossentropy()
# model.compile(optimizer = optim, loss=loss_fn, metrics=[tf.keras.metrics.BinaryAccuracy(),BinaryIoU(num_classes=2,threshold = 0.5)])
# pred_mask = model.predict(np.expand_dims(images[2], axis=0))
# pred_mask = np.squeeze(pred_mask, axis=0)
# print("All Values: ",np.unique(pred_mask.astype(np.int32)))
# pred_mask = np.where(pred_mask>0.4,1,0).astype(np.int32)
# header = create_nrrd_header(pred_mask)
# nrrd.write('./pred_mask/output2.nrrd',pred_mask,header)
pred_mask= np.load("/Users/ayushbhakat/Documents/Eureka/prostrate/mask/mask_P3_T2W.npy")
# model = tf.keras.models.load_model("/Users/ayushbhakat/Documents/Eureka/prostrate/prostrate_CNN.h5",custom_objects={"BinaryIoU": BinaryIoU,"weighted_binary_crossentropy": weighted_binary_crossentropy})
gif_images = (images[4]*225).clip(0,255)
gif_images = gif_images.astype(np.uint8)
mask_images = (pred_mask*225).clip(0,255)
mask_images = mask_images.astype(np.uint8)
combined_data = combine_data_with_mask(gif_images,mask_images)
visualize_data_gif(combined_data)
