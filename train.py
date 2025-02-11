# import os
# import numpy as np


# def load_img(img_dir, img_list):
#     images=[]
#     for i, image_name in enumerate(img_list):    
#         if (image_name.split('.')[1] == 'npy'):
#             print(image_name)
#             image = np.load(img_dir+img_list[i])
                      
#             images.append(image)
#     images = np.array(images)
    
#     return(images)
# imgs = os.listdir("/Users/ayushbhakat/Documents/Eureka/prostrate/data")
# images = load_img("/Users/ayushbhakat/Documents/Eureka/prostrate/data/",imgs)
# print(images)

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

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

imgs = os.listdir("/Users/ayushbhakat/Documents/Eureka/prostrate/data")
mask = os.listdir("/Users/ayushbhakat/Documents/Eureka/prostrate/mask")

images = load_img("/Users/ayushbhakat/Documents/Eureka/prostrate/data/", imgs)
masks = load_img("/Users/ayushbhakat/Documents/Eureka/prostrate/mask/",mask)
# print("images shape : ",images.shape)
# print("masks shape: ",masks.shape)
import imageio
from IPython.display import Image


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
    imageio.mimsave("./gif/overlayed_brain_scan1.gif",images,duration=0.01)
    return Image(filename = './gif/overlayed_brain_scan1.gif',format = 'png')

gif_images = (images[0]*225).clip(0,255)
gif_images = gif_images.astype(np.uint8)
combined_data = combine_data_with_mask(gif_images,masks[0])
visualize_data_gif(combined_data)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            Y = Y.astype(np.float32)

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

train_datagen = imageLoader("/Users/ayushbhakat/Documents/Eureka/prostrate/data/",imgs,"/Users/ayushbhakat/Documents/Eureka/prostrate/mask/",mask,2)
#img, msk = train_datagen.__next__() 
import segmentation_models_3D as sm
# Dice loss
dice_loss =  sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss  = dice_loss + (1* focal_loss)
metrics = ['accuracy']#,sm.metrics.IOUScore(threshold = 0.5)]

LR = 5e-5
optim = tf.keras.optimizers.Adam(LR)
batch_size = 1
steps_per_epoch = len(imgs)//batch_size*8
from model  import simple_unet_model

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
    

def weighted_binary_crossentropy(class_weights):
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

# Example class weights
class_weights = {0: 15.0, 1: 120.0}  # Adjust according to your needs
loss_fn = weighted_binary_crossentropy(class_weights)

model = simple_unet_model(IMG_HEIGHT=256, 
                          IMG_WIDTH=256, 
                          IMG_CHANNELS=20, 
                        #   IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=loss_fn, metrics=[tf.keras.metrics.BinaryAccuracy(),BinaryIoU(num_classes=2,threshold = 0.5)])
print(model.summary())
# print(model.input_shape)
# print(model.output_shape)
# class_weights = {0: 1.0, 1: 30.0}
with tf.device('/GPU:0'):
    history = model.fit(train_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs = 100,
                        verbose = 1,
                        #class_weight=class_weights
                        )
model.save('prostrate_CNN.h5')
import matplotlib.pyplot as plt
#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training loss
ax1.plot(epochs, loss, 'y', label='Training loss')
# ax1.plot(epochs, val_loss, 'r', label='Validation loss') # Uncomment for validation loss
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot training accuracy
acc = history.history['binary_accuracy']
# val_acc = history.history['val_accuracy']  # Uncomment for validation accuracy
ax2.plot(epochs, acc, 'y', label='Training accuracy')
# ax2.plot(epochs, val_acc, 'r', label='Validation accuracy')  # Uncomment for validation accuracy
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

# Show the plot
plt.tight_layout()
plt.show()
plt.show()