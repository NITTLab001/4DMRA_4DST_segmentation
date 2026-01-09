import tensorflow as tf
from tensorflow.keras import backend as K

#%% Multiple shared layers loss function

end_loss_ratio = 0.7
mid_loss_ratio = 1-end_loss_ratio

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) # binary cross entropy

def dice_coefficient_msl(y_true, y_pred, smooth=1):
  
  y_preds = y_pred[:,:,:,:,0] #final layer
  y_true_f = K.flatten(K.cast(y_true, 'float32'))
  y_pred_f = K.flatten(K.cast(y_preds, 'float32'))

  intersection = K.sum(y_true_f * y_pred_f)
  union = K.sum(y_true_f**2) + K.sum(y_pred_f**2)
  dice1 = (2. * intersection + smooth) / (union + smooth)
  
  y_preds = y_pred[:,:,:,:,1] #intermediate middle slice
  y_true_f = K.flatten(K.cast(y_true, 'float32'))
  y_pred_f = K.flatten(K.cast(y_preds, 'float32'))

  intersection = K.sum(y_true_f * y_pred_f)
  union = K.sum(y_true_f**2) + K.sum(y_pred_f**2)
  dice2 = (2. * intersection + smooth) / (union + smooth)
  
  dice = (end_loss_ratio*dice1) + (mid_loss_ratio*dice2)   # end/center loss ratio for DSC ///////////////////////
  return dice    

def bce_msl(y_true, y_pred, alpha=0.5):
    finalmask = y_pred[:, :, :, :, 0]  # get final slice
    midmask = y_pred[:, :, :, :, 1]  # get middle slice
    bce1 = bce(y_true, finalmask)
    bce2 = bce(y_true, midmask)
    bce_msl = (end_loss_ratio*bce1)+(mid_loss_ratio*bce2) # end/center loss ratio for BCE /////////////////////
    return bce_msl    

def bce_dice_loss_msl(y_true, y_pred, alpha=0.5):
  finalmask = y_pred[:,:,:,:,0]# get final slice
  midmask = y_pred[:,:,:,:,1]# get middle slice
 
  bce1 = bce(y_true,finalmask)
  bce2 = bce(y_true,midmask)
  bce_msl = (end_loss_ratio*bce1)+(mid_loss_ratio*bce2)     # end/center loss ratio for BCE ////////////////////
  return (1-alpha)*bce_msl+alpha*dice_loss_msl(y_true, y_pred)

def dice_loss_msl(y_true, y_pred):
  return 1 - dice_coefficient_msl(y_true, y_pred)
