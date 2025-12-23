# 4DST ResDense Model with Prenormalization

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, MaxPooling3D, Concatenate, Activation, Conv2D, AveragePooling3D, Conv3DTranspose, Permute, Add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import GroupNormalization, Lambda
import numpy as np
    
# ==== Helper function for normalization and activation ====
def apply_gn_relu(x):
    x = GroupNormalization(groups=-1, axis=-1)(x)
    return relu(x)

def unet_4dst_resdense_prenorm(input_shape, num_channels, num_classes, kernel_initializer='glorot_uniform'):
    
    input1 = Input(shape=input_shape)
    
    #=========================Shared layer instances====================================
    
    # Contracting path
    conv1a = Conv3D(32, (3, 3, 3), padding='same')
    conv1b = Conv3D(32, (3, 3, 3), padding='same')
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))
    
    conv2a = Conv3D(64, (3, 3, 3), padding='same')
    conv2b = Conv3D(64, (3, 3, 3), padding='same')
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))
    
    conv3a = Conv3D(128, (3, 3, 5), padding='same')
    conv3b = Conv3D(128, (3, 3, 5), padding='same')
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))
    
    # Bottleneck
    bottleneck1 = Conv3D(256, (3, 3, 7), padding='same')
    bottleneck2 = Conv3D(256, (3, 3, 7), padding='same')
    
    # Expanding path
    up1 = UpSampling3D(size=(2, 2, 2))
    conv4a = Conv3D(128, (3, 3, 5), padding='same')
    conv4b = Conv3D(128, (3, 3, 5), padding='same')
    
    up2 = UpSampling3D(size=(2, 2, 2))
    conv5a = Conv3D(64, (3, 3, 3), padding='same')
    conv5b = Conv3D(64, (3, 3, 3), padding='same')
    
    up3 = UpSampling3D(size=(2, 2, 1))
    conv6a = Conv3D(32, (3, 3, 3), padding='same')
    conv6b = Conv3D(32, (3, 3, 3), padding='same')
    
    # Final layers
    collapse_conv = Conv3D(32, (1, 1, 24), strides=(1, 1, 2), padding='valid')
    final_conv = Conv3D(1, (1, 1, 1)) # no activation for residual connection
    
    #==============================Split input by slices================================
    
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=4), name='split_tensor')(input1)
    
    in_0 = Lambda(lambda x: tf.squeeze(x, axis=-1), name='squeeze_0')(split_tensors[0])
    in_1 = Lambda(lambda x: tf.squeeze(x, axis=-1), name='squeeze_1')(split_tensors[1])
    in_2 = Lambda(lambda x: tf.squeeze(x, axis=-1), name='squeeze_2')(split_tensors[2])
    # Pre-norm
    in_0 = GroupNormalization(groups=-1, axis=-1)(in_0)
    in_1 = GroupNormalization(groups=-1, axis=-1)(in_1)
    in_2 = GroupNormalization(groups=-1, axis=-1)(in_2)
    
    #==============================SLice 0==============================================
                               
    x = apply_gn_relu(conv1a(in_0))                             # (192,192,24,32)
    conv1_out = apply_gn_relu(conv1b(x))                     # (192,192,24,32)

    x = pool1(conv1_out)                                     # (96,96,24,32)
    x = apply_gn_relu(conv2a(x))                             # (96,96,24,64)
    conv2_out = apply_gn_relu(conv2b(x))                     # (96,96,24,64)

    x = pool2(conv2_out)                                     # (48,48,12,64)
    x = apply_gn_relu(conv3a(x))                             # (48,48,12,128)
    conv3_out = apply_gn_relu(conv3b(x))                     # (48,48,12,128)

    x = pool3(conv3_out)                                     # (24,24,6,128)
    x = apply_gn_relu(bottleneck1(x))                        # (24,24,6,256)
    x = apply_gn_relu(bottleneck2(x))                        # (24,24,6,256)

    x = up1(x)                                               # (48,48,12,256)
    x = Concatenate(axis=-1)([conv3_out, x])                 # (48,48,12,384)
    x = apply_gn_relu(conv4a(x))                             # (48,48,12,128)
    x = apply_gn_relu(conv4b(x))                             # (48,48,12,128)

    x = up2(x)                                               # (96,96,24,128)
    x = Concatenate(axis=-1)([conv2_out, x])                 # (96,96,24,192)
    x = apply_gn_relu(conv5a(x))                             # (96,96,24,64)
    x = apply_gn_relu(conv5b(x))                             # (96,96,24,64)

    x = up3(x)                                               # (192,192,24,64)
    x = Concatenate(axis=-1)([conv1_out, x])                 # (192,192,24,96)
    x = apply_gn_relu(conv6a(x))                             # (192,192,24,32)
    x = apply_gn_relu(conv6b(x))                             # (192,192,24,32)

    x = apply_gn_relu(collapse_conv(x))                      # (192,192,1,32)
    out_0 = final_conv(x)                                      # (192,192,1,1)
    
    #==============================SLice 1 MID =========================================
    
    x = apply_gn_relu(conv1a(in_1))                             # (192,192,24,32)
    conv1_out = apply_gn_relu(conv1b(x))                     # (192,192,24,32)

    x = pool1(conv1_out)                                     # (96,96,24,32)
    x = apply_gn_relu(conv2a(x))                             # (96,96,24,64)
    conv2_out = apply_gn_relu(conv2b(x))                     # (96,96,24,64)

    x = pool2(conv2_out)                                     # (48,48,12,64)
    x = apply_gn_relu(conv3a(x))                             # (48,48,12,128)
    conv3_out = apply_gn_relu(conv3b(x))                     # (48,48,12,128)

    x = pool3(conv3_out)                                     # (24,24,6,128)
    x = apply_gn_relu(bottleneck1(x))                        # (24,24,6,256)
    x = apply_gn_relu(bottleneck2(x))                        # (24,24,6,256)

    x = up1(x)                                               # (48,48,12,256)
    x = Concatenate(axis=-1)([conv3_out, x])                 # (48,48,12,384)
    x = apply_gn_relu(conv4a(x))                             # (48,48,12,128)
    x = apply_gn_relu(conv4b(x))                             # (48,48,12,128)

    x = up2(x)                                               # (96,96,24,128)
    x = Concatenate(axis=-1)([conv2_out, x])                 # (96,96,24,192)
    x = apply_gn_relu(conv5a(x))                             # (96,96,24,64)
    x = apply_gn_relu(conv5b(x))                             # (96,96,24,64)

    x = up3(x)                                               # (192,192,24,64)
    x = Concatenate(axis=-1)([conv1_out, x])                 # (192,192,24,96)
    x = apply_gn_relu(conv6a(x))                             # (192,192,24,32)
    x = apply_gn_relu(conv6b(x))                             # (192,192,24,32)

    x = apply_gn_relu(collapse_conv(x))                      # (192,192,1,32)
    out_1 = final_conv(x)                                      # (192,192,1,1) # output for refinement
    out_1_sig = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)                  # output for deep supervision
    
    #==============================SLice 2==============================================
    
    x = apply_gn_relu(conv1a(in_2))                             # (192,192,24,32)
    conv1_out = apply_gn_relu(conv1b(x))                     # (192,192,24,32)

    x = pool1(conv1_out)                                     # (96,96,24,32)
    x = apply_gn_relu(conv2a(x))                             # (96,96,24,64)
    conv2_out = apply_gn_relu(conv2b(x))                     # (96,96,24,64)

    x = pool2(conv2_out)                                     # (48,48,12,64)
    x = apply_gn_relu(conv3a(x))                             # (48,48,12,128)
    conv3_out = apply_gn_relu(conv3b(x))                     # (48,48,12,128)

    x = pool3(conv3_out)                                     # (24,24,6,128)
    x = apply_gn_relu(bottleneck1(x))                        # (24,24,6,256)
    x = apply_gn_relu(bottleneck2(x))                        # (24,24,6,256)

    x = up1(x)                                               # (48,48,12,256)
    x = Concatenate(axis=-1)([conv3_out, x])                 # (48,48,12,384)
    x = apply_gn_relu(conv4a(x))                             # (48,48,12,128)
    x = apply_gn_relu(conv4b(x))                             # (48,48,12,128)

    x = up2(x)                                               # (96,96,24,128)
    x = Concatenate(axis=-1)([conv2_out, x])                 # (96,96,24,192)
    x = apply_gn_relu(conv5a(x))                             # (96,96,24,64)
    x = apply_gn_relu(conv5b(x))                             # (96,96,24,64)

    x = up3(x)                                               # (192,192,24,64)
    x = Concatenate(axis=-1)([conv1_out, x])                 # (192,192,24,96)
    x = apply_gn_relu(conv6a(x))                             # (192,192,24,32)
    x = apply_gn_relu(conv6b(x))                             # (192,192,24,32)

    x = apply_gn_relu(collapse_conv(x))                      # (192,192,1,32)
    out_2 = final_conv(x)                                      # (192,192,1,1)
    
    # ==================================================================================
    # ==============================Stack slice segments================================
    # ==================================================================================
    
    seg3d = Lambda(lambda outputs: tf.stack(outputs, axis=3), name='stack_segments')([out_0, out_1, out_2]) # create stacked volume [N,x,y,z,c], all features are kept, stacked in z
    seg3d = Lambda(lambda x: tf.squeeze(x, axis=-2), name='squeeze_3')(seg3d)
    print('seg3d size', seg3d.shape)
    
    # ==================================================================================
    # ==============================MIP and residual path==============================
    # ==================================================================================
    # image input [tmip] -> [conv3d n=2] = [N,x,y,z,c]
    tmip = Lambda(lambda x: tf.reduce_max(x, axis=3))(input1)
    
    x = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(tmip)
    x = GroupNormalization(groups=-1, axis=-1)(x)
    x = relu(x)

    x = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
    x = GroupNormalization(groups=-1, axis=-1)(x)
    x = relu(x) # shape = [N,x,y,z,16]
    
    x = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
    mip_f = GroupNormalization(groups=-1, axis=-1)(x)
    
    resdense_input = Concatenate(axis=-1)([tmip,mip_f, seg3d])
  
    # Layer 1
    x1 = Conv3D(16, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(resdense_input)
    x1 = GroupNormalization(groups=-1, axis=-1)(x1)
    x1 = Activation('relu')(x1)

    # Layer 2
    concat1 = Concatenate(axis=-1)([resdense_input, x1])
    x2 = Conv3D(16, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(concat1)
    x2 = GroupNormalization(groups=-1, axis=-1)(x2)
    x2 = Activation('relu')(x2)

    # Layer 3
    concat2 = Concatenate(axis=-1)([resdense_input, x1, x2])
    x3 = Conv3D(16, kernel_size=3, padding='same', kernel_initializer=kernel_initializer)(concat2)
    x3 = GroupNormalization(groups=-1, axis=-1)(x3)
    x3 = Activation('relu')(x3)

    # Final concat of only the outputs of the dense layers (x1, x2, x3)
    dense_outputs = Concatenate(axis=-1)([x1, x2, x3])

    # Projection to match input channels
    projected = Conv3D(resdense_input.shape[-1], kernel_size=1, padding='same', kernel_initializer=kernel_initializer)(dense_outputs)
    projected = GroupNormalization(groups=-1, axis=-1)(projected)
    # projected = Activation('relu')(projected)

    # Residual connection
    resdense_out = Add()([resdense_input, projected])
    
    # ==================================================================================
    # ==========================Collapse slides=======================================
    # ==================================================================================

    conv_drec = Conv3D(32, (1, 1, 3), strides=(1, 1, 2), padding='valid')(resdense_out) # learnable layer that collapses layers (slides)
    conv_drec = GroupNormalization(groups=-1, axis=-1)(conv_drec)
    conv_drec = Activation('relu')(conv_drec)
  
    conv18 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv_drec) # collapse channels and apply sigmoid
  
    convout = Concatenate(axis=-1)([conv18, out_1_sig]) #concatenate branches as chan 0=final 1=deep_sup
    
    model = Model(inputs=input1, outputs=convout)

    
    return model
