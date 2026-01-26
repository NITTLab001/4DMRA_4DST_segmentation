PCASL based 4DMRA intracranial vessel segmentation  
Original paper: https://onlinelibrary.wiley.com/doi/10.1002/mrm.70173

**4DST Model Architecture**
![Figure_1_github](https://github.com/user-attachments/assets/04e89ba5-dcf5-4970-8f52-56154ace9421)  
(A) 3DST U-Net architecture (B) 4DST U-Net showing the parallel configured 3DST U-Nets and ResDense refinement module 

**Example** 
![4DST_MIP_github](https://github.com/user-attachments/assets/9d491827-a259-4207-83c1-ffdbf84c8d96)

**Software Versions:**   
Python 3.10.14  
Keras 3.1.1  
Tensorflow 2.16.1  

**Recommended ASL-MRI**  
PASL ASL based labeling  
Dynamic time range ~ 2s  
Temporal resolution ~ 100ms  
The dynamic time range should measure both inflow and outflow dynamics   

**Usage**
For 4DST inference
1) z-score normalize 
2) Prepare your data to match the original implementation (192×192×3×24) (XYZT)
3) Get the pretrained 4DST .h5 weights from the latest GitHub release
