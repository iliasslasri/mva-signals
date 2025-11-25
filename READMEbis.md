# mva-signals


#### Remarks:
- The best all models do in the classification of signals with 0dB snr is 50%, which means that the model learns something meaningful, even in this severe case. Keeping those samples does not affect the performance of the classification of the signals with higher SNR.

- Training with one SNR class shows a slight degradation in the calssification performance.
- Until now, frequency domain (STFT) models show mediocre performance.

- How the data augmentation affect the performance? 
    - Data augmentation accelerates the generalization of the model, so training is reduced, we achieve better performance on the validation set in 1/3 of the training time.
    - We get 3% better in accuracy on the test set with 1/3 of the training time.


- To train the model we combine the train set and samples set, total: 30200 sample.

# Results on test set
### SNR in model input & without augmentation

==================== RESULTS ====================

Test Loss:      0.2786
Test Accuracy:  86.57%

---- Accuracy per Class ----
Class 0: 91.81%
Class 1: 88.42%
Class 2: 85.48%
Class 3: 88.32%
Class 4: 88.06%
Class 5: 77.21%

---- Accuracy per SNR ----
SNR 0 dB: 46.53%
SNR 10 dB: 98.15%
SNR 20 dB: 99.96%
SNR 30 dB: 100.00%

---- Confusion Matrix ----
[[1558   80   57    0    0    2]
 [  83 1459  108    0    0    0]
 [  81  159 1413    0    0    0]
 [   0    0    0 1460  170   23]
 [   0    0    0  168 1483   33]
 [   0    0    0  146  233 1284]]

=================================================

### SNR in model input & with augmentation
==================== RESULTS ====================

Test Loss:      0.3068
Test Accuracy:  84.65%

---- Accuracy per Class ----
Class 0: 87.51%
Class 1: 75.33%
Class 2: 88.99%
Class 3: 87.72%
Class 4: 78.68%
Class 5: 89.66%

---- Accuracy per SNR ----
SNR 0 dB: 40.62%
SNR 10 dB: 96.73%
SNR 20 dB: 99.44%
SNR 30 dB: 100.00%

---- Confusion Matrix ----
[[1485   48  164    0    0    0]
 [  21 1243  386    0    0    0]
 [  15  167 1471    0    0    0]
 [   0    0    0 1450   18  185]
 [   0    0    0  139 1325  220]
 [   0    0    0  121   51 1491]]

=================================================

