# mva-signals


#### Remarks:
- The best all models do in the classification of signals with 0dB snr is 50%, which means that the model learns something meaningful, even in this severe case. Keeping those samples does not affect the performance of the classification of the signals with higher SNR.

- Training with one SNR class shows a slight degradation in the calssification performance.
- Until now, frequency domain (STFT) models show mediocre performance.

- How the data augmentation affect the performance? 