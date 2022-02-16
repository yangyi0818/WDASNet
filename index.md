## WDASnet

Authors: Yi Yang, Hangting Chen, Pengyuan Zhang
Key Laboratory of Speech Acoustics and Content Understanding, Institute of Acoustics, Chinese Academy of Sciences, Beijing, China

We propose a weighted-direction-aware speech separation network (WDASnet) to achieve a DOA-assisted speech separation on sparsely overlapped mixtures in a multi-people meeting environment. 
First, based on the Convolutional Recurrent Neural Network (CRNN) DOA-estimation model, we provide a variant system by leveraging a weighted-pooling block which reduces the influence of silent and interference speaker frames. 
Second, we achieve an end-to-end utterance-wise DOA-estimation. No prior VAD, pre-extraction with adaptation utterance information or post-processing is needed. 
Third, we take a deep look of our system into multi-people meeting environment.


### Demo (Two speakers Separation)

mixture (overlap=0.5)

<img src="/mix-overlap0.5.png" width="40%">
<audio src="/mix-overlap0.5.wav" controls="controls"> </audio>

oracle
<p float="left">
  <img src="/Oracle-s1.png" width="40%" />
  <img src="/Oracle-s2.png" width="40%" /> 
</p>
<audio src="/Oracle-s1.wav" controls="controls"> </audio> <audio src="/Oracle-s2.wav" controls="controls"> </audio>

BLSTM sep (sdr=2.39dB)
<p float="left">
  <img src="/BLSTM-s1.png" width="40%" />
  <img src="/BLSTM-s2.png" width="40%" /> 
</p>
<audio src="/BLSTM-s1.wav" controls="controls"> </audio> <audio src="BLSTM-s2.wav" controls="controls"> </audio>

BLSTM sep with average-pooling DoA estimation (sdr=5.72dB)
<p float="left">
  <img src="/Average-pooling-s1.png" width="40%" />
  <img src="/Average-pooling-s2.png" width="40%" /> 
</p>
<audio src="/Average-pooing-s1.wav" controls="controls"> </audio> <audio src="Average-pooing-s2.wav" controls="controls"> </audio>

BLSTM sep with oracle AF (sdr=13.92dB)
<p float="left">
  <img src="/BLSTM-AF-s1.png" width="40%" />
  <img src="/BLSTM-AF-s2.png" width="40%" /> 
</p>
<audio src="/BLSTM-AF-s1.wav" controls="controls"> </audio> <audio src="BLSTM-AF-s2.wav" controls="controls"> </audio>

BLSTM sep with proposed weighted-pooling DoA estimation (WDASnet) (sdr=14.62dB)
<p float="left">
  <img src="/Proposed-s1.png" width="40%" />
  <img src="/Proposed-s2.png" width="40%" /> 
</p>
<audio src="/Proposed-s1.wav" controls="controls"> </audio> <audio src="Proposed-s2.wav" controls="controls"> </audio>

### Example1

This illustration is the same with Fig.2 in the paper.
Visualization of the estimated weight. Three pairs of blocks from left to right refer to speech of target speaker, silent frames and overlapped speech, respectively.

<img src="/Fig2.png" width="40%">
<audio src="/overlap0.2_4381-1296.wav" controls="controls"> </audio>

### Example2

Example2 shows the variation of estimated weights in condition of different sirs (-5,0,5dB)

<img src="/overlap0.4-sx408-si1993.png" width="40%">
<p float="left">
  <img src="/-5.png" width="30%" />
  <img src="/0.png" width="30%" /> 
  <img src="/5.png" width="30%" /> 
</p>
<audio src="/overlap0.4-sx408-si1993 -5dB.wav" controls="controls"> </audio> <audio src="/overlap0.4-sx408-si1993 0dB.wav" controls="controls"> </audio> <audio src="/overlap0.4-sx408-si1993 5dB.wav" controls="controls"> </audio>

### Questions

If you have any advice or questions, please contact us.
