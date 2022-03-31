## WDASnet

Authors: Yi Yang, Hangting Chen, Pengyuan Zhang   
Key Laboratory of Speech Acoustics and Content Understanding, Institute of Acoustics, Chinese Academy of Sciences, Beijing, China    

We propose a Weighted-Direction-Aware speech Separation network (WDASnet) to achieve a DOA-assisted speech separation on sparsely overlapped mixtures in a multi-people meeting environment. First, based on the Convolutional Recurrent Neural Network (CRNN) DOA-estimation model, we provide a variant system by leveraging a weighted-pooling block which reduces the influence of silent and interference speaker frames. Second, we achieve an end-to-end utterance-wise DOA-estimation. Prior VAD, pre- or post-processing is not needed. Third, we take a deep look of our system into multi-people meeting environment. Fourth, we analyze the advantages and limitations of this model.   


### Demos

mixture (overlap=0.5)    

<img src="figs/mix-overlap0.5.png" width="40%">
<audio src="wav/mix-overlap0.5.wav" controls="controls"> </audio>    
    
ground-truth    
<p float="left">
  <img src="figs/Oracle-s1.png" width="40%" />
  <img src="figs/Oracle-s2.png" width="40%" /> 
</p>
<audio src="wav/Oracle-s1.wav" controls="controls"> </audio> <audio src="wav/Oracle-s2.wav" controls="controls"> </audio>       
    
BLSTM separation with average-pooling DoA estimation (sdr=5.72dB)    
<p float="left">
  <img src="figs/Average-pooling-s1.png" width="40%" />
  <img src="figs/Average-pooling-s2.png" width="40%" /> 
</p>
<audio src="wav/Average-pooing-s1.wav" controls="controls"> </audio> <audio src="wav/Average-pooing-s2.wav" controls="controls"> </audio>    
    
BLSTM separation with oracle AF (sdr=13.92dB)    
<p float="left">
  <img src="figs/BLSTM-AF-s1.png" width="40%" />
  <img src="figs/BLSTM-AF-s2.png" width="40%" /> 
</p>
<audio src="wav/BLSTM-AF-s1.wav" controls="controls"> </audio> <audio src="wav/BLSTM-AF-s2.wav" controls="controls"> </audio>    
    
BLSTM separation with the proposed weighted-pooling DoA estimation (WDASnet) (sdr=14.62dB)    
<p float="left">
  <img src="figs/Proposed-s1.png" width="40%" />
  <img src="figs/Proposed-s2.png" width="40%" /> 
</p>
<audio src="wav/Proposed-s1.wav" controls="controls"> </audio> <audio src="wav/Proposed-s2.wav" controls="controls"> </audio>    
    
### Example1    
    
This illustration is the same with Fig.2 in the paper.    
Visualization of the estimated weight. Three pairs of blocks from left to right refer to speech of target speaker, silent frames and overlapped speech, respectively.    

<img src="figs/Fig2.png" width="40%">
<audio src="wav/overlap0.2_4381-1296.wav" controls="controls"> </audio>

### Example2    
    
Example2 shows the variation of estimated weights in condition of different sirs (-5,0,5dB)    

<img src="figs/overlap0.4-sx408-si1993.png" width="40%">
<p float="left">
  <img src="figs/-5.png" width="30%" />
  <img src="figs/0.png" width="30%" /> 
  <img src="figs/5.png" width="30%" /> 
</p>
<audio src="wav/overlap0.4-sx408-si1993 -5dB.wav" controls="controls"> </audio> <audio src="wav/overlap0.4-sx408-si1993 0dB.wav" controls="controls"> </audio> <audio src="wav/overlap0.4-sx408-si1993 5dB.wav" controls="controls"> </audio>    
    
### Contact    
    
If you have any advice or questions, please feel free to contact me!

