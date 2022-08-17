# fall-detection-under-partial-occlusion-using-C3D-neural-network
One of the most possible dangers that older people face in their daily lives is falling. A fall is an accident that results in disability and crippling. If the person is not able to inform others about their condition after a fall, this dilemma gets more critical, and the person may even lose their life; therefore, the existence of intelligent and efficient systems for detecting falls in the elderly is essential. As deep learning and computer vision have developed in recent years, researchers have used these techniques to detect falls. Among the computer vision challenges, occlusion is one of the biggest challenges of these systems and degrades their performance to a considerable extent. Here, we provide an effective solution for occlusion handling in vision-based fall detection systems. We propose a fall detection algorithm under partial occlusion using the C3D neural network presented in [fall-detection-under-partial-occlusion-using-C3D-neural-network](https://github.com/sarakhalili/fall-detection-under-partial-occlusion-using-C3D-neural-network) to tackle this problem.
## Main Idea
Occlusion can lead a model to learn unrealistic features or force the model to ignore the blocked spatial area in un-occluded samples. In other words, body characteristics and sources of occlusion simultaneously affect the distribution of features. As a result, occluded data inevitably has a negative effect on the extraction of normal data features; therefore, it is necessary to design an effective learning strategy to optimize the model. To this end, we present weighted model training by defining a new cost function. The new cost function is defined by L=Ln+(λ*(n/o)*Lo), where Ln is the classification error of normal samples and Lo is the classification error of occluded samples[1]. This framework can be applied to various fall detection systems. Here we use C3D neural network for feature extraction.
One of the most successful neural networks in motion detection and activity recognition is C3D network.  This network is capable of extracting the temporal and spatial features, useful for body motion detection, human activity localization, and human-scene interaction tasks [2] In order to apply weighted training according to L=Ln+(λ*(n/o)*Lo), we define two stages in each iteration of training. In the first stage, a batch of normal video sequences are fed to the network, and their classification error is calculated based on the current network parameters. In the second stage, the occluded samples generated from the same normal samples are fed to the network and their classification error is calculated. The final error is then calculated according to (2), and this error goes back to update the parameters. This process continues until we reach the maximum number of iterations.
## Feature Extractor
The video sequences, as they are, cannot be used as the inputs of C3d model. We therefore divide each video sequence into consecutive non-overlapping segments. First, the input images should be resized to 112*112 pixels using the "resize" code. For each video folder, a text file should be created which contains 3 numbers indicating the start and stop of falling frames and the total number of frames for each video. These numbers are essential for defining the label of each input segment. 
Feature extraction is done in "extract" function. Two parameters in "option" are defined called sampling-rate and stride. The sampling-rate parameter reduces the number of frames. The stride parameter specifies the distance between the first frames of two consecutive video segments. We use "fc6" layer as the feature extractor whose output dimension is 4096*1. Therefore, after extracting the features by the C3D extractor, 4096 features are generated for every 16 consecutive frames with their corresponding labels
## Training Phase
The produced feature extractor is generated using the "C3D_Sport1M_weights_keras_2.2.4.h5" which is not optimized for fall detection. As a result we ,first, need to train the network for distinguishing falling from other daily life activities and then use the weights of the trained model for our feature extractor. In order to do so, we add a layer with two neurons to the end of the C3D structure. The purpose of this layer is classifing falling and non-falling activities. We also only train fully connected layers of the network and leave the rest unchanged. Moreover, the training of fully connected layers are done using the weighted model training method. This part is done by "training_C3D" code. 

## Detection phase 
Finally, we use the SVM classifier to classify the extracted features. SVM is a supervised machine learning algorithm that is very popular due to its high accuracy. The classification task is done in “cross sample” code and again SVM is trained with weighted loss function.
### Any Questions?
If you had any questions about using this code, Please contact [Sara Khalili](sarahkhalili89@gmail.com)

### Refrences
[1]	C. Shao et al., "Biased Feature Learning for Occlusion Invariant Face Recognition," in IJCAI, 2020, pp. 666-672.

[2]	P. Viola, M. J. Jones, and D. Snow, "Detecting pedestrians using patterns of motion and appearance," International Journal of Computer Vision, vol. 63, no. 2, pp. 153-161, 2005.

[3]	H. M. Amirreza Razmjoo. (2018). Fall-Detection_Dynamical_haar_Cascade. Available: https://github.com/amrn9674/Fall-Detection_Dynamical_haar_Cascade/tree/master
