# DefNet
By Winson Ye</br>
Supervised By Dr. Horea Ilies

<h1> Abstract </h1>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The goal of this project was to develop software that could compute the deformation of geometries under loads in real time. This task is nontrivial for real time graphics, and even harder for haptics applications. Traditional approaches have sacrificed accuracy for speed or vice versa. However, if one trains an AI on FEA data, they could get a dramatic speedup while still preserving high accuracy. Most recently, similar works have used deep neural networks (DNNs) to accomplish this task, but traditional DNNs are slow to train and execute. MLPs have been used before for deflection analysis, but without the benefit of recent advances in training neural networks. **Thus, this project distinguishes itself from previous works by aiming to provide an extremely lightweight and accurate framework (DefNet) that uses MLPs to predict general nonlinear deformations for any object under applied loads.** While MLPs may not seem like a natural choice for complicated nonlinear structural analyses, the preliminary research has shown that one can achieve good results with simple MLPs, especially when combined with state of the art approaches to training modern neural networks like He initialization. Two analyses were investigated: compression and twisting. Please see the video for a demonstration: [Real Time Deformation Prediction Preliminary Results](https://youtu.be/vy1nESJ7vLQ)

<h1> Methods </h1>

**Software Used**
1. Visualization: ParaView
2. FEA Analysis: FEBio
3. Coding Neural Networks: Tensorflow
4. Data analysis: MATLAB and Python

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Three MLPs were developed: UXNet, UYNet, and UZNet. Each one was responsible for one of the three Cartesian displacement axes: u<sub>x</sub>, u<sub>y</sub>, and u<sub>z</sub>. Each MLP was designed to approximate the function mapping loads to displacements. The input and output were both specified as matrices with the same general structure for all nets, and differed depending only on the FEA analysis. 41 training samples corresponding to different frames were used for both FEA analyses. To create the different frames, FEBio solved for the displacements at different time steps according to a load curve. All nets were trained using a training load curve first, and then tested for generalization performance on a different load curve. For compression, the training set was based on moving the rigid body down into the beam, and then from left to right. The test set was based on moving the rigid body halfway to the right and halfway down into the beam. For twisting, the training set was based on twisting the beam for one revolution, and the test set was based on twisting the beam for half a revolution. In terms of data representation, for compression, a single input vector had two inputs, where the first entry was the X displacement and the second entry was the Z displacement. The entire input matrix was of shape [2, 41]. For the output, a single output vector had 5029 outputs, one for each displacement value for each node. The entire output matrix was of shape [5029, 41]. For twisting, a single input vector had one input, which was the rotation around the X axis. The whole input matrix was of shape [1, 41]. The output vector had 764 entries, one for each displacement value for each node. THe whole output matrix was of shape [764, 41].</br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The nets were all MLPs with a single hidden layer, and differed only on their hyperparameters, such as number of hidden neurons, learning rate, etc. **All nets used Adam optimization with He initialization and leaky ReLU activation functions. In addition, the input features were scaled to be between [-1, 1]**. These architectural decisions have allowed for much greater success in training the nets. The author decided to use three different MLPs in order to make it easier to parallelize execution when the nets are deployed. Additionally, it is easier (better results) for 3 different MLPs to approximate the mapping between loads and displacements of the X, Y, and Z directions respectively than to have a single MLP approximate the function for displacements in all directions.

<h1> Results </h1>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After being trained with only 41 training samples, the MLPs were able to memorize the data very effectively. While the nets generalized well for the twisting scenario, there was some noticeable error in the Z direction for the compression scenario when the nets were asked to extrapolate, perhaps due to the fact that the FEA mesh for compression was made of 5029 nodes, while the twisting mesh was only made of 764. To get an idea of the speedup, it takes an average of 7.579 seconds (real time over 5 trials) for a script to assemble all the results from UXNet, UYNet, and UZNet needed to generate a 41 frame animation for the compression training data set. Meanwhile, in FEBio, it takes an average of 51.8 seconds to generate all the data for the 41 frame animation. **Thus, there is an 85% speedup, with accuracy that in some cases is almost indistinguishable from the ground truth (please see video for demonstration).** It is important to note that these are pessimestic esimates of the MLPs because no optimization took place. For real time performance, UXNet, UYNet, and UZNet can run on separate cores of the machine simultaneously, the nets themselves can be optimized for deployment, GPUs can be used to speed up calculations, and special workstations designed to minimize system interrupts can be used. More testing must be conducted on the speed and generalizability of this framework before it is adopted. Applications for this technology include immersive VR applications, surgical simulation, and high fidelity scientific simulations. 

<h1> References </h1>

1. Physics-Driven Neural-Networks-Based Simulation System for Nonlinear Deformable Objects (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3357955/#R55)
2. Stress Field Prediction Using Conv Nets (https://arxiv.org/pdf/1808.08914.pdf)
3. Neural Networks in Mechanics of Structures and Materials (https://www.sciencedirect.com/science/article/pii/S0045794901000839)
4. Real Time Deformation Using Finite Element and Neural Networks in VR (https://www-sciencedirect-com.ezproxy.lib.uconn.edu/science/article/pii/S0168874X06000461)
5. Determination of Deformation and Failure Properties of Ductile Properties Using Small Punch Test and Neural Networks (https://www.sciencedirect.com/science/article/pii/S0927025603001484)
6. Determination of Constitutive Properties from Spherical Indentation Data Using Neural Networks. (https://www.sciencedirect.com/science/article/pii/S0167663618306367)
7. Framework for Predicting 3D prostate Deformation in Real Time (https://onlinelibrary.wiley.com/doi/full/10.1002/rcs.1493)
8. Fast Soft Tissue Def Using Mass Spring Model for Maxillof. Surgery (https://link.springer.com/chapter/10.1007/978-3-540-30136-3_46)
9. GANNs for Real Time Robot Navigation (https://arxiv.org/pdf/1804.05928.pdf)
10. Adversarial VAE for Real Time Prediction of Object Deformation (https://arxiv.org/pdf/1805.00328.pdf)
11. Machine Learning Approach for Real Time Modeling of Tissue Deformation in Image Guided Neurosurgery (https://www.sciencedirect.com/science/article/pii/S0933365716304687)
12. Real time haptic simulation of tissue deformation (https://www.academia.edu/27884798/Real-Time_Haptic_Simulation_of_Soft_Tissue_Deformation)
13. Vector Field Approximation Using Neural Networks
(https://link-springer-com.ezproxy.lib.uconn.edu/content/pdf/10.1007%2F978-3-540-74690-4_73.pdf
https://ieeexplore.ieee.org/document/1389970)
14. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
15. Tensorflow 1.x Deep Learning Cookbook


