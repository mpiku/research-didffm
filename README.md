## Data-Driven Inverse Dynamics for Flexure Manipulator
This repository stores datasets and scripts complementary to paper M. Pikulinski*, P. Malczyk, R. Aarts, (2024), _Data-Driven Inverse Dynamics Modeling Using Neural-Networks and Regression-Based Techniques_, Multibody System Dynamics (under review).

*Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

### Abstract
This research proposes a novel approach for the residual modeling of inverse dynamics employed to control a real robotic device. Specifically, we use techniques based on linear regression for residual modeling while a nominal model is discovered by physics-informed neural networks such as the Lagrangian Neural Network and the Feedforward Neural Network. We introduce an efficient online learning mechanism of the residual models that utilizes rank-one updates based on the Sherman-Morrison formula. This enables faster adaptation and updates to effects not captured by the neural networks. While the time complexity of updating the model is comparable to other successful learning methods, the method excels in prediction complexity, which depends solely on the model dimension.

We propose two online learning strategies: a weighted approach that gradually diminishes the influence of past measurements on the model and a windowed approach that sharply excludes the oldest data from impacting the model. We explore the relationship between these strategies, offering recommendations for parameter selection and practical application. Special attention is given to optimizing the computation time of the weighted approach when recomputation techniques are implemented, which results in comparable or even lower execution times of the weighted controller than the windowed one. Additionally, we assess other methods, such as the Woodbury identity, QR decomposition, and Cholesky decomposition, which can be implicitly used to update the model.

We empirically validate our approach using real data from a 2-degree-of-freedom flexible manipulator, demonstrating consistent improvements in feedforward controller performance.

### Basic configuration
The main script, ```didModel.m```, presents hands-on experience building and predicting errors using the Data-Driven Inverse Dynamics (DID) approach. Additionally, the script ```plotCompare.m``` draws plots that compare the results of the different control structures discussed in the original paper (see Fig. 12). The plotting script should be run after computations from the ```didModel.m``` script finish.

### Dataset
The included dataset is a part of the measurements collected for E. Heerze, B. Rosic, R. Aarts, (2024). _Feedforward Control for a Manipulator with Flexure Joints Using a Lagrangian Neural Network_. In K. Nachbagauer, & A. Held (Eds.), Optimal Design and Control of Multibody Systems: Proceedings of the IUTAM Symposium (pp. 130-141). (IUTAM Bookseries; Vol. 42). Springer Nature. Advance online publication. https://doi.org/10.1007/978-3-031-50000-8_12. We would like to sincerely thank the authors for letting us publish it.

The dataset consists of three trajectories: _Random 1_, _Random 2_, _Spiral_. The data is organized in the following structure
- ```data.x``` - Measured state
- ```data.r``` - Desired trajectory
- ```data.f``` - Actuators' force
   - ```data.f.fb``` - Feedback force (tracking the desired trajectory results in measured states)
   - ```data.f.ident.r``` - Identification-based feedforward (FF) on the desired traj.
   - ```data.f.nn``` - Neural-network-based FF
      - ```data.f.nn.lnn``` - LNN-based FF
         - ```data.f.nn.lnn.x``` - LNN-based FF values on the measured traj.
         - ```data.f.nn.lnn.r``` - LNN-based FF values on the desired traj.
      - ```data.f.nn.fnn``` - FNN-based FF
         - ```data.f.nn.fnn.x``` - FNN-based FF values on the measured traj.
         - ```data.f.nn.fnn.r``` - FNN-based FF values on the desired traj.