# PORTEND_Performance_Model
This is a repository for paper "PORTEND: A Joint Performance Model for Partitioned Early-Exiting DNNs" published in ...

Abstract

The computation and storage requirements of Deep Neural Networks (DNNs) make them challenging to deploy on edge devices, which often have limited resources. Conversely, offloading DNNs to cloud servers incurs high communication overheads. Partitioning and early exiting are commonly proposed approaches for reducing such computational costs and improving inference speed. Current work, however, treats these approaches as separate steps, thus missing the opportunity to jointly tailor these complementary approaches to application requirements and edge network topology. Moreover, existing performance models are oversimplified, addressing two-tier networks and neglecting common communication intricacies on edge networks such as de(serialization) and data transmission overheads.
We present PORTEND, a novel performance model that considers partitioning and early exiting holistically. By quickly estimating the inference time and accuracy of partitioned and early exiting DNNs, PORTEND allows joint optimization of both the DNN’s early exit points and partitioning of its layers across a multi-tier edge network, while meeting application constraints such as minimal accuracy. PORTEND can thus improve deploy- ment over prior approaches, for example, reducing end-to-end latency by 49% for a ResNet-110 deployment on a 3-tier topology.


folders:

- aws_profiling_results: this folder contains profiling results on multiple AWS EC2 instances (a1.large, a1.medium, g4dn.xlarge, g5.xlarge, m4.large). We profiled computation latency of 5 models (EfficientNetB0, EfficientNet7, ResNet20, ResNet110, VGGish) on these instances in a partition-wise manner. We used these numbers in the performance model to predict inference latency.

- instance_folder: this folder contains the code for agents (source_agent, inference_agent, destination_agents) running on AWS instances for automatic accelerated inference. source_agent gets the request and starts the inference. inference_agent does the inference and decides to exit or stay, and destination agent receives the inference result and stores them for further analysis. there's also a script that configures the TC (traffic control) ip route table on AWS EC2 instances. By configuring the routing table we can connect the instances in a desired topology for experiments.

- performance_model: this folder contains the code for some test case studies in the paper. for special scenarios.

- plots: plotting codes for paper.

- setup_experiment_scripts: setup script starts/restarts the AWS instances automatically. It configures the instances based on the user's preference, synchronizes instances with Chrony, runs required TC (traffic control) and ip route table settings, and copies the required files to instances.
the experiment script runs after the setup script. It receives parameters like model, dataset, buffer size, and sample number from users and starts the experiment. It runs source_agent, inference_agent, and destination agent on AWS instances.

- weights: some of the trained model weights (not all of them).
