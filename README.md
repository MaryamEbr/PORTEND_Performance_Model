# PORTEND_Performance_Model
This is a repository for paper "PORTEND: A Joint Performance Model for Partitioned Early-Exiting DNNs" published in ...

Abstract

The computation and storage requirements of Deep Neural Networks (DNNs) make them challenging to deploy on edge devices, which often have limited resources. Conversely, offloading DNNs to cloud servers incurs high communication overheads. Partitioning and early exiting are commonly proposed approaches for reducing such computational costs and improving inference speed. Current work, however, treats these approaches as separate steps, thus missing the opportunity to jointly tailor these complementary approaches to application requirements and edge network topology. Moreover, existing performance models are oversimplified, addressing two-tier networks and neglecting common communication intricacies on edge networks such as de(serialization) and data transmission overheads.
We present PORTEND, a novel performance model that considers partitioning and early exiting holistically. By quickly estimating the inference time and accuracy of partitioned and early exiting DNNs, PORTEND allows joint optimization of both the DNN’s early exit points and partitioning of its layers across a multi-tier edge network, while meeting application constraints such as minimal accuracy. PORTEND can thus improve deploy- ment over prior approaches, for example, reducing end-to-end latency by 49% for a ResNet-110 deployment on a 3-tier topology.
