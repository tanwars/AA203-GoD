# Drone Racing in Airsim
#### Siddharth Tanwar, Adam Dai, Shubh Gupta

This repository contains code for end-of-quarter project in course AA203, Optimal and Learning based control (Spring 2020). In this project, we aim to explore methods of autonomous drone racing by utilizing the AirSim Drone Racing Labframework [1] used in NeurIPS 2019 to host the Game of Drones competition. We implemented two approached to perform waypoint following for a quadrotor: a non-learning based and a learning based approach.  The non-learning based approach comprises of two parts:  minimum snap trajectory generation [3] and MPC controller [4] to follow the nominal path. The learning based method uses an Actor-Critic architecture for velocity control with clipped double Q-learning objective function [5, 6]. 

## References
[1]  R. Madaan, N. Gyde, S. Vemprala, M. Brown, K. Nagami, T. Taubner, E. Cristofalo, D. Scaramuzza, M. Schwa-ger, and A. Kapoor, “Airsim drone racing lab,”arXiv preprint arXiv:2003.05654, 2020.
[2]  S. Shah, D. Dey, C. Lovett, and A. Kapoor, “Airsim: High-fidelity visual and physical simulation for autonomousvehicles,” inField and service robotics.    Springer, 2018, pp. 621–635.
[3]  C. Richter, A. Bry, and N. Roy, “Polynomial trajectory planning for aggressive quadrotor flight in dense indoorenvironments,” inRobotics Research.    Springer, 2016, pp. 649–666.
[4]  M. Kamel, M. Burri, and R. Siegwart, “Linear vs nonlinear MPC for trajectory tracking applied to rotary wingmicro aerial vehicles,” Nov. 2016.[5]  H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” inThirtieth AAAIconference on artificial intelligence, 2016.
[6]  S. Fujimoto, H. Van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,”arXiv preprint arXiv:1802.09477, 2018.
