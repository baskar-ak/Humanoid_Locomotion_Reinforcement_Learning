# CS5180: Reinforcement Learning and Sequential Decision Making 
## Project: Humanoid Locomotion using Reinforcement Learning
### Motivation / Problem Statement:
Humanoid locomotion is the ability of the robots to move like humans, typically walking or running. This remains a challenging problem in robotics and reinforcement learning due to the complexities involved such as – multi joint coordination like head, arms, torso, legs. It is not just making the robot move but maintaining balance and ensuring natural and stable human like movements. Humans are very skilled at maintaining balance while walking, whereas a robot needs to adjust its posture dynamically and stay balanced while walking, especially on uneven surfaces. Moreover, the more degree of freedom a robot has, the harder it is to manage and coordinate the movement of each joint to ensure smooth motion. Solving humanoid locomotion problems will allow robots to assist humans with everyday tasks like carrying objects, helping elderly, etc. They will be able to work in environments such as homes, offices, and hospitals as well as outdoor environments (e.g., rescues missions) providing assistance.<br/><br/>

### Objective:
* Make a humanoid robot walk as fast as possible without falling and maintaining balance. 
* Apply and compare two closely related RL algorithms.<br/><br/>

### Environment Details:
**Humanoid-v4 MuJoCo OpenAI Gym Environment**
* Observation Space: 348 position and velocities of robot's body parts
* Action Space: 17 joints (torque)
* Reward: Healthy reward + Forward reward - Control cost - Contact cost
* Termination: Falling<br/><br/>

### Solution Approach:
<img src="https://github.com/user-attachments/assets/61358ac7-d799-4b82-86f6-c531c8fb26e6" width="750"><br/>
The Humanoid is trained using Soft Actor Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) RL algorithms.<br/><br/>

### Results:
<img src="https://github.com/user-attachments/assets/6e894206-433c-4559-80ef-e5f94581d9b7" width="300">
<img src="https://github.com/user-attachments/assets/f182fc53-4cd6-4801-a13f-fe7e1f817f2f" width="300">
<img src="https://github.com/user-attachments/assets/aa0f3510-377d-4ba3-ba07-c7113519d42c" width="300"><br/>

https://github.com/user-attachments/assets/59c5bc6f-166c-4708-bc3c-fa091fd1b0ef

https://github.com/user-attachments/assets/b30f62c3-d421-475b-96d2-a31dd5d71cf8


### Running the script:
* Open new terminal in Jupyter Notebook<br/>
* For SAC training: `$ python projectScript.py Humanoid-v4 SAC -t`<br/>
* For TD3 training: `$ python projectScript.py Humanoid-v4 TD3 -t`<br/>
* To view logs on Tensorboard: `$ tensorboard --logdir logs`<br/>
* For SAC testing: `$ python projectScript.py Humanoid-v4 SAC -s .\models\SAC_50000.pth`<br/>
* For TD3 testing: `$ python projectScript.py Humanoid-v4 TD3 -s .\models\TD3_50000.pth`<br/>
