# ADC-MADDPG

#### 介绍
{This study focuses on the resilient path planning of
a data-collection unmanned aerial vehicle (UAV) swarm within
a scenario that efficiently gathers information from Internet of
Things (IoT) nodes. Note that the UAV swarm is confronted
with the challenge posed by collaborative intelligent jamming
adversaries employing highly efficient encoder-decoder enhanced
communication. Equipped with adaptive learning capabilities and
the prioritized experience replay module, these adversaries can
dynamically coordinate the deployment of the jamming signal to
degrade the communication quality between UAV swarms and
IoT nodes. To counteract the aforementioned attacks, the UAV
swarm utilizes a path planning algorithm that not only ensures
resilience but also facilitates the efficient collection of data from
various IoT nodes. This is accomplished by taking into account
various constraints, including kinematic limitations, airspace
restrictions, and mission-specific deadlines. Inspired by the multi-
agent deep deterministic policy gradient (MADDPG) algorithm,
a resilient reinforcement learning scheme for the UAV swarm
is put forward by this work. In addition, this study improves
the MADDPG algorithm in two ways. First, an attention-critic
module is introduced by integrating an attention mechanism into
the critic. Second, a dual-critic module is added to avoid over-
scoring. Employing these two modules, the attention-dual-critic-
MADDPG algorithm is proposed. The simulation part thoroughly
evaluates the resilience of different defensive algorithms under
various scenarios, including the condition without jamming
attacks, the one under fixed attacks, and the one under intelligent
jamming attacks. It also demonstrates that the proposed modules
can bring improvements in both efficiency and resilience.}


#### 安装教程

pip -r requirements.txt

#### 使用说明

1.  训练 main_openai.py

2.  测试 test.py

3. 参数

    ·主要参数1 --jammer_act store_true类型，添加该参数将会在训练过程中允许干扰机运动

    ·主要参数2 --train_model (m: MADDPG, a: AC-MADDPG, d: DC-MADDPG, adc: ADC-MADDPG)

