# PAtechKDD

## Upload results only 

Follow the naming format algorithm_prior_parameter_new
算法名字+是否固定第一轮+具体参数+新老环境
Use a csv or txt or md file to report results.

### Tasks
- kangys: 树模型+固定action1+调参（c[5000,6000],ration[0.6,0.7,0.8],step[11,6],新老环境）
- zhuzw: PPO+PG[新老环境，是否固定action1]
- zuolei: DDPG[新老环境，是否固定action1]
- cony: 树模型+固定action1+调参（c[2000,3000,4000,5000,6000],ration[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],step[11,6],新老环境）
- tingting：DQN[新老环境，是否固定action1]
- kw:遗传算法+[新老环境, 固定action1 + rules 同年0-1.8 后四sum<4.5 同一policy邻年不同 步长0.2 6代2人]
