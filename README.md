# cogs106-Simulate-and-recover

Project Title: Simulation and Recovery of the EZ Diffusion Model

This is the final project of the class cogs106. The goal of this project is to using our previous knowledge we learned during class to evaluafte how well the EZ diffusion model can recover its own parameters. Diffution models, widely used in cognitive psychology, describle the decision-making processes associated with accuracy(binary choice tasks) and response time. They assume that evidence accumulation occurs over time unitl a threshold is reached, leading to a decision. Additionally, EZ diffusion model is a simplified version of it designed for ease of parameter estimation.

For this project, we randomly selected realistic model parameters within predefined ranges. The boundary separation (a) was set between 0.5 and 2, the drift rate (v) ranged from 0.5 to 2, and the non-decision time (t) was chosen between 0.1 and 0.5. Using these parameters, we generated simulated data and applied the EZ diffusion model to recover them. To ensure reliable statistical results, we repeated this simulate-and-recover process 1000 times for each condition. Additionally, we varied the sample size (N) across three levels: 10, 40, and 4000, leading to a total of 3000 simulations.

Boundary separation (a) represents the amount of evidence required to make a decision. A larger value of indicates a more cautious decision strategy, leading to longer response times but fewer errors. Drift rate (v) reflects the speed and direction of evidence accumulation. A higher drift rate suggests that the decision-maker is processing information more efficiently and is more likely to reach a correct decision quickly. Finally, non-decision time (t) accounts for the time spent on processes that are not related to evidence accumulation, such as sensory encoding and motor response execution.

In our analysis, we computed estimation bias (Bias) and mean squared error (MSE). Ideally, the bias should average to zero, and MSE should decrease as the sample size increases. Since we randomly selected model parameters within predefined ranges, the results are different from times to times. For example, in one run, I got the results like this:
Starting EZ Diffusion Model Simulate-and-Recover Experiment...

N      Bias_a      Bias_v      Bias_t0     MSE_a     MSE_v     MSE_t0
 
10 0.214403 0.126333 -0.023423 0.374084 0.66112 0.024606

N     Bias_a      Bias_v      Bias_t0     MSE_a      MSE_v     MSE_t0
 
40 0.02684 0.021995 0.000666 0.041122 0.144515 0.004862

N     Bias_a      Bias_v      Bias_t0      MSE_a      MSE_v     MSE_t0
   
4000 -0.000304 -0.000261 0.000009 0.000156 0.001178 0.000045

Our results showed that when N=10, the recovered parameters had relatively large biases and high MSE values compared to N=40 and N=4000. As N increased to 40, all biases decreased(for boundary separation, it decreases from 0.214403 to 0.02684; for drift rate, it decreases from 0.126333 to 0.021995; for non-decision time, it decreases from 0.023423 to 0.000666), and all MSE were reduced accordingly. When N reached 4000, all the estimated biases were close to zero, and all MSE dropped significantly. These findings align with theoretical expectations, demonstrating that as sample size increases, the EZ diffusion model improves in its ability to recover its own parameters.

In summary, this project confirms that the EZ diffusion model can accurately estimate its own parameters under ideal conditions. However, in small sample cases (e.g., N=10), the estimation accuracy was significantly lower due to increased variability and a greater likelihood of parameter misestimation. The recovered values in these cases deviated substantially from the true parameters, leading to larger estimation errors. As the sample size increased to N=40, the accuracy of parameter recovery improved, with biases decreasing and estimation errors becoming more stable. When N reached 4000, the recovery process became highly reliable, with biases converging toward zero and mean squared errors reaching minimal levels. This trend highlights the importance of larger sample sizes when applying the EZ diffusion model, as greater data availability leads to more precise and consistent parameter recovery. These findings reinforce the model's validity as a cognitive modeling tool while emphasizing its limitations in small-sample conditions, where estimation noise can be considerable.
