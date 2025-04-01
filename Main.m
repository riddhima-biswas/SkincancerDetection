clc;
tic; % Start timing

% Optimized PSO Hyperparameters
pso_particles = 6; % Reduced from 40
pso_iterations = 20; % Reduced from 100
pso_imsize = 74;
pso_cnnbatch = 4; % Increased batch size for efficiency
pso_cnnepochs = 10; % Reduced from 200
pso_input = 'PSOINPUT';
pso_target = 'PSOTARGET';
pso_parallel = false; % Enable parallel for speed

% Optimized CNN Hyperparameters
final_imsize = 500;
final_cnnbatch = 4; % Increased batch size
final_cnnepochs = 20; % Reduced from 1000
final_input = 'FINALINPUT';
final_target = 'FINALTARGET';

% Enable Parallel Computing
if pso_parallel && isempty(gcp('nocreate'))
    parpool;
end

% Run PSO Optimization
disp("Running PSO-based Optimization...");
[pso_bestpos, bestloss] = custPSO(pso_particles, pso_iterations, pso_imsize, ...
                                   pso_cnnbatch, pso_cnnepochs, pso_input, pso_target, pso_parallel);

% Run CNN Training with Best PSO Parameters
disp("Training Final CNN...");
[loss] = createCNN(pso_bestpos, pso_imsize, pso_cnnbatch, pso_cnnepochs, pso_input, pso_target, true);
save('finalinput.mat', 'pso_bestpos');
save('finaloutput.mat', 'loss');
toc; % End timing
disp("Best PSO Position:");
disp(pso_bestpos);
figure;
plot(loss);
title("Training Loss Over Epochs");
xlabel("Epochs");
ylabel("Loss");
disp("Final Training Completed!");
