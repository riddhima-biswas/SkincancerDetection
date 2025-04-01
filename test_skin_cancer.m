% Clear workspace and command window
clc;
clear;
close all;

% Add paths to required folders
addpath("C:\Users\KIIT\Downloads\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Models");
addpath("C:\Users\KIIT\Downloads\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Data");
addpath("C:\Users\KIIT\Downloads\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Code")

% Load the pre-trained models
try
    alexnetData = load("C:\Users\KIIT\Downloads\skincancer\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Code\skin_disease_alexnet.mat");
    resnetData = load("C:\Users\KIIT\Downloads\skincancer\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Code\skin_disease_resnet50.mat");

    netAlexNet = alexnetData.netTransfer; % Extract the trained AlexNet
    netResNet = resnetData.netTransfer;   % Extract the trained ResNet50

    disp("Models Loaded Successfully!");
catch
    error("Error loading models. Ensure the file paths are correct and the models are available.");
end

% Load and preprocess the test image
testImagePath = "C:\Users\KIIT\Downloads\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Data\Test\Test\melanoma\ISIC_0000049.jpg";
if exist(testImagePath, 'file')
    testImage = imread(testImagePath);
else
    error("Test image not found. Please check the path.");
end

% Resize images to match input size of both networks
inputSizeAlexNet = netAlexNet.Layers(1).InputSize(1:2);
inputSizeResNet = netResNet.Layers(1).InputSize(1:2);

resizedImageAlexNet = imresize(testImage, inputSizeAlexNet);
resizedImageResNet = imresize(testImage, inputSizeResNet);

% Perform classification using both models
[YPredAlexNet, scoreAlexNet] = classify(netAlexNet, resizedImageAlexNet);
[YPredResNet, scoreResNet] = classify(netResNet, resizedImageResNet);

% Display results
disp(['AlexNet Prediction: ', char(YPredAlexNet)]);
disp(['ResNet50 Prediction: ', char(YPredResNet)]);

% Display the original and preprocessed images
figure;
subplot(1,3,1);
imshow(testImage);
title("Original Image");

subplot(1,3,2);
imshow(resizedImageAlexNet);
title("Preprocessed for AlexNet");

subplot(1,3,3);
imshow(resizedImageResNet);
title("Preprocessed for ResNet50");

% Convert to grayscale and apply Fuzzy C-Means segmentation
grayImage = rgb2gray(testImage);
data = double(grayImage(:));
numClusters = 2; % Background and skin cancer
[centers, U] = fcm(data, numClusters);
[~, maxIndex] = max(U);
clusteredImage = reshape(maxIndex, size(grayImage));
skinCancerMask = clusteredImage == 2; % Adjust based on clustering result

% Display segmented image
figure;
imshow(skinCancerMask);
title("Segmented Skin Cancer Area (Fuzzy C-Means)");
