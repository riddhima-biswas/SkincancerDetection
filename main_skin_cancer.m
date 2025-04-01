clc; clear; close all;

%% 🔹 Load Training and Validation Data
imds = imageDatastore("C:\Users\KIIT\Downloads\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Data\Train", ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into Training and Validation (80% Train, 20% Validation)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% 🔹 Preprocessing Function (Ensures RGB + Correct Size)
function outputImage = preprocess(inputFile, netType)
    img = imread(inputFile);

    % Ensure image has 3 channels (RGB)
    if size(img, 3) == 1
        img = cat(3, img, img, img); % Convert grayscale to RGB
    end

    % Resize according to network requirements
    if strcmp(netType, 'AlexNet')
        img = imresize(img, [227 227]); % AlexNet requires 227x227x3
    elseif strcmp(netType, 'ResNet50')
        img = imresize(img, [224 224]); % ResNet50 requires 224x224x3
    end

    % Image Enhancement (Sharpening + Contrast Adjustment)
    img = imsharpen(img);
    img = imadjust(img);

    outputImage = img;
end

% Preprocess Training & Validation Data
augimdsTrainAlexNet = augmentedImageDatastore([227 227], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsValidationAlexNet = augmentedImageDatastore([227 227], imdsValidation, 'ColorPreprocessing', 'gray2rgb');

augimdsTrainResNet50 = augmentedImageDatastore([224 224], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsValidationResNet50 = augmentedImageDatastore([224 224], imdsValidation, 'ColorPreprocessing', 'gray2rgb');

%% 🔹 Transfer Learning Using AlexNet
netAlexNet = alexnet;
numClasses = numel(categories(imdsTrain.Labels));

% Modify AlexNet Layers
layersAlexNet = [
    netAlexNet.Layers(1:end-3)
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer
];

% Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidationAlexNet, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train AlexNet Model
netAlexNet = trainNetwork(augimdsTrainAlexNet, layersAlexNet, options);
save('skin_disease_alexnet.mat', 'netAlexNet');

%% 🔹 Transfer Learning Using ResNet50
netResNet50 = resnet50;

% Modify ResNet50 Layers
lgraph = layerGraph(netResNet50);
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc');

% Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-3, ...
    'Momentum', 0.9, ...
    'ValidationData', augimdsValidationResNet50, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train ResNet50 Model
netResNet50 = trainNetwork(augimdsTrainResNet50, lgraph, options);
save('skin_disease_resnet50.mat', 'netResNet50');

%% 🔹 Model Evaluation
YPredAlexNet = classify(netAlexNet, augimdsValidationAlexNet);
YPredResNet50 = classify(netResNet50, augimdsValidationResNet50);
YValidation = imdsValidation.Labels;

accuracyAlexNet = mean(YPredAlexNet == YValidation) * 100;
accuracyResNet50 = mean(YPredResNet50 == YValidation) * 100;

disp(['✅ AlexNet Validation Accuracy: ', num2str(accuracyAlexNet), '%']);
disp(['✅ ResNet50 Validation Accuracy: ', num2str(accuracyResNet50), '%']);

%% 🔹 Load Test Image for Prediction
testImageFile = "C:\Users\KIIT\Downloads\skincancer\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Skin-cancer-detection-from-dermoscopic-images-using-deep-learning-main\Data\Test\Test\vascular lesion\ISIC_0024375.jpg";
originalImage = imread(testImageFile);

% Preprocess Image for Both Models
testImageAlexNet = preprocess(testImageFile, 'AlexNet');
testImageResNet = preprocess(testImageFile, 'ResNet50');

augTestAlexNet = augmentedImageDatastore([227 227], testImageAlexNet);
augTestResNet50 = augmentedImageDatastore([224 224], testImageResNet);

% Classify Image
YPredAlexNet = classify(netAlexNet, augTestAlexNet);
YPredResNet50 = classify(netResNet50, augTestResNet50);

disp(['🔍 AlexNet Prediction: ', char(YPredAlexNet)]);
disp(['🔍 ResNet50 Prediction: ', char(YPredResNet50)]);

%% 🔹 Fuzzy C-Means Segmentation
grayImage = rgb2gray(originalImage);
data = double(grayImage(:));

numClusters = 2;
[centers, U] = fcm(data, numClusters);
[~, maxIndex] = max(U);
clusteredImage = reshape(maxIndex, size(grayImage));

skinCancerMask = clusteredImage == 2;

%% 🔹 Display Original, Enhanced, and Segmented Images
figure;

subplot(1,3,1);
imshow(originalImage);
title('🔹 Original Image');

subplot(1,3,2);
imshow(testImageAlexNet);
title('🔹 Enhanced Image');

subplot(1,3,3);
imshow(skinCancerMask);
title('🔹 Segmented Cancer Area (Fuzzy C-Means)');

% Save segmented image
imwrite(skinCancerMask, 'segmented_skin_cancer_mask.jpg');

%% 🔹 GUI Application for Skin Cancer Detection (To be implemented)
% Example GUI Code
% figure;
% uicontrol('Style', 'pushbutton', 'String', 'Upload Image', 'Position', [20 350 100 40], 'Callback', @uploadImage);
% function uploadImage(~,~)
%   [file, path] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select an Image');
%   if isequal(file,0)
%       disp('No file selected');
%   else
%       img = imread(fullfile(path, file));
%       imshow(img);
%       title('Uploaded Image for Skin Cancer Detection');
%   end
% end
