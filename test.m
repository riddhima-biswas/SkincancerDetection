clc; clear; close all;

%% 🔹 Load Trained Models
load('skin_disease_alexnet.mat', 'netAlexNet');
load('skin_disease_resnet50.mat', 'netResNet50');

%% 🔹 Load Test Image
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Images (*.jpg, *.png, *.jpeg)'}, 'Select a Skin Image');
testImagePath = fullfile(path, file);
originalImage = imread(testImagePath);

% Show Original Image
figure;
subplot(1,3,1);
imshow(originalImage);
title('Original Image');

%% 🔹 Preprocess Image
load('finalinput.mat', 'pso_bestpos');

if iscell(pso_bestpos)
    pso_bestpos = cellfun(@double, pso_bestpos, 'UniformOutput', false);
    pso_bestpos = [pso_bestpos{:}];
end

if numel(pso_bestpos) < 2
    error('pso_bestpos does not contain enough valid resizing parameters.');
end
resizeDims = round(pso_bestpos(1:2));

img = imresize(originalImage, resizeDims);

if size(img, 3) == 3
    img = rgb2gray(img);
end

img = imadjust(img);
img = medfilt2(img, [3 3]);
img = imsharpen(img);

% Show Enhanced Image
subplot(1,3,2);
imshow(img);
title('Enhanced Image');

%% 🔹 Fuzzy C-Means Segmentation
data = double(img(:));
numClusters = 2; % Two clusters: Cancerous vs Non-Cancerous
[centers, U] = fcm(data, numClusters);
[~, maxIndex] = max(U);
clusteredImage = reshape(maxIndex, size(img));

skinCancerMask = clusteredImage == 2;

% Show Segmented Image
subplot(1,3,3);
imshow(skinCancerMask);
title('Segmented Image');

imwrite(skinCancerMask, 'segmented_skin_cancer_mask.jpg');

%% 🔹 Classification using AlexNet & ResNet50
inputSizeAlexNet = [227 227]; % AlexNet input size
inputSizeResNet = [224 224]; % ResNet50 input size

augTestAlexNet = augmentedImageDatastore(inputSizeAlexNet, originalImage);
augTestResNet = augmentedImageDatastore(inputSizeResNet, originalImage);

YPredAlexNet = classify(netAlexNet, augTestAlexNet);
YPredResNet50 = classify(netResNet50, augTestResNet);

%% 🔹 Display Classification Result
disp(['🩺 AlexNet Prediction: ', char(YPredAlexNet)]);
disp(['🩺 ResNet50 Prediction: ', char(YPredResNet50)]);

% Show Classification Result in a New Figure
figure;
imshow(originalImage);
title(['Classified as: ', char(YPredResNet50)]);
text(10, 20, ['AlexNet: ', char(YPredAlexNet)], 'Color', 'yellow', 'FontSize', 12);
text(10, 40, ['ResNet50: ', char(YPredResNet50)], 'Color', 'red', 'FontSize', 12);
