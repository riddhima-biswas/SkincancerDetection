clc;clear all;
inputFolder = 'PSOINPUT';
targetFolder = 'PSOTARGET';
imageFolder = 'source';
imageFiles = dir(fullfile(imageFolder, '*.png'));
minWidth = Inf;
minHeight = Inf;
targetImages={};
for i = 1:length(imageFiles)
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    img=rgb2gray(img);

    if size(img, 2) < size(img, 1)
        img = imrotate(img, 90);
    end
    minWidth = min(minWidth, size(img, 2));
    minHeight = min(minHeight, size(img, 1));
    targetImages{i} = img;
end
for i = 1:length(targetImages)
    img = targetImages{i};
    padWidth = minWidth - size(img, 2);
    padHeight = minHeight - size(img, 1);
    img = padarray(img, [padHeight padWidth], 0, 'post');
    if rand > 0.5
        img = flip(img, 2); % Flip horizontally
    end
    img=im2double(img);
    targetImages{i} = img;
end

inputImages = cell(size(targetImages));

for i = 1:length(targetImages)
    img = targetImages{i};
    sigma = rand * 2;
    img = imgaussfilt(img, sigma);
    choice = randi(3);
    switch choice
        case 1
            gamma = 0.5 + rand * 1.5;
            img = imadjust(img, [], [], gamma);
        case 2
            x1 = 0.1 + rand * 0.1; 
            x2 = 0.8 + rand * 0.1; 
            img = imadjust(img, [0.1 0.9], [x1 x2]);
        case 3
            x1 = 0.1 + rand * 0.2; 
            x2 = 0.7 + rand * 0.3; 
            img = imadjust(img, stretchlim(img, 0.2), [x1 x2]);
    end
    choice = randi(2);
    switch choice
        case 1
            density = rand * 0.0009;
            img = imnoise(img, 'salt & pepper', density);
        case 2
            img = imnoise(img, 'poisson');
    end

    img = imresize(img, 0.5);
    inputImages{i} = img;
end
mkdir(inputFolder);
mkdir(targetFolder);
for i = 1:length(inputImages)
    imwrite(inputImages{i}, fullfile(inputFolder, sprintf('input_%04d.tiff', i)));
    imwrite(targetImages{i}, fullfile(targetFolder, sprintf('target_%04d.tiff', i)));
end
