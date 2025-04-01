classdef customLossLayer < nnet.layer.RegressionLayer
    properties
        Alpha 
        Beta
        LambdaEdge
        LambdaNoise 
        LambdaEntropy
    end
    
    methods
        function layer = customLossLayer(alpha, beta, lambdaEdge, lambdaNoise, lambdaEntropy, name)
            layer.Alpha = alpha;
            layer.Beta = beta;
            layer.LambdaEdge = lambdaEdge;
            layer.LambdaNoise = lambdaNoise;
            layer.LambdaEntropy = lambdaEntropy;
            if nargin == 6
                layer.Name = name;
            end
            layer.Description = "Custom loss with NRMSE, MS-SSIM, EdgeStrength, NoiseLevel, and Entropy";
        end
        
        function loss = forwardLoss(layer, Y, T)

            nrmse = computeNRMSE(Y, T);
            mssim = ms_ssim(Y, T);
            edgeStrength = computeEdgeStrength(Y);
            noiseLevel = computeNoiseLevel(Y);
            entropyY = computeEntropy(Y);
            loss = layer.Alpha* nrmse+ layer.Beta * mssim - layer.LambdaEntropy * entropyY- layer.LambdaEdge * edgeStrength+ layer.LambdaNoise * noiseLevel ;
        end
    end
end

function nrmse=computeNRMSE(Y, T)
    mse = mean((Y - T).^2, 'all');
    nrmse = sqrt(mse) / (max(T(:)) - min(T(:)));
end

function mssimVal = ms_ssim(Y, T)
    if any(~isfinite(Y(:)))
        mssimVal=dlarray(1, 'SSCB');
        return;
    end
    T=dlarray(T, 'SSCB');
    Y=dlarray(Y,'SSCB');
    msSSIM = multissim(Y, T);
    msSSIM=squeeze(mean(msSSIM,'all'));
    mssimVal  = 1-msSSIM;
end



function edgeStrength = computeEdgeStrength(Y)
    Y = extractdata(Y); 
    gradMag =sqrt(imfilter(Y, [-1 0 1; -2 0 2; -1 0 1], "replicate","same").^2 + ...
                 imfilter(Y, [-1 -2 -1; 0 0 0; 1 2 1], "replicate","same").^2);

    threshold = mean(gradMag, [1, 2])./ std(gradMag, 0, [1, 2]);

    gradMag = gradMag ./ (1 + exp(-10 * (gradMag - threshold)));
    edgeStrength = mean(gradMag, 'all');
    if isnan(edgeStrength)
        edgeStrength = 0;
    end
end

function noiseLevel = computeNoiseLevel(Y)
    Y = extractdata(Y); 
    if any(~isfinite(Y(:)))
        noiseLevel=1e6;
        return;
    end
    localMean = imfilter(Y, ones(3)/9, "replicate","same");
    localVar = abs(imfilter(Y.^2, ones(3)/9, "replicate","same") - localMean.^2);
    noiseLevel = mean(localVar, 'all');
end

function entropyY = computeEntropy(Y) 
    Y = extractdata(Y); 
    numBins = 256; 
    sigma = 0.01;  
    binCenters = linspace(0, 1, numBins);  
    
    binCenters = reshape(binCenters, 1, 1, numBins);
    
    histY = sum(exp(-(Y - binCenters).^2 / (2 * sigma^2)), [1, 2]); 
    histY = histY ./ sum(histY, 3); 

    % Compute entropy
    entropyY = -sum(histY .* log(histY + eps), 3); 
    entropyY = mean(entropyY, 'all'); 
    if isnan(entropyY)
        entropyY = 0;
    end
end