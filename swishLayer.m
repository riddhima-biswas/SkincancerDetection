classdef swishLayer < nnet.layer.Layer
    properties
        Beta 
    end
    
    methods
        function layer = swishLayer(beta, name)
            layer.Beta = beta;
            if nargin == 2
                layer.Name = name;
            end
            layer.Description = "Swish activation with parameter " + beta;
        end
        
        function Z = predict(layer, X)
            Z = X .* sigmoid(layer.Beta * X);
        end
    end
end