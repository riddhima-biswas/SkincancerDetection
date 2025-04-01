classdef preluLayer < nnet.layer.Layer
    properties
        Alpha 
    end
    
    methods
        function layer = preluLayer(alpha, name)
            layer.Alpha = alpha;
            if nargin == 2
                layer.Name = name;
            end
            layer.Description = "PReLU activation with parameter " + alpha;
        end
        
        function Z = predict(layer, X)
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
    end
end