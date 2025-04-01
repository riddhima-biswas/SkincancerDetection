classdef geluLayer < nnet.layer.Layer
    methods
        function layer = geluLayer(name)
            if nargin == 1
                layer.Name = name;
            end
            layer.Description = "GELU activation";
        end
        
        function Z = predict(layer, X)
            Z = 0.5 * X .* (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X.^3)));
        end
    end
end