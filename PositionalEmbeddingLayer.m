classdef PositionalEmbeddingLayer < nnet.layer.Layer
    % Custom layer to add positional encoding to input embeddings.

    properties
        OutputSize
    end
    
    methods
        function layer = PositionalEmbeddingLayer(outputSize)
            layer.Name = 'positional_embedding';
            layer.OutputSize = outputSize;
        end
        
        function Z = predict(layer, X)
            % Here, you can add the logic for positional encoding.
            % In this simple case, we'll add a fixed sinusoidal encoding.
            % Normally, you would generate this encoding based on the sequence length.
            position = 1:size(X, 2);
            positionalEncoding = sin(position' / 10000 .^ (2 * (1:layer.OutputSize) / layer.OutputSize));
            Z = X + positionalEncoding;
        end
    end
end
