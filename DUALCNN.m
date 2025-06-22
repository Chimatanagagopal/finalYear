function layers = DUALCNN(inputSize)
    % This function defines the DUALCNN model using a Vision Transformer (ViT) 
    % and custom multi-head attention layers.

    % Define the Vision Transformer (ViT) branch
    vitBranch = [
        imageInputLayer(inputSize, 'Name', 'input_vit', 'Normalization', 'none')
        convolution2dLayer(16, 256, 'Stride', 16, 'Padding', 'same', 'Name', 'patch_embedding')
        PositionalEmbeddingLayer(256)  % Custom layer for positional encoding
        layerNormalizationLayer('Name', 'layer_norm1')
        MultiHeadAttentionLayer(8, 256, 'Name', 'mh_attention')  % Custom multi-head attention layer
        fullyConnectedLayer(256, 'Name', 'ffn_fc1')
        reluLayer('Name', 'relu_ffn')
        fullyConnectedLayer(256, 'Name', 'ffn_fc2')
        layerNormalizationLayer('Name', 'layer_norm2')
        flattenLayer('Name', 'vit_flatten')
        dropoutLayer(0.2, 'Name', 'vit_dropout')
    ];

    % Define the CNN branch
    cnnBranch = [
        imageInputLayer(inputSize, 'Name', 'input_cnn', 'Normalization', 'none')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 1, 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        flattenLayer('Name', 'cnn_flatten')
    ];

    % Fusion layer to combine both branches
    fusionLayer = [
        concatenationLayer(3, 2, 'Name', 'fusion')  % Concatenate the outputs of the two branches
        fullyConnectedLayer(512, 'Name', 'fusion_fc')
        reluLayer('Name', 'fusion_relu')
        fullyConnectedLayer(7, 'Name', 'output')  % Final classification layer (7 classes)
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classOutput')
    ];

    % Create layer graph
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, vitBranch);
    lgraph = addLayers(lgraph, cnnBranch);
    lgraph = addLayers(lgraph, fusionLayer);

    % Connect layers between branches and fusion layer
    lgraph = connectLayers(lgraph, 'vit_flatten', 'fusion/in1');
    lgraph = connectLayers(lgraph, 'cnn_flatten', 'fusion/in2');
    
    % Return the final layer graph
    layers = lgraph;
end
