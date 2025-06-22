% Load the image dataset
% Assume that your image dataset is organized into subfolders for each class

datasetPath = 'archive'; % Replace with the path to your dataset
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display some sample images
figure;
perm = randperm(length(imds.Files), 20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    title(imds.Labels(perm(i)));
end

% Split the dataset into training and validation sets (e.g., 80% training, 20% validation)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.9, 'randomized');

% Define the CNN architecture
layers = [
    imageInputLayer([64 64 3]) % Use the same size as the augmented images
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(4) % 4 classes
    softmaxLayer
    classificationLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Resize images in the training and validation sets
augmentedTrainingSet = augmentedImageDatastore([64 64 3], imdsTrain);
augmentedValidationSet = augmentedImageDatastore([64 64 3], imdsValidation);

% Train the network
net = trainNetwork(augmentedTrainingSet, layers, options);

% Evaluate the network on validation data
predictedLabels = classify(net, augmentedValidationSet);
trueLabels = imdsValidation.Labels;

% Calculate the accuracy
accuracy = mean(predictedLabels == trueLabels);
disp(['Validation Accuracy: ', num2str(accuracy)]);

% Calculate precision, recall, and F1-score for each class
confMat = confusionmat(trueLabels, predictedLabels);
precision = diag(confMat)./sum(confMat,2);
recall = diag(confMat)./sum(confMat,1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

disp('Precision per class:');
disp(precision);
disp('Recall per class:');
disp(recall);
disp('F1-score per class:');
disp(f1Score);

% Display a confusion matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for Validation Data');

% Save the results to a .mat file
resultsFileName = 'classification_results.mat';
save(resultsFileName, 'accuracy', 'precision', 'recall', 'f1Score', 'confMat', 'predictedLabels', 'trueLabels');
disp(['Results saved to ', resultsFileName]);