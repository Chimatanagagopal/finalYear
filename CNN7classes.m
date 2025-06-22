% Load the image dataset
datasetPath = 'DATA'; % Replace with your dataset path
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Data augmentation for training set
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXReflection', true);

augmentedTrainingSet = augmentedImageDatastore([64 64 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augmentedValidationSet = augmentedImageDatastore([64 64 3], imdsValidation);

% Define CNN architecture
layers = [
    imageInputLayer([64 64 3], 'Name', 'InputLayer')
    
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BatchNorm1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool1')
    
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BatchNorm2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool2')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BatchNorm3')
    reluLayer('Name', 'ReLU3')
    dropoutLayer(0.3, 'Name', 'Dropout1')
    
    fullyConnectedLayer(256, 'Name', 'FC1')
    reluLayer('Name', 'ReLU4')
    dropoutLayer(0.4, 'Name', 'Dropout2')
    
    fullyConnectedLayer(7, 'Name', 'FCOutput') % 7 classes
    softmaxLayer('Name', 'Softmax')
    classificationLayer('Name', 'Output')];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augmentedValidationSet, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 0.0005, ...
    'Shuffle', 'every-epoch');

% Train the network
net = trainNetwork(augmentedTrainingSet, layers, options);

% Evaluate the network on validation data
[predictedLabels, scores] = classify(net, augmentedValidationSet);
trueLabels = imdsValidation.Labels;

% Calculate the accuracy
accuracy = mean(predictedLabels == trueLabels);
disp(['Validation Accuracy: ', num2str(accuracy)]);

% Calculate precision, recall, and F1-score
confMat = confusionmat(trueLabels, predictedLabels);
precision = diag(confMat) ./ sum(confMat, 2);
recall = diag(confMat) ./ sum(confMat, 1)';
f1Score = 2 * (precision .* recall) ./ (precision + recall);

disp('Precision per class:');
disp(precision);
disp('Recall per class:');
disp(recall);
disp('F1-score per class:');
disp(f1Score);

% Display confusion matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for Validation Data');

% Save the results to a .mat file
resultsFileName = 'classification_results_7_classes.mat';
save(resultsFileName, 'accuracy', 'precision', 'recall', 'f1Score', 'confMat', 'predictedLabels', 'trueLabels');
disp(['Results saved to ', resultsFileName]);

% Calculate and plot ROC curve and AUC for each class
figure;
hold on;
classNames = categories(trueLabels);
numClasses = numel(classNames);

for i = 1:numClasses
    trueClassLabels = (trueLabels == classNames{i});
    [X, Y, ~, AUC] = perfcurve(trueClassLabels, scores(:, i), 1);
    plot(X, Y, 'DisplayName', ['Class ', char(classNames{i}), ' (AUC=', num2str(AUC), ')']);
end

hold off;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Each Class');
legend('Location', 'Best');
