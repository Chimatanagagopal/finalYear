% Assuming you have your image data in a folder structure like:
% data/train/label1/*.jpg, data/train/label2/*.jpg, etc.
% and you have corresponding labels for these images.

% Step 1: Load your image dataset
% Define the folder containing the training images
imageFolder = 'DATA';  % Replace with your actual data folder path

% Load the images using an image datastore
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

% Resize images to the input size expected by your model (224x224x3)
imds.ReadFcn = @(filename)imresize(imread(filename), [224, 224]);

% Step 2: Split your dataset into training and validation sets
% Split the dataset (70% for training, 30% for validation)
[trainImgs, valImgs] = splitEachLabel(imds, 0.7, 'randomized');

% Step 3: Define your model and training options
inputSize = [224, 224, 3];  % Example input size for the images

% Get the layers of the DUALCNN model
layers = DUALCNN(inputSize);

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Step 4: Train the model
% The labels are automatically inferred from the folder names in imageDatastore
trainedModel = trainNetwork(trainImgs, layers, options);

% Step 5: Evaluate the model (on validation data)
% Predict labels for validation data
predictedLabels = classify(trainedModel, valImgs);

% Calculate the accuracy
trueLabels = valImgs.Labels;
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);

fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
