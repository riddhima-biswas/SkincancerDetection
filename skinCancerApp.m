
classdef skinCancerApp < matlab.apps.AppBase
    properties (Access = public)
        UIFigure matlab.ui.Figure
        ImportButton matlab.ui.control.Button
        AnalyzeButton matlab.ui.control.Button
        ImageAxes matlab.ui.control.UIAxes
        ResultLabel matlab.ui.control.Label
        ResultField matlab.ui.control.EditField
        ModelFileLabel matlab.ui.control.Label
        ModelFileField matlab.ui.control.EditField
        ClassLabel matlab.ui.control.Label
        ClassField matlab.ui.control.EditField
        ScoreLabel matlab.ui.control.Label
        ScoreField matlab.ui.control.EditField
    end
    
    methods (Access = private)
        
        % Import button callback
        function ImportButtonPushed(app, event)
            global a;
            [filename, folderPath] = uigetfile('.', 'Pick an Image');
            filename = strcat(folderPath, filename);
            fullFilePath = fullfile(folderPath, filename);
            a = imread(filename);
            imshow(a, 'Parent', app.ImageAxes);
        end

        % Analyze button callback
        function AnalyzeButtonPushed(app, event)
            global a;
            
            % Load the trained network model
            loadedData = load('skin_disease_resnet50.mat');
            model = loadedData.netTransfer;

            % Pre-process the image
            resizedImage = imresize(a, [224, 224]);
            [predictedLabel, scores] = classify(model, resizedImage);

            % Display results
            app.ClassField.Value = char(predictedLabel);
            app.ScoreField.Value = num2str(max(scores) * 100);
            app.ModelFileField.Value = 'ResNet50 Model';
        end
        
    end
    
    methods (Access = public)
        
        % Create the app
        function createComponents(app)
            app.UIFigure = uifigure('Position', [100, 100, 500, 400]);
            
            % Import Button
            app.ImportButton = uibutton(app.UIFigure, 'push', ...
                'Position', [50, 350, 100, 30], 'Text', 'Import Image');
            app.ImportButton.ButtonPushedFcn = createCallbackFcn(app, @ImportButtonPushed, true);
            
            % Analyze Button
            app.AnalyzeButton = uibutton(app.UIFigure, 'push', ...
                'Position', [200, 350, 100, 30], 'Text', 'Analyze');
            app.AnalyzeButton.ButtonPushedFcn = createCallbackFcn(app, @AnalyzeButtonPushed, true);
            
            % UI Axes for displaying image
            app.ImageAxes = axes(app.UIFigure, 'Position', [0.1, 0.2, 0.4, 0.5]);
            
            % Result Labels and Fields
            app.ResultLabel = uilabel(app.UIFigure, 'Position', [350, 320, 100, 22], 'Text', 'Result:');
            app.ResultField = uieditfield(app.UIFigure, 'text', 'Position', [400, 320, 80, 22]);
            
            app.ClassLabel = uilabel(app.UIFigure, 'Position', [350, 270, 100, 22], 'Text', 'Class:');
            app.ClassField = uieditfield(app.UIFigure, 'text', 'Position', [400, 270, 80, 22]);
            
            app.ScoreLabel = uilabel(app.UIFigure, 'Position', [350, 220, 100, 22], 'Text', 'Score:');
            app.ScoreField = uieditfield(app.UIFigure, 'text', 'Position', [400, 220, 80, 22]);
            
            app.ModelFileLabel = uilabel(app.UIFigure, 'Position', [350, 170, 100, 22], 'Text', 'Model File:');
            app.ModelFileField = uieditfield(app.UIFigure, 'text', 'Position', [400, 170, 80, 22]);
        end
        
        % Run the app
        function runApp(app)
            createComponents(app);
            app.UIFigure.Visible = 'on';
        end
    end
end
