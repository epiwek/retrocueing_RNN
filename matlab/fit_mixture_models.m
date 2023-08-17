% This script fits mixture models to the choice data from models from
% Experiment 4 (cue validities 0.5 and 0.75). Responses from each RNN model 
% are fit with a mixture model containing 4 parameters:
% 'K' - memory precision
% 'pT' - probability of making the target response
% 'pNT' - probability of making the non-target response
% 'pU' - probability of making random guesses
% Data from valid and invalid trials are fit separately.

% Note - the script requires the Analogue Toolbox from Bays Lab, available
% here: https://www.paulbays.com/toolbox/#mixture_methods

% set the data_path
data_path = '/data/';
save_path = '/results/';

% set up environment and variables
conditions = {'0.5','0.75'};
trial_type = {'valid_','invalid_'};
n_models = 30;
K_valid = zeros(n_models*2,1);
pT_valid = zeros(n_models*2,1);
pNT_valid = zeros(n_models*2,1);
pU_valid = zeros(n_models*2,1);
K_invalid = zeros(n_models*2,1);
pT_invalid = zeros(n_models*2,1);
pNT_invalid = zeros(n_models*2,1);
pU_invalid = zeros(n_models*2,1);
condition = vertcat(repmat(conditions(1),n_models,1),repmat(conditions(2),n_models,1));

% loop through conditions and trial types
for c=1:size(conditions,2)
    fprintf('Condition %d out of %d\n',[c;size(conditions,2)])
    for t = 1:size(trial_type,2)
        
        fprintf('      Trial type %d out of %d\n',[t;size(trial_type,2)])
        fprintf('      Load data\n')
        % load data
        data = load(strcat(data_path, 'expt_4_cue_val_', conditions{c}, '_', trial_type{t},'mixmodel_data.mat'));
        % fit mixture model to data from each RNN model
        fprintf('      Fit mixture models\n')
        for m = 1:n_models
            fprintf('          Model %d out of %d\n',[m,;n_models])
            B = mixtureFit(data.reported_colour(m,:)',...
                data.probed_colour',data.unprobed_colour');
            % B = [K,pT,pN,pU]
            % want to save both conditions stacked in one vector
            model_ix = m+(c-1)*n_models; 
            if t==1
                % valid trials
                K_valid(model_ix) = B(1);
                pT_valid(model_ix) = B(2);
                pNT_valid(model_ix) = B(3);
                pU_valid(model_ix) = B(4);
            else
                % invalid trials
                K_invalid(model_ix) = B(1);
                pT_invalid(model_ix) = B(2);
                pNT_invalid(model_ix) = B(3);
                pU_invalid(model_ix) = B(4);
            end
            fprintf('                       ...Done\n')
        end
        
    end
end

%% save data into tables

% all params
T = table(K_valid,pT_valid,pNT_valid,pU_valid,K_invalid,pT_invalid,...
    pNT_invalid,pU_invalid,condition);

writetable(T,strcat(data_path, 'mix_model_params.csv'))

% single param tables
K_table = table(K_valid,K_invalid,condition);
pT_table = table(pT_valid,pT_invalid,condition);
pNT_table = table(pNT_valid,pNT_invalid,condition);
pU_table = table(pU_valid,pU_invalid,condition);

writetable(K_table,strcat(save_path,'K_table.csv'))
writetable(pT_table,strcat(save_path,'pT_table.csv'))
writetable(pNT_table,strcat(save_path,'pNT_table.csv'))
writetable(pU_table,strcat(save_path,'pU_table.csv'))
 