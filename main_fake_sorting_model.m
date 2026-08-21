%% ============================================================
% SIMPLE RESIDENTIAL SORTING MODEL WITH FAKE DATA
% ============================================================

clearvars; close all; clc;
clear LL_function_fake


%% ============================================================
% STEP 1: CREATE A SMALL FAKE DATASET
% ============================================================

% We have:
%   2 years
%   4 households per year
%   3 neighborhoods per year

years = 2013:2014;


%% ------------------------------------------------------------
% HOUSEHOLD DATA
% ------------------------------------------------------------

HH = table();

% Household ID
HH.i = [1;2;3;4; ...
        1;2;3;4];

% Year
HH.t = [2013;2013;2013;2013; ...
        2014;2014;2014;2014];

% Neighborhood actually chosen by each household
HH.j = ["A";"B";"A";"C"; ...
        "B";"C";"B";"A"];


% ------------------------------------------------------------
% Household characteristics
%
% Household 1 = white, high income
% Household 2 = low income
% Household 3 = Black
% Household 4 = Hispanic
% ------------------------------------------------------------

HH.inc_low = [0;1;0;0; ...
              0;1;0;0];

HH.inc_med = [0;0;0;0; ...
              0;0;0;0];

HH.race_black = [0;0;1;0; ...
                 0;0;1;0];

HH.race_asian = [0;0;0;0; ...
                 0;0;0;0];

HH.hispanic = [0;0;0;1; ...
               0;0;0;1];


%% ------------------------------------------------------------
% NEIGHBORHOOD DATA
% ------------------------------------------------------------

jt = table();

% Neighborhood ID
jt.location_j = ["A";"B";"C"; ...
                 "A";"B";"C"];

% Year
jt.t = [2013;2013;2013; ...
        2014;2014;2014];


% Voluntary buyout exposure

jt.voluntary_count_intensity = ...
    [0.10; 0.20; 0.00; ...
     0.15; 0.25; 0.05];


% Mandatory buyout exposure
%
% No mandatory buyouts in 2013.
% In 2014, neighborhoods B and C have mandatory exposure.

jt.mandatory_count_intensity = ...
    [0.00; 0.00; 0.00; ...
     0.00; 0.20; 0.10];


%% ============================================================
% STEP 2: SORT THE DATA
% ============================================================

% Sort households by year and household ID.
HH = sortrows(HH, {'t','i'});

% Sort neighborhoods by year and neighborhood ID.
jt = sortrows(jt, {'t','location_j'});


%% ============================================================
% STEP 3: STORE INFORMATION FOR EACH YEAR
% ============================================================

year_info = struct();

for yy = 1:length(years)

    yr = years(yy);

    % Find the household rows belonging to this year.
    year_info(yy).HH_idx = find(HH.t == yr);

    % Find the neighborhood rows belonging to this year.
    year_info(yy).BG_idx = find(jt.t == yr);

    % Number of households this year.
    year_info(yy).N = length(year_info(yy).HH_idx);

    % Number of neighborhoods this year.
    year_info(yy).J = length(year_info(yy).BG_idx);

    % Store the actual year.
    year_info(yy).year = yr;

end


%% ============================================================
% STEP 4: CALCULATE OBSERVED MARKET SHARES
% ============================================================

% We want to know what fraction of households actually chose
% each neighborhood.

J_total = height(jt);

s_obs = zeros(J_total,1);


for yy = 1:length(years)

    % Get the households and neighborhoods for this year.
    hh_idx = year_info(yy).HH_idx;
    bg_idx = year_info(yy).BG_idx;

    % Number of households this year.
    N = year_info(yy).N;

    % Actual neighborhood chosen by each household.
    chosen_j = HH.j(hh_idx);

    % Calculate the observed share for each neighborhood.
    for j = 1:length(bg_idx)

        s_obs(bg_idx(j)) = ...
            sum(chosen_j == jt.location_j(bg_idx(j))) / N;

    end

end


%% ============================================================
% STEP 5: CREATE THE INITIAL DELTA
% ============================================================

% delta is the neighborhood-specific component of utility.
%
% We start with a guess based on observed market shares.

delta_init = log(max(s_obs, 1e-12));


%% ------------------------------------------------------------
% NORMALIZE DELTA WITHIN EACH YEAR
% ------------------------------------------------------------

% Delta is only identified relative to other neighborhoods within
% the same year.
%
% Therefore, we normalize the first neighborhood in each year
% to have delta = 0.

for yy = 1:length(years)

    bg_idx = year_info(yy).BG_idx;

    first_idx = bg_idx(1);

    delta_init(bg_idx) = ...
        delta_init(bg_idx) - delta_init(first_idx);

end


%% ============================================================
% STEP 6: DEFINE THE PARAMETERS WE WANT TO ESTIMATE
% ============================================================

beta_names = [ ...
    "Voluntary baseline"
    "Mandatory baseline"
    "Voluntary × low income"
    "Voluntary × Black"
    "Voluntary × Hispanic"
    "Mandatory × low income"
    "Mandatory × Hispanic"
];

nBeta = length(beta_names);


% Initial guess for every beta.

beta_init = 0.01 * ones(nBeta,1);


%% ============================================================
% STEP 7: CHOOSE WHEN MANDATORY BUYOUTS BEGIN
% ============================================================

mandatory_start_year = 2014;


%% ============================================================
% STEP 8: SET UP THE OPTIMIZER
% ============================================================

options = optimoptions('fminunc', ...
    'Display','iter', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxFunctionEvaluations',10000, ...
    'MaxIterations',10000);


%% ============================================================
% STEP 9: ESTIMATE THE MODEL
% ============================================================

[beta_est,fval,exitflag,output,grad,hessian] = ...
    fminunc(@(b) LL_function_fake( ...
    b, delta_init, HH, jt, year_info, mandatory_start_year), ...
    beta_init, options);


%% ============================================================
% STEP 10: DISPLAY RESULTS
% ============================================================

results = table( ...
    beta_names, ...
    beta_est, ...
    'VariableNames', ...
    {'Parameter','Estimate'});

disp(results)