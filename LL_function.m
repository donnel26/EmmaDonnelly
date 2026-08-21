function [LL, grad] = LL_function_fake( ...
    beta, delta_init, HH, jt, year_info, mandatory_start_year)


%% ============================================================
% WHAT THIS FUNCTION DOES
% ============================================================

% fminunc chooses a candidate set of beta parameters.
%
% For those beta parameters:
%
%       Start with delta
%              ↓
%       Calculate household utilities
%              ↓
%       Calculate choice probabilities
%              ↓
%       Calculate predicted market shares
%              ↓
%       Compare predicted shares to observed shares
%              ↓
%       Update delta
%              ↓
%       Repeat until delta converges
%              ↓
%       Calculate likelihood and gradient
%              ↓
%       Return them to fminunc
%              ↓
%       fminunc chooses a new beta
%
% This continues until fminunc finds the beta values that best
% explain the observed household choices.


%% ============================================================
% STEP 1: START WITH DELTA
% ============================================================

J_total = length(delta_init);


% Keep the previous converged delta in memory.
%
% This gives us a good starting point the next time fminunc
% calls this function.

persistent delta_old


if isempty(delta_old) || length(delta_old) ~= J_total

    delta = delta_init;

else

    delta = delta_old;

end


%% ============================================================
% STEP 2: BERRY CONTRACTION MAPPING
% ============================================================

tol = 1e-4;

diff = Inf;

iter = 0;


while diff > tol

    iter = iter + 1;


    if iter > 100000

        error("Contraction mapping did not converge.");

    end


    % Start with the current delta.
    delta_new = delta;


    %% --------------------------------------------------------
    % LOOP OVER YEARS
    % --------------------------------------------------------

    for yy = 1:length(year_info)


        %% ----------------------------------------------------
        % GET THIS YEAR'S DATA
        % ----------------------------------------------------

        yr = year_info(yy).year;

        hh_idx = year_info(yy).HH_idx;

        bg_idx = year_info(yy).BG_idx;


        HH_year = HH(hh_idx,:);

        BG_year = jt(bg_idx,:);


        %% ----------------------------------------------------
        % GET THIS YEAR'S DELTA
        % ----------------------------------------------------

        delta_year = delta(bg_idx);


        %% ----------------------------------------------------
        % NEIGHBORHOOD CHARACTERISTICS
        % ----------------------------------------------------

        vol = ...
            BG_year.voluntary_count_intensity;

        mand_raw = ...
            BG_year.mandatory_count_intensity;


        % Only include mandatory buyouts beginning in the
        % selected year.

        if yr >= mandatory_start_year

            mand = mand_raw;

        else

            mand = zeros(size(mand_raw));

        end


        %% ----------------------------------------------------
        % HOUSEHOLD CHARACTERISTICS
        % ----------------------------------------------------

        inc_low = HH_year.inc_low;

        black = HH_year.race_black;

        hispanic = HH_year.hispanic;


        %% ====================================================
        % STEP 3: HOUSEHOLD-SPECIFIC BUYOUT PREFERENCES
        % ====================================================

        % Each household can have a different response to
        % voluntary buyout exposure.

        vol_taste = ...
            beta(1) + ...
            beta(3) * inc_low + ...
            beta(4) * black + ...
            beta(5) * hispanic;


        % Each household can also have a different response to
        % mandatory buyout exposure.

        mand_taste = ...
            beta(2) + ...
            beta(6) * inc_low + ...
            beta(7) * hispanic;


        %% ====================================================
        % STEP 4: CALCULATE UTILITY FOR EVERY HOUSEHOLD-
        %         NEIGHBORHOOD COMBINATION
        % ====================================================

        % delta_year' is 1 x J.
        %
        % vol_taste * vol' is N x J.
        %
        % mand_taste * mand' is N x J.
        %
        % Therefore, U is N x J:
        %
        % Each ROW = one household.
        %
        % Each COLUMN = one neighborhood.
        %
        % Each CELL = utility household i receives from
        % choosing neighborhood j.

        U = ...
            delta_year' + ...
            vol_taste * vol' + ...
            mand_taste * mand';


        %% ====================================================
        % STEP 5: TURN UTILITIES INTO CHOICE PROBABILITIES
        % ====================================================

        % Find the largest utility for each household.

        U_max = max(U, [], 2);


        % Subtracting the maximum improves numerical stability.
        %
        % It does not change the choice probabilities.

        U_stable = U - U_max;


        % Exponentiate utility.

        expU = exp(U_stable);


        % Add exponentiated utility across all possible
        % neighborhoods for each household.

        denom = sum(expU, 2);


        % Calculate logit choice probabilities.
        %
        % P is N x J.
        %
        % Each row sums to 1.

        P = expU ./ denom;


        %% ====================================================
        % STEP 6: CALCULATE MODEL-PREDICTED MARKET SHARES
        % ====================================================

        % Average the choice probabilities across households.
        %
        % For each neighborhood:
        %
        % s_model(j) = average probability that households choose j

        s_model = mean(P,1)';


        %% ====================================================
        % STEP 7: CALCULATE OBSERVED MARKET SHARES
        % ====================================================

        chosen_j = HH_year.j;

        J_year = year_info(yy).J;

        s_obs_year = zeros(J_year,1);


        for j = 1:J_year

            % Calculate the fraction of households that actually
            % chose neighborhood j.

            s_obs_year(j) = ...
                mean(chosen_j == BG_year.location_j(j));

        end


        %% ====================================================
        % STEP 8: BERRY CONTRACTION UPDATE
        % ====================================================

        % Prevent log(0).

        s_model = max(s_model, 1e-12);

        s_obs_year = max(s_obs_year, 1e-12);


        % Update delta.
        %
        % If observed share > predicted share:
        %
        %       delta increases.
        %
        % The neighborhood becomes more attractive.
        %
        % If observed share < predicted share:
        %
        %       delta decreases.
        %
        % The neighborhood becomes less attractive.

        delta_new(bg_idx) = ...
            delta_year + ...
            log(s_obs_year) - ...
            log(s_model);


        %% ----------------------------------------------------
        % NORMALIZE DELTA
        % ----------------------------------------------------

        % Set the first neighborhood's delta equal to zero.

        first_idx = bg_idx(1);

        delta_new(bg_idx) = ...
            delta_new(bg_idx) - ...
            delta_new(first_idx);

    end


    %% ========================================================
    % STEP 9: CHECK WHETHER DELTA HAS CONVERGED
    % ========================================================

    diff = max(abs(delta_new - delta));


    if isnan(diff) || isinf(diff)

        error("Contraction produced NaN or Inf.");

    end


    % Use the updated delta in the next iteration.

    delta = delta_new;

end


%% ============================================================
% STEP 10: SAVE THE CONVERGED DELTA
% ============================================================

delta_old = delta;


%% ============================================================
% STEP 11: CALCULATE THE NEGATIVE LOG LIKELIHOOD
%           AND GRADIENT
% ============================================================

LL = 0;

grad = zeros(size(beta));


%% ------------------------------------------------------------
% LOOP OVER YEARS
% ------------------------------------------------------------

for yy = 1:length(year_info)


    %% --------------------------------------------------------
    % GET THIS YEAR'S DATA
    % --------------------------------------------------------

    yr = year_info(yy).year;

    hh_idx = year_info(yy).HH_idx;

    bg_idx = year_info(yy).BG_idx;

    N = year_info(yy).N;


    HH_year = HH(hh_idx,:);

    BG_year = jt(bg_idx,:);


    %% --------------------------------------------------------
    % VARIABLES
    % --------------------------------------------------------

    delta_year = delta(bg_idx);

    vol = ...
        BG_year.voluntary_count_intensity;

    mand_raw = ...
        BG_year.mandatory_count_intensity;


    if yr >= mandatory_start_year

        mand = mand_raw;

    else

        mand = zeros(size(mand_raw));

    end


    inc_low = HH_year.inc_low;

    black = HH_year.race_black;

    hispanic = HH_year.hispanic;


    %% --------------------------------------------------------
    % HOUSEHOLD-SPECIFIC TASTES
    % --------------------------------------------------------

    vol_taste = ...
        beta(1) + ...
        beta(3) * inc_low + ...
        beta(4) * black + ...
        beta(5) * hispanic;


    mand_taste = ...
        beta(2) + ...
        beta(6) * inc_low + ...
        beta(7) * hispanic;


    %% --------------------------------------------------------
    % UTILITY
    % --------------------------------------------------------

    U = ...
        delta_year' + ...
        vol_taste * vol' + ...
        mand_taste * mand';


    %% --------------------------------------------------------
    % CHOICE PROBABILITIES
    % --------------------------------------------------------

    U_max = max(U, [], 2);

    U_stable = U - U_max;

    expU = exp(U_stable);

    denom = sum(expU, 2);

    P = expU ./ denom;


    %% ========================================================
    % STEP 12: CALCULATE LIKELIHOOD AND GRADIENT
    % ========================================================

    chosen_j = HH_year.j;

    J_year = year_info(yy).J;


    for i = 1:N


        %% ----------------------------------------------------
        % FIND THE NEIGHBORHOOD THIS HOUSEHOLD ACTUALLY CHOSE
        % ----------------------------------------------------

        chosen_position = ...
            find(BG_year.location_j == chosen_j(i), 1);


        %% ----------------------------------------------------
        % PROBABILITY OF THE ACTUAL CHOICE
        % ----------------------------------------------------

        P_chosen = P(i, chosen_position);


        %% ----------------------------------------------------
        % ADD THIS HOUSEHOLD'S CONTRIBUTION TO THE
        % NEGATIVE LOG LIKELIHOOD
        % ----------------------------------------------------

        LL = LL - log(max(P_chosen, 1e-12));


        %% ----------------------------------------------------
        % CREATE OBSERVED CHOICE INDICATOR
        % ----------------------------------------------------

        % y_row is:
        %
        % [1 0 0] if the household chose neighborhood A
        %
        % [0 1 0] if the household chose neighborhood B
        %
        % [0 0 1] if the household chose neighborhood C

        y_row = zeros(1,J_year);

        y_row(chosen_position) = 1;


        %% ----------------------------------------------------
        % OBSERVED MINUS PREDICTED CHOICE
        % ----------------------------------------------------

        residual = y_row - P(i,:);


        %% ----------------------------------------------------
        % DERIVATIVE OF UTILITY WITH RESPECT TO EACH BETA
        % ----------------------------------------------------

        % Voluntary baseline
        x1 = vol';

        % Mandatory baseline
        x2 = mand';

        % Voluntary × low income
        x3 = vol' * inc_low(i);

        % Voluntary × Black
        x4 = vol' * black(i);

        % Voluntary × Hispanic
        x5 = vol' * hispanic(i);

        % Mandatory × low income
        x6 = mand' * inc_low(i);

        % Mandatory × Hispanic
        x7 = mand' * hispanic(i);


        %% ----------------------------------------------------
        % ADD THIS HOUSEHOLD'S CONTRIBUTION TO THE GRADIENT
        % ----------------------------------------------------

        grad(1) = grad(1) - sum(residual .* x1);

        grad(2) = grad(2) - sum(residual .* x2);

        grad(3) = grad(3) - sum(residual .* x3);

        grad(4) = grad(4) - sum(residual .* x4);

        grad(5) = grad(5) - sum(residual .* x5);

        grad(6) = grad(6) - sum(residual .* x6);

        grad(7) = grad(7) - sum(residual .* x7);

    end

end


end
