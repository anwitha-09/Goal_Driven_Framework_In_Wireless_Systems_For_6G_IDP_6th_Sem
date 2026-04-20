%% =========================================================
%  PHASE 1 (REWRITTEN): MIMO-OFDM Digital Twin Core
%  Project: Goal-Driven DT-Based Adaptive IRS & Waveform
%           Optimization for 6G Wireless Systems
%  ---------------------------------------------------------
%  DESIGN PHILOSOPHY (why previous versions failed):
%    - Path loss at 3.5 GHz over 50m = ~110-140 dB
%    - If applied to signal, received power ~ 1e-14
%    - noise_var = 1/SNR assumes signal power = 1
%    - Result: effective SNR = -100 dB regardless of input
%
%  FIX: Standard simulation practice —
%    - Channel H is normalized (unit average power per tap)
%    - SNR is directly controlled via noise variance
%    - Path loss, shadow fading = RECORDED as channel features
%      for the dataset but NOT applied to signal magnitude
%    - This gives correct BER vs SNR curves
%    - Phase 2 dataset uses path loss as an input FEATURE
%      (it affects which config Python chooses) not a signal scalar
%
%  ARCHITECTURE:
%    - MIMO: 32 TX, 4 RX, 4 spatial streams
%    - SVD precoding + MMSE equalization
%    - Per-subcarrier frequency domain processing
%    - Convolutional coding (rate 1/2, K=7)
%    - QPSK / 16-QAM / 64-QAM
%    - 3GPP UMa geometry + Doppler + delay spread (as features)
%% =========================================================

clc; clear; close all;

%% =========================================================
%  SECTION 1: PARAMETERS
%% =========================================================

params.N_tx               = 32;
params.N_rx               = 4;
params.N_streams          = 4;

params.N_subcarriers      = 64;
params.N_used             = 52;
params.CP_length          = 16;
params.subcarrier_spacing = 15e3;       % Hz
params.symbol_duration    = 1 / 15e3;
params.mod_order          = 2;          % 1=QPSK,2=16QAM,3=64QAM
params.carrier_freq       = 3.5e9;
params.bandwidth          = 64 * 15e3;
params.distance_m         = 50;
params.n_paths            = 6;
params.shadow_std_dB      = 8.0;
params.velocity_kmh       = 3.0;
params.ant_spacing        = 0.5;        % wavelengths
params.N_bits             = 1040;
params.N_ofdm_symbols     = 10;
params.SNR_dB_range       = -5:2:30;

% Coding
trellis   = poly2trellis(7, [171 133]);
tbdepth   = 35;
code_rate = 0.5;

fprintf('=== Phase 1 (Rewritten): MIMO-OFDM Digital Twin ===\n');
fprintf('MIMO: %d TX x %d RX | %d streams\n', params.N_tx, params.N_rx, params.N_streams);
fprintf('Mod: %d-QAM | CP=%d | Δf=%.0f kHz\n\n', ...
    2^(2*params.mod_order), params.CP_length, params.subcarrier_spacing/1e3);


%% =========================================================
%  SECTION 2: CHANNEL GENERATOR
%  Generates NORMALIZED H (unit avg power per element)
%  Path loss computed separately as a feature
%% =========================================================

function [H_freq, ch_feat] = generate_channel(params)
% Returns:
%   H_freq  : N_rx x N_tx x N_used  (normalized, unit avg power)
%   ch_feat : struct of channel state features for dataset

    Nrx = params.N_rx;
    Ntx = params.N_tx;
    Nk  = params.N_used;
    Np  = params.n_paths;
    d   = params.ant_spacing;
    fc  = params.carrier_freq;
    lam = 3e8 / fc;

    %% Random geometry per realization
    AoD = -60 + 120*rand(1,Np);   % degrees
    AoA = -60 + 120*rand(1,Np);

    %% Power delay profile (exponential)
    pows = exp(-(0:Np-1));
    pows = pows / sum(pows);

    %% Integer delays in samples (within CP)
    max_del = floor(params.CP_length * 0.6);
    delays  = sort(randperm(max_del, Np) - 1);

    %% Complex path gains (Rayleigh)
    g = sqrt(pows/2) .* (randn(1,Np) + 1j*randn(1,Np));

    %% Doppler
    v_ms  = params.velocity_kmh / 3.6;
    f_max = v_ms / lam;
    f_dop = f_max * cosd(AoA);

    %% ULA steering vectors
    a_tx = @(phi) exp(1j*2*pi*d*sind(phi)*(0:Ntx-1)') / sqrt(Ntx);
    a_rx = @(phi) exp(1j*2*pi*d*sind(phi)*(0:Nrx-1)') / sqrt(Nrx);

    %% Build H per subcarrier (NORMALIZED — no path loss applied)
    H_freq = zeros(Nrx, Ntx, Nk);
    for k = 1:Nk
        Hk = zeros(Nrx, Ntx);
        for p = 1:Np
            tau_phase = exp(-1j*2*pi*k*delays(p)/params.N_subcarriers);
            dop_phase = exp(1j*2*pi*f_dop(p)*params.symbol_duration);
            Hk = Hk + g(p) * tau_phase * dop_phase * ...
                      (a_rx(AoA(p)) * a_tx(AoD(p))');
        end
        H_freq(:,:,k) = Hk;
    end

    %% Normalize H so average Frobenius norm = sqrt(Nrx*Ntx)
    %  This keeps signal power = 1 after precoding
    avg_frob = mean(sqrt(sum(sum(abs(H_freq).^2,1),2)));
    norm_factor = sqrt(Nrx * Ntx) / max(avg_frob, 1e-10);
    H_freq = H_freq * norm_factor;

    %% Channel state FEATURES (for dataset — not applied to signal)
    PL_dB     = 13.54 + 39.08*log10(max(params.distance_m,1)) + ...
                20*log10(params.carrier_freq/1e9);
    shadow_dB = params.shadow_std_dB * randn;

    % Channel gain (Frobenius norm, normalized)
    ch_gain   = mean(sum(sum(abs(H_freq).^2,1),2)) / (Nrx*Ntx);

    % Effective rank
    Havg = mean(H_freq, 3);
    sv   = svd(Havg);
    svn  = sv/sum(sv); svn = svn(svn>1e-10);
    eff_rank = exp(-sum(svn.*log(svn)));

    % RMS delay spread
    rms_ds = sqrt(sum(pows.*(delays - sum(pows.*delays)).^2));

    % Doppler spread
    dop_sp = sqrt(sum(pows.*(f_dop - sum(pows.*f_dop)).^2));

    ch_feat.channel_gain       = ch_gain;
    ch_feat.effective_rank     = eff_rank;
    ch_feat.path_loss_dB       = PL_dB;
    ch_feat.shadow_dB          = shadow_dB;
    ch_feat.total_path_loss_dB = PL_dB + shadow_dB;
    ch_feat.rms_delay_spread   = rms_ds;
    ch_feat.doppler_spread_Hz  = dop_sp;
    ch_feat.distance_m         = params.distance_m;
    ch_feat.AoD_mean           = mean(AoD);
    ch_feat.AoA_mean           = mean(AoA);
    ch_feat.H_avg              = Havg;
    ch_feat.irs_gain_dB        = 0;
end


%% =========================================================
%  SECTION 3: MODULATION HELPERS
%% =========================================================

function syms = mod_bits(bits, mod_order)
    M   = 2^(2*mod_order);
    bps = log2(M);
    rem = mod(length(bits), bps);
    if rem ~= 0, bits = [bits, zeros(1, bps-rem)]; end
    idx  = bi2de(reshape(bits, bps, [])', 'left-msb');
    syms = qammod(idx, M, 'gray', 'UnitAveragePower', true);
end

function bits = demod_syms(syms, mod_order)
    M    = 2^(2*mod_order);
    bps  = log2(M);
    idx  = qamdemod(syms(:), M, 'gray', 'UnitAveragePower', true);
    bits = reshape(de2bi(idx, bps, 'left-msb')', 1, []);
end


%% =========================================================
%  SECTION 4: CORE SIMULATION
%  All processing in frequency domain per subcarrier
%  SNR directly controls noise variance (no path loss scaling)
%% =========================================================

function [metrics, ch_feat] = run_simulation(params, SNR_dB, trellis, tbdepth)

    Ns   = params.N_streams;
    Nrx  = params.N_rx;
    Ntx  = params.N_tx;
    Nk   = params.N_used;
    bps  = 2 * params.mod_order;
    cr   = 0.5;   % code rate

    %% Generate channel
    [H_freq, ch_feat] = generate_channel(params);

    %% SVD precoding from average channel
    [U, S, V] = svd(ch_feat.H_avg);
    W_tx = V(:, 1:Ns);       % Ntx x Ns — TX precoder
    W_rx = U(:, 1:Ns)';      % Ns x Nrx — RX combiner

    %% Noise variance — directly from SNR
    %  Since H is normalized and signal power = 1 per symbol,
    %  noise_var = 1/SNR gives correct received SNR
    SNR_lin  = 10^(SNR_dB/10);
    nv       = 1.0 / SNR_lin;   % noise variance per complex dim

    %% Compute symbols needed
    n_bits       = params.N_bits;
    n_coded      = n_bits * 2;                    % rate 1/2
    n_syms       = ceil(n_coded / bps);           % QAM symbols
    n_ofdm       = ceil(n_syms / Nk);             % OFDM symbols
    n_syms_pad   = n_ofdm * Nk;                   % padded length

    %% Per-stream TX/RX
    all_tx_bits  = zeros(Ns, n_bits);
    stream_BER   = zeros(1, Ns);
    stream_SINR  = zeros(Ns, Nk);    % linear SINR per stream per subcarrier

    %% Build freq-domain TX matrix: Ns x n_syms_pad
    TX_fd = zeros(Ns, n_syms_pad);
    for s = 1:Ns
        tb = randi([0 1], 1, n_bits);
        all_tx_bits(s,:) = tb;
        coded = convenc(tb, trellis);
        syms  = mod_bits(coded, params.mod_order);
        TX_fd(s, 1:length(syms)) = syms;
    end

    %% Frequency-domain MIMO processing — per OFDM symbol, per subcarrier
    RX_eq = zeros(Ns, n_syms_pad);

    for ofdm_i = 1:n_ofdm
        k_range = (ofdm_i-1)*Nk+1 : ofdm_i*Nk;
        X_blk   = TX_fd(:, k_range);    % Ns x Nk (TX symbols this OFDM sym)

        for k = 1:Nk
            x_k = X_blk(:, k);          % Ns x 1

            %% Effective channel: H_eff_k = H_k * W_tx  (Nrx x Ns)
            H_k     = H_freq(:,:,k);
            H_eff_k = H_k * W_tx;

            %% Received signal
            y_k = H_eff_k * x_k + ...
                  sqrt(nv/2)*(randn(Nrx,1) + 1j*randn(Nrx,1));

            %% MMSE equalizer: Ns x Nrx
            %  W = H_eff^H * (H_eff*H_eff^H + nv*I)^{-1}
            W_eq = H_eff_k' / (H_eff_k*H_eff_k' + nv*eye(Nrx));

            %% Equalized output: Ns x 1
            x_hat = W_eq * y_k;
            RX_eq(:, k_range(k)) = x_hat;

            %% Per-stream SINR at this subcarrier (linear)
            for s = 1:Ns
                % Signal component
                sig = W_eq(s,:) * H_eff_k(:,s);     % scalar
                sig_pwr = abs(sig)^2;

                % Inter-stream interference
                intf_pwr = 0;
                for s2 = 1:Ns
                    if s2 ~= s
                        intf = W_eq(s,:) * H_eff_k(:,s2);
                        intf_pwr = intf_pwr + abs(intf)^2;
                    end
                end

                % Noise enhancement
                noise_out = nv * sum(abs(W_eq(s,:)).^2);

                stream_SINR(s,k) = sig_pwr / max(intf_pwr + noise_out, 1e-15);
            end
        end
    end

    %% Decode per stream
    for s = 1:Ns
        rx_s   = RX_eq(s, 1:n_syms);
        rx_bit = demod_syms(rx_s, params.mod_order);
        dl     = floor(length(rx_bit)/2)*2;
        rx_dec = vitdec(rx_bit(1:dl), trellis, tbdepth, 'trunc', 'hard');
        ml     = min(n_bits, length(rx_dec));
        stream_BER(s) = sum(all_tx_bits(s,1:ml) ~= rx_dec(1:ml)) / ml;
    end

    %% Aggregate
    BER_avg  = mean(stream_BER);

    % SINR per stream in dB
    sinr_lin_per_stream = mean(stream_SINR, 2);   % avg over subcarriers
    sinr_dB_per_stream  = 10*log10(sinr_lin_per_stream);
    SINR_avg_dB         = mean(sinr_dB_per_stream);

    % Spectral efficiency
    se = 0;
    for s = 1:Ns
        se = se + cr * (Nk/params.N_subcarriers) * ...
             log2(1 + sinr_lin_per_stream(s));
    end
    throughput = se;   % bps/Hz total

    % Latency
    T_sym      = 1/params.subcarrier_spacing + params.CP_length/params.bandwidth;
    latency_ms = params.N_ofdm_symbols * T_sym * 1e3;

    % Energy
    tx_pwr_mW      = 100;
    energy_per_bit = (tx_pwr_mW*1e-3 * latency_ms*1e-3) / ...
                      max(n_bits*Ns*cr, 1);

    % Effective rank
    sv   = svd(ch_feat.H_avg);
    svn  = sv/sum(sv); svn=svn(svn>1e-10);
    eff_rank = exp(-sum(svn.*log(svn)));

    metrics.BER             = BER_avg;
    metrics.stream_BER      = stream_BER;
    metrics.throughput      = throughput;
    metrics.latency_ms      = latency_ms;
    metrics.energy_per_bit  = energy_per_bit;
    metrics.SNR_input_dB    = SNR_dB;
    metrics.SINR_dB         = SINR_avg_dB;
    metrics.SINR_per_stream = sinr_dB_per_stream;
    metrics.effective_rank  = eff_rank;
    metrics.irs_gain_dB     = 0;

    ch_feat.SINR_dB         = SINR_avg_dB;
    ch_feat.SNR_actual_dB   = SNR_dB;
    ch_feat.effective_rank  = eff_rank;
end


%% =========================================================
%  SECTION 5: VALIDATION SWEEP
%% =========================================================

n_snr    = length(params.SNR_dB_range);
N_trials = 10;

BER_res    = zeros(1,n_snr);
TP_res     = zeros(1,n_snr);
SINR_res   = zeros(1,n_snr);
LAT_res    = zeros(1,n_snr);
RANK_res   = zeros(1,n_snr);
ENERGY_res = zeros(1,n_snr);

fprintf('Running validation sweep...\n');
fprintf('SNR(dB) | BER       | Tput(bps/Hz) | SINR(dB) | Rank | Lat(ms)\n');
fprintf('--------|-----------|--------------|----------|------|--------\n');

for i = 1:n_snr
    SNR_dB = params.SNR_dB_range(i);
    ba=0;ta=0;sa=0;la=0;ra=0;ea=0;

    for t = 1:N_trials
        [m, cp] = run_simulation(params, SNR_dB, trellis, tbdepth);
        ba=ba+m.BER; ta=ta+m.throughput; sa=sa+m.SINR_dB;
        la=la+m.latency_ms; ra=ra+m.effective_rank; ea=ea+m.energy_per_bit;
    end

    BER_res(i)    = ba/N_trials;
    TP_res(i)     = ta/N_trials;
    SINR_res(i)   = sa/N_trials;
    LAT_res(i)    = la/N_trials;
    RANK_res(i)   = ra/N_trials;
    ENERGY_res(i) = ea/N_trials;

    fprintf('  %5.1f   | %.3e  |   %8.4f   |  %6.2f  | %.2f | %.4f\n', ...
        SNR_dB, BER_res(i), TP_res(i), SINR_res(i), RANK_res(i), LAT_res(i));
end

fprintf('\nDone.\n');
fprintf('BER range:  [%.3e, %.3e]\n', min(BER_res), max(BER_res));
fprintf('Tput range: [%.3f, %.3f] bps/Hz\n', min(TP_res), max(TP_res));
fprintf('SINR range: [%.2f, %.2f] dB\n', min(SINR_res), max(SINR_res));


%% =========================================================
%  SECTION 6: PLOTS
%% =========================================================

snr_ax = params.SNR_dB_range;
figure('Name','Phase 1 Rewritten - Validation','Position',[50 50 1400 500]);

subplot(1,4,1);
semilogy(snr_ax, BER_res+1e-10, 'b-o','LineWidth',2,'MarkerFaceColor','b');
hold on;
yline(1e-3,'r--','LineWidth',1.5,'Label','10^{-3}');
xlabel('SNR (dB)'); ylabel('BER'); title('BER vs SNR');
grid on; ylim([1e-5 1]); xlim([snr_ax(1) snr_ax(end)]);

subplot(1,4,2);
plot(snr_ax, TP_res, 'g-s','LineWidth',2,'MarkerFaceColor','g');
xlabel('SNR (dB)'); ylabel('Spectral Efficiency (bps/Hz)');
title('Throughput vs SNR'); grid on;
ylim([0 max(TP_res)*1.15 + 0.5]);

subplot(1,4,3);
plot(snr_ax, SINR_res, 'm-^','LineWidth',2,'MarkerFaceColor','m');
xlabel('SNR (dB)'); ylabel('SINR (dB)');
title('SINR vs SNR'); grid on;

subplot(1,4,4);
plot(snr_ax, RANK_res, 'r-d','LineWidth',2,'MarkerFaceColor','r');
xlabel('SNR (dB)'); ylabel('Effective Rank');
title('Spatial DoF vs SNR'); grid on;
ylim([0 params.N_streams+1]);

sgtitle('Phase 1 (Rewritten): MIMO-OFDM Digital Twin Validation','FontSize',13);


%% =========================================================
%  SECTION 7: SAVE
%% =========================================================

phase1_results.params      = params;
phase1_results.trellis     = trellis;
phase1_results.tbdepth     = tbdepth;
phase1_results.code_rate   = code_rate;
phase1_results.BER         = BER_res;
phase1_results.Throughput  = TP_res;
phase1_results.SINR        = SINR_res;
phase1_results.Latency_ms  = LAT_res;
phase1_results.EffRank     = RANK_res;
phase1_results.Energy      = ENERGY_res;
phase1_results.SNR_range   = params.SNR_dB_range;

save('phase1_results.mat','phase1_results');
fprintf('\nSaved: phase1_results.mat\n');
fprintf('=== Phase 1 complete. Run Phase 2 next. ===\n');
