%% =========================================================
%  PHASE 4: Closed-Loop Validation (FIXED)
%  Goal-Driven Digital Twin + AI Decision Engine
%
%  FLOW:
%  1. User picks preference
%  2. Generate random channel (MATLAB)
%  3. Write channel state to CSV
%  4. Call Python decision engine
%  5. Read best IRS index + waveform from CSV
%  6. Load actual IRS phase vector from phase2_results.mat
%  7. Simulate: AI config vs Default (same channel)
%  8. Show results and plots
%% =========================================================

clc; clear; close all;

%% =========================================================
%  PATHS
%% =========================================================
PYTHON_EXE = 'C:/Users/ADMIN/AppData/Local/Python/pythoncore-3.14-64/python.exe';
BRIDGE_PY  = 'C:/Users/ADMIN/Downloads/IDP_PYTHON/Phase4_DecisionBridge.py';
INPUT_CSV  = 'C:/Users/ADMIN/Downloads/IDP_PYTHON/channel_input.csv';
OUTPUT_CSV = 'C:/Users/ADMIN/Downloads/IDP_PYTHON/config_output.csv';

%% =========================================================
%  STEP 1: USER PICKS PREFERENCE
%% =========================================================
fprintf('===========================================\n');
fprintf('  6G AI-DRIVEN CLOSED LOOP DEMO\n');
fprintf('===========================================\n\n');
fprintf('Choose your communication goal:\n');
fprintf('  1. MaxReliability    (minimize errors)\n');
fprintf('  2. MaxThroughput     (maximize speed)\n');
fprintf('  3. UltraLowLatency   (minimize delay)\n');
fprintf('  4. EnergyEfficient   (save power)\n');
fprintf('  5. Balanced          (all equal)\n');
fprintf('  6. ReliabilitySpeed  (reliability + speed)\n');
fprintf('  7. GreenThroughput   (speed + energy)\n');
fprintf('  8. MissionCritical   (reliability + latency)\n\n');

pref_list = {'MaxReliability','MaxThroughput','UltraLowLatency',...
             'EnergyEfficient','Balanced','ReliabilitySpeed',...
             'GreenThroughput','MissionCritical'};

choice = input('Enter number (1-8): ');
if isempty(choice) || choice < 1 || choice > 8
    choice = 5;
end
preference_label = pref_list{choice};
fprintf('\nSelected: %s\n\n', preference_label);

%% =========================================================
%  STEP 2: LOAD PARAMS AND IRS CONFIGS
%% =========================================================
load('phase1_results.mat');
load('phase2_results.mat');

params     = phase1_results.params;
trellis    = phase1_results.trellis;
tbdepth    = phase1_results.tbdepth;
params.N_streams = 1;

% Load actual IRS phase vectors used during training
irs_all    = phase2_results.irs_all;   % 13 x 32 — exact vectors from training
irs_struct = phase2_results.irs;

% Fixed SNR and distance for demo
SNR_dB            = 5;
params.distance_m = 50;

fprintf('Channel: SNR=%.0fdB | Distance=%.0fm\n\n', SNR_dB, params.distance_m);

%% =========================================================
%  STEP 3: GENERATE RANDOM CHANNEL
%% =========================================================
fprintf('Step 1/6: Generating random channel...\n');

p_ch = params;
p_ch.subcarrier_spacing = 15e3;
p_ch.CP_length          = 16;
p_ch.symbol_duration    = 1/15e3;
p_ch.bandwidth          = params.N_subcarriers * 15e3;
p_ch.mod_order          = 2;

[H_direct, ch_feat] = gen_channel(p_ch);

% Initial simulation with zero-phase IRS and default waveform
phi_BS = -20; phi_UE = 15;
H_init = apply_irs(H_direct, zeros(1,32), irs_struct, p_ch, phi_BS, phi_UE);
m_init = sim_engine(p_ch, SNR_dB, trellis, tbdepth, H_init);

fprintf('  Channel gain     = %.4f\n', ch_feat.channel_gain);
fprintf('  Effective rank   = %.2f\n', ch_feat.effective_rank);
fprintf('  Path loss        = %.2f dB\n', ch_feat.path_loss_dB);
fprintf('  Doppler spread   = %.4f Hz\n', ch_feat.doppler_spread_Hz);
fprintf('  Initial BER      = %.4f\n', m_init.BER);
fprintf('  Initial Latency  = %.4f ms\n\n', m_init.latency_ms);

%% =========================================================
%  STEP 4: WRITE CHANNEL STATE TO CSV
%% =========================================================
fprintf('Step 2/6: Writing channel state to CSV...\n');

fid = fopen(INPUT_CSV, 'w');
fprintf(fid, ['snr_input_dB,sinr_dB,channel_gain,effective_rank,' ...
              'path_loss_dB,shadow_dB,rms_delay_spread,doppler_spread_Hz,' ...
              'distance_m,irs_gain_dB,AoD_mean,AoA_mean,preference_label\n']);
fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n', ...
    SNR_dB, m_init.SINR_dB, ch_feat.channel_gain, ch_feat.effective_rank, ...
    ch_feat.path_loss_dB, ch_feat.shadow_dB, ch_feat.rms_delay_spread, ...
    ch_feat.doppler_spread_Hz, ch_feat.distance_m, 0.0, ...
    ch_feat.AoD_mean, ch_feat.AoA_mean, preference_label);
fclose(fid);
fprintf('  Written: %s\n\n', INPUT_CSV);

%% =========================================================
%  STEP 5: CALL PYTHON DECISION ENGINE
%% =========================================================
fprintf('Step 3/6: Running AI decision engine (Python)...\n');

cmd = sprintf('"%s" "%s"', PYTHON_EXE, BRIDGE_PY);
[status, py_output] = system(cmd);

if status ~= 0
    fprintf('ERROR: Python failed.\n%s\n', py_output);
    return;
end
fprintf('%s\n', py_output);

%% =========================================================
%  STEP 6: READ BEST CONFIG
%% =========================================================
fprintf('Step 4/6: Reading AI-chosen config...\n');

cfg        = readtable(OUTPUT_CSV);
irs_idx    = cfg.irs_candidate_idx(1);   % 1 to 13
wf_scs_kHz = cfg.wf_scs_kHz(1);
wf_cp      = cfg.wf_cp(1);
wf_mod     = cfg.wf_mod(1);
pred_BER   = 10^cfg.predicted_log_BER(1);
pred_tput  = cfg.predicted_throughput(1);
pred_lat   = cfg.predicted_latency_ms(1);

% Use ACTUAL phase vector from training — no reconstruction
phase_vec  = irs_all(irs_idx, :);

fprintf('  AI chose:\n');
fprintf('    IRS config index  : %d\n', irs_idx);
fprintf('    Subcarrier spacing: %d kHz\n', wf_scs_kHz);
fprintf('    CP length         : %d\n', wf_cp);
fprintf('    Modulation        : %s\n', mod_name(wf_mod));
fprintf('  ML predictions:\n');
fprintf('    Predicted BER     : %.6f\n', pred_BER);
fprintf('    Predicted Tput    : %.4f bps/Hz\n', pred_tput);
fprintf('    Predicted Latency : %.4f ms\n\n', pred_lat);

%% =========================================================
%  STEP 7: SIMULATE AI CONFIG VS DEFAULT
%% =========================================================
fprintf('Step 5/6: Simulating...\n');

% AI optimized
p_ai = params;
p_ai.subcarrier_spacing = wf_scs_kHz * 1e3;
p_ai.CP_length          = wf_cp;
p_ai.mod_order          = wf_mod;
p_ai.bandwidth          = params.N_subcarriers * p_ai.subcarrier_spacing;
p_ai.symbol_duration    = 1 / p_ai.subcarrier_spacing;

H_ai = apply_irs(H_direct, phase_vec, irs_struct, p_ai, phi_BS, phi_UE);
m_ai = sim_engine(p_ai, SNR_dB, trellis, tbdepth, H_ai);

% Default: W2 waveform + zero IRS
p_def = params;
p_def.subcarrier_spacing = 15e3;
p_def.CP_length          = 16;
p_def.mod_order          = 2;
p_def.bandwidth          = params.N_subcarriers * 15e3;
p_def.symbol_duration    = 1/15e3;

H_def = apply_irs(H_direct, zeros(1,32), irs_struct, p_def, phi_BS, phi_UE);
m_def = sim_engine(p_def, SNR_dB, trellis, tbdepth, H_def);

%% =========================================================
%  STEP 8: SHOW RESULTS
%% =========================================================
fprintf('\nStep 6/6: Results\n');
fprintf('===========================================\n');
fprintf('  User Goal: %s\n', preference_label);
fprintf('===========================================\n');
fprintf('%-22s %14s %14s %14s\n','Metric','Default','AI Optimized','ML Predicted');
fprintf('%-22s %14s %14s %14s\n','------','-------','------------','------------');
fprintf('%-22s %14.6f %14.6f %14.6f\n','BER',...
    m_def.BER, m_ai.BER, pred_BER);
fprintf('%-22s %14.4f %14.4f %14.4f\n','Throughput(bps/Hz)',...
    m_def.throughput, m_ai.throughput, pred_tput);
fprintf('%-22s %14.4f %14.4f %14.4f\n','Latency(ms)',...
    m_def.latency_ms, m_ai.latency_ms, pred_lat);
fprintf('%-22s %14.2e %14.2e %14s\n','Energy/bit(J)',...
    m_def.energy_per_bit, m_ai.energy_per_bit, '-');
fprintf('===========================================\n\n');

ber_imp = (m_def.BER - m_ai.BER) / max(m_def.BER, 1e-10) * 100;
lat_imp = (m_def.latency_ms - m_ai.latency_ms) / m_def.latency_ms * 100;
tp_imp  = (m_ai.throughput - m_def.throughput) / max(m_def.throughput,1e-10) * 100;
e_imp   = (m_def.energy_per_bit - m_ai.energy_per_bit) / m_def.energy_per_bit * 100;

fprintf('Improvements over default:\n');
fprintf('  BER reduction      : %.1f%%\n', ber_imp);
fprintf('  Latency reduction  : %.1f%%\n', lat_imp);
fprintf('  Throughput gain    : %.1f%%\n', tp_imp);
fprintf('  Energy reduction   : %.1f%%\n\n', e_imp);

%% =========================================================
%  PLOTS
%% =========================================================
figure('Name','Phase 4 Results','Position',[50 50 1400 500]);

% Plot 1: Bar comparison
subplot(1,3,1);
def_vals = [m_def.BER, m_def.throughput, m_def.latency_ms, m_def.energy_per_bit*1e7];
ai_vals  = [m_ai.BER,  m_ai.throughput,  m_ai.latency_ms,  m_ai.energy_per_bit*1e7];
X = 1:4; bw = 0.35;
b1 = bar(X-bw/2, def_vals, bw); b1.FaceColor = [0.85 0.33 0.10];
hold on;
b2 = bar(X+bw/2, ai_vals,  bw); b2.FaceColor = [0.00 0.45 0.74];
set(gca,'XTick',1:4,'XTickLabel',{'BER','Tput','Lat','Energy(x1e-7)'});
legend('Default','AI Optimized','Location','northeast');
title(sprintf('Performance: %s', preference_label));
ylabel('Value'); grid on;

% Plot 2: ML Predicted vs Simulated
subplot(1,3,2);
pv = [pred_tput, pred_lat];
sv = [m_ai.throughput, m_ai.latency_ms];
scatter(pv, sv, 150, 'filled', 'b');
hold on;
allv = [pv, sv];
lims = [min(allv)*0.8, max(allv)*1.2];
plot(lims, lims, 'r--', 'LineWidth', 2);
xlabel('ML Predicted'); ylabel('MATLAB Simulated');
title('ML Prediction vs Simulation');
text(pred_tput, m_ai.throughput, '  Throughput', 'FontSize',10,'Color','w');
text(pred_lat,  m_ai.latency_ms, '  Latency',    'FontSize',10,'Color','w');
grid on;

% Plot 3: % Improvement
subplot(1,3,3);
impr_vals = [ber_imp, tp_imp, lat_imp, e_imp];
cols = zeros(4,3);
for ci=1:4
    if impr_vals(ci)>=0, cols(ci,:)=[0.20 0.63 0.17];
    else,                cols(ci,:)=[0.85 0.33 0.10]; end
end
bh = bar(1:4, impr_vals, 0.6);
bh.FaceColor = 'flat'; bh.CData = cols;
yline(0,'w--','LineWidth',1.5);
set(gca,'XTick',1:4,'XTickLabel',{'BER','Tput','Lat','Energy'});
title('% Improvement over Default');
ylabel('Improvement (%)'); grid on;
for ci=1:4
    text(ci, impr_vals(ci)+sign(impr_vals(ci))*2, ...
        sprintf('%.1f%%',impr_vals(ci)),...
        'HorizontalAlignment','center','FontSize',9,'Color','w');
end

sgtitle(sprintf('Phase 4 Closed Loop — Goal: %s | SNR=%.0fdB',...
    preference_label, SNR_dB), 'FontSize',13);

fprintf('=== Phase 4 Complete ===\n');

%% =========================================================
%  HELPER FUNCTIONS
%% =========================================================
function name = mod_name(mod_order)
    switch mod_order
        case 1, name='QPSK';
        case 2, name='16-QAM';
        case 3, name='64-QAM';
        otherwise, name='Unknown';
    end
end

function [H_freq, ch_feat] = gen_channel(params)
    Nrx=params.N_rx; Ntx=params.N_tx; Nk=params.N_used;
    Np=params.n_paths; d=params.ant_spacing;
    AoD=-60+120*rand(1,Np); AoA=-60+120*rand(1,Np);
    pows=exp(-(0:Np-1)); pows=pows/sum(pows);
    max_del=max(floor(params.CP_length*0.6),Np+1);
    delays=sort(randperm(max_del,Np)-1);
    g=sqrt(pows/2).*(randn(1,Np)+1j*randn(1,Np));
    v_ms=params.velocity_kmh/3.6; lam=3e8/params.carrier_freq;
    f_dop=(v_ms/lam)*cosd(AoA);
    a_tx=@(phi) exp(1j*2*pi*d*sind(phi)*(0:Ntx-1)')/sqrt(Ntx);
    a_rx=@(phi) exp(1j*2*pi*d*sind(phi)*(0:Nrx-1)')/sqrt(Nrx);
    H_freq=zeros(Nrx,Ntx,Nk);
    for k=1:Nk
        Hk=zeros(Nrx,Ntx);
        for p=1:Np
            tp=exp(-1j*2*pi*k*delays(p)/params.N_subcarriers);
            dp=exp(1j*2*pi*f_dop(p)*params.symbol_duration);
            Hk=Hk+g(p)*tp*dp*(a_rx(AoA(p))*a_tx(AoD(p))');
        end
        H_freq(:,:,k)=Hk;
    end
    avg_pwr=mean(mean(mean(abs(H_freq).^2)));
    H_freq=H_freq/max(sqrt(avg_pwr),1e-10);
    PL_dB=13.54+39.08*log10(max(params.distance_m,1))+...
          20*log10(params.carrier_freq/1e9);
    shad_dB=params.shadow_std_dB*randn;
    Havg=mean(H_freq,3);
    sv=svd(Havg); svn=sv/sum(sv); svn=svn(svn>1e-10);
    eff_rank=exp(-sum(svn.*log(svn)));
    ch_gain=mean(mean(mean(abs(H_freq).^2)));
    rms_ds=sqrt(max(sum(pows.*(delays-sum(pows.*delays)).^2),0));
    dop_sp=sqrt(max(sum(pows.*(f_dop-sum(pows.*f_dop)).^2),0));
    ch_feat.channel_gain=double(ch_gain);
    ch_feat.effective_rank=double(eff_rank);
    ch_feat.path_loss_dB=double(PL_dB);
    ch_feat.shadow_dB=double(shad_dB);
    ch_feat.rms_delay_spread=double(rms_ds);
    ch_feat.doppler_spread_Hz=double(dop_sp);
    ch_feat.distance_m=double(params.distance_m);
    ch_feat.AoD_mean=double(mean(AoD));
    ch_feat.AoA_mean=double(mean(AoA));
    ch_feat.H_avg=Havg;
end

function H_eff=apply_irs(H_direct,phase_vec,irs,params,phi_BS,phi_UE)
    Nrx=params.N_rx; Ntx=params.N_tx; Nk=params.N_used;
    Nirs=irs.N_elements; d=irs.ant_spacing;
    a_irs=@(phi) exp(1j*2*pi*d*sind(phi)*(0:Nirs-1)')/sqrt(Nirs);
    a_tx =@(phi) exp(1j*2*pi*d*sind(phi)*(0:Ntx-1)')/sqrt(Ntx);
    a_rx =@(phi) exp(1j*2*pi*d*sind(phi)*(0:Nrx-1)')/sqrt(Nrx);
    K=5;
    H_IRS_BS=sqrt(K/(K+1))*(a_irs(phi_BS)*a_tx(phi_BS)')+...
             sqrt(1/(K+1))*(randn(Nirs,Ntx)+1j*randn(Nirs,Ntx))/sqrt(2);
    H_IRS_UE=sqrt(K/(K+1))*(a_rx(phi_UE)*a_irs(phi_UE)')+...
             sqrt(1/(K+1))*(randn(Nrx,Nirs)+1j*randn(Nrx,Nirs))/sqrt(2);
    Theta=diag(exp(1j*phase_vec(:)));
    H_refl=H_IRS_UE*Theta*H_IRS_BS;
    pwr_d=mean(mean(mean(abs(H_direct).^2)));
    pwr_r=mean(mean(abs(H_refl).^2));
    scale=sqrt(0.15*pwr_d/max(pwr_r,1e-12));
    H_eff=H_direct;
    for k=1:Nk, H_eff(:,:,k)=H_direct(:,:,k)+H_refl*scale; end
end

function metrics=sim_engine(p,SNR_dB,trellis,tbdepth,H_eff)
    Nk=p.N_used; bps=2*p.mod_order; M=2^bps; cr=0.5;
    Havg=mean(H_eff,3);
    [U,~,V]=svd(Havg);
    w_tx=V(:,1); w_rx=U(:,1)';
    h_raw=zeros(1,Nk);
    for k=1:Nk, h_raw(k)=w_rx*H_eff(:,:,k)*w_tx; end
    h_phase=exp(1j*angle(h_raw));
    SNR_lin=10^(SNR_dB/10); nv=1.0/SNR_lin;
    n_bits=p.N_bits; n_coded=n_bits*2;
    n_syms=ceil(n_coded/bps); n_ofdm=ceil(n_syms/Nk);
    n_syms_pad=n_ofdm*Nk;
    tx_bits=randi([0 1],1,n_bits);
    coded=convenc(tx_bits,trellis);
    r=mod(length(coded),bps); if r~=0, coded=[coded,zeros(1,bps-r)]; end
    idx_tx=bi2de(reshape(coded,bps,[])', 'left-msb');
    syms_tx=qammod(idx_tx,M,'gray','UnitAveragePower',true);
    TX_fd=zeros(1,n_syms_pad);
    L=min(length(syms_tx),n_syms_pad); TX_fd(1:L)=syms_tx(1:L);
    RX_eq=zeros(1,n_syms_pad);
    for oi=1:n_ofdm
        kr=(oi-1)*Nk+1:oi*Nk;
        for k=1:Nk
            x_k=TX_fd(kr(k)); h_k=h_phase(k);
            y_k=h_k*x_k+sqrt(nv/2)*(randn+1j*randn);
            x_hat=conj(h_k)*y_k/(abs(h_k)^2+nv);
            RX_eq(kr(k))=x_hat;
        end
    end
    rx_s=RX_eq(1:n_syms);
    idx_rx=qamdemod(rx_s(:),M,'gray','UnitAveragePower',true);
    rb=reshape(de2bi(idx_rx,bps,'left-msb')',1,[]);
    rb_use=rb(1:min(length(rb),n_coded));
    dl=floor(length(rb_use)/2)*2;
    if dl<2*tbdepth, BER_out=0.5;
    else
        rd=vitdec(rb_use(1:dl),trellis,tbdepth,'trunc','hard');
        nc=min(n_bits,length(rd));
        BER_out=sum(tx_bits(1:nc)~=rd(1:nc))/nc;
    end
    sinr_lin=mean(abs(h_raw).^2)/max(nv,1e-15);
    SINR_dB_out=10*log10(max(sinr_lin,1e-15));
    se=cr*(Nk/p.N_subcarriers)*log2(1+sinr_lin);
    T_sym=1/p.subcarrier_spacing+p.CP_length/p.bandwidth;
    lat_ms=p.N_ofdm_symbols*T_sym*1e3;
    e_p_bit=(100e-3*lat_ms*1e-3)/max(n_bits*cr,1);
    metrics.BER=double(BER_out);
    metrics.throughput=double(se);
    metrics.latency_ms=double(lat_ms);
    metrics.energy_per_bit=double(e_p_bit);
    metrics.SINR_dB=double(SINR_dB_out);
end
