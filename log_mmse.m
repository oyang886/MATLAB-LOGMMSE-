% test log-mmse

fs = 48000;
BLOCK = 512;
OVERLAP = BLOCK/2;
LENGTH = length(data1);
ITR = fix(LENGTH/OVERLAP);

win = hanning(BLOCK);
noise_mean = zeros(BLOCK, 1);

n_frames = 64;

last_output = zeros(OVERLAP, 1);
x1 = zeros(OVERLAP, 1);
k = 1;
aa = 0.95;
mu = 0.95;
eta = 0.1;
ksi_min = 10^(-25/10);
prev = 0;
output = zeros(LENGTH, 1);
hw = zeros(BLOCK, 1);
hw(1) = 1;
hw = fft(hw);
vk = zeros(BLOCK, 1);

for n = 1:ITR
    temp = data1((n - 1) * OVERLAP + 1:n * OVERLAP);
    x = [x1;temp] .* win;
    X1 = fft(x);
    x1(:) = temp;

    abs_X1 = abs(X1);
    pow_X1 = abs_X1 .^ 2;

    if n <= n_frames
        noise_mean = noise_mean + abs_X1;
        noise_mu = noise_mean/n_frames;
        noise_mu2 = noise_mu.^2;
    elseif n == n_frames + 1
        div_noise2 = 1./noise_mu2;
        gammak = min(pow_X1 .* div_noise2, 40);

        ksi = aa + (1 - aa) * max(gammak - 1, 0);
    else
        div_noise2 = 1./noise_mu2;
        gammak = min(pow_X1 .* div_noise2, 40);
        ksi = aa * prev .* div_noise2 + (1 - aa) * max(gammak - 1, 0);
        ksi = max(ksi_min, ksi);

        A = ksi./(1 + ksi);
        log_sigma_k = gammak .* A - log(1 + ksi);

        vad_decision = sum(log_sigma_k)/BLOCK;
        if(vad_decision < eta)
            noise_mu2 = mu * noise_mu2 + (1 - mu) * pow_X1;
        end

        v = A .* gammak;
        for k = 1:BLOCK
            if v(k) < 0.1
                vk(k) = -2.3 * log10(v(k)) - 0.6;
            elseif v(k) > 1
                vk(k) = 10^(-0.52 * v(k) - 0.26);
            else
                vk(k) = -1.544 * log10(v(k)) + 0.166;
            end
        end

        ei_vk = exp(0.5 * vk);
        hw = A .* ei_vk;

        sig = abs_X1 .* hw;
        prev = sig .^ 2;
    end
    get_times = hw .* X1;
    filtered = real(ifft(get_times));
    output((n - 1) * OVERLAP + 1:n * OVERLAP) = filtered(1:OVERLAP) + last_output;
    last_output(:) = filtered(OVERLAP + 1:end);
end
