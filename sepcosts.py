import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
import librosa
import matplotlib.pyplot as plt

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target, interference):
        mse = self.mse(prediction, target)
        return mse

class SignalDistortionRatio(nn.Module):
    def __init__(self, l1_penalty=0, epsilon = 2e-7):
        super(SignalDistortionRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference):
        #sdr = torch.mean(prediction**2) / torch.mean(prediction * target)**2
        sdr = -torch.mean(prediction * target)**2 / (torch.mean(prediction**2) + self.epsilon)
        return sdr

class SignalInterferenceRatio(nn.Module):
    def __init__(self, epsilon=2e-7):
        super(SignalInterferenceRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference):
        # prediction = prediction / (torch.std(prediction, dim=1, keepdim=True) + self.epsilon)
        sir = torch.mean(prediction * interference)**2 / (torch.mean(prediction * target)**2 + self.epsilon)
        # sir = -torch.mean(prediction * target)**2 / (torch.mean(prediction * interference)**2 + self.epsilon)
        return sir

class SignalArtifactRatio(nn.Module):
    def __init__(self, epsilon=2e-7):
        super(SignalArtifactRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference):
        # prediction = prediction / (torch.std(prediction, dim=1, keepdim=True) + self.epsilon)
        inter_norm = torch.mean(interference**2, dim = 1, keepdim=True)
        target_norm = torch.mean(target**2, dim = 1, keepdim=True)
        ref_correlation = torch.mean(prediction * target, dim = 1, keepdim=True)
        inter_correlation = torch.mean(prediction * interference, dim = 1, keepdim=True)
        project = inter_norm * ref_correlation * target + target_norm * inter_correlation * interference
        #sar = -torch.mean(project**2) / (torch.mean(prediction**2) + self.epsilon)
        sar = torch.mean(prediction**2) / (torch.mean(project**2) + self.epsilon)
        return sar

class ShortTimeObjectiveIntelligibility(nn.Module):
    def __init__(self):
        super(ShortTimeObjectiveIntelligibility, self).__init__()
        self.fs = 16000
        self.num_bands = 15
        self.center_freq = 150
        self.min_energy = 40
        self.fft_size = 512
        self.fft_in_frame_size = 256
        self.hop = 128
        self.num_frames = 30
        self.beta =  1 + 10**(15 / 20)
        self.fft_pad = (self.fft_size - self.fft_in_frame_size) // 2

        scale = self.fft_size / self.hop
        window = np.hanning(self.fft_in_frame_size)
        zero_pad = np.zeros(self.fft_pad)
        window = np.concatenate([zero_pad, window, zero_pad])
        fft = np.fft.fft(np.eye(self.fft_size))
        self.rows = self.fft_size // 2 + 1
        fft = np.vstack((np.real(fft[:self.rows,:]), np.imag(fft[:self.rows,:])))
        fft = window * fft
        self.fftmat = nn.Parameter(torch.FloatTensor(fft).unsqueeze(1), requires_grad=False)
        self.octmat, _ = self._get_octave_mat(self.fs, self.fft_size,
                                              self.num_bands, self.center_freq)
        self.octmat = nn.Parameter(torch.FloatTensor(self.octmat), requires_grad=False)

    def forward(self, prediction, target, inteference):
        # pred, targ = self._remove_silent_frames(prediction, target)

        # (batch, 1, time) to (batch, fft_size, frames)
        pred_mag, pred_phase = self._stft(prediction)
        targ_mag, targ_phase = self._stft(target)

        # (batch, fft_size, frames) to (batch, frames, fft_size)
        pred_mag = pred_mag.permute(0, 2, 1).contiguous()
        targ_mag = targ_mag.permute(0, 2, 1).contiguous()

        # (batch, frames, fft_size) to (batch, frames, num_bands)
        x = torch.sqrt(F.linear(targ_mag**2, self.octmat))
        y = torch.sqrt(F.linear(pred_mag**2, self.octmat))

        # (batch, frames, num_bands) to (batch, num_bands, frames)
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()

        corr = 0
        for i, m in enumerate(range(self.num_frames, x.size()[2])):
            # segment (batch, num_bands, frames) to (batch, num_bands, new_frames)
            x_seg = x[:, :, m - self.num_frames : m]
            y_seg = y[:, :, m - self.num_frames : m]
            alpha = torch.sqrt(torch.sum(x_seg**2, dim=2, keepdim=True) / (torch.sum(y_seg**2, dim=2, keepdim=True) + 1e-7))
            y_prime = torch.min(alpha * y_seg, self.beta * x_seg)
            corr += self._correlation(x_seg, y_prime)

        return -corr / (i + 1)

    def _stft(self, seq):
        seq = seq.unsqueeze(1)
        stft = F.conv1d(seq, self.fftmat, stride=self.hop, padding=self.fft_pad)
        real = stft[:, :self.rows, :]
        imag = stft[:, self.rows:, :]
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        return mag, phase

    def _get_octave_mat(self, fs, nfft, numBands, mn):
        f = np.linspace(0, fs, nfft+1)
        f = f[:int(nfft/2)+1]
        k = np.arange(float(numBands))
        cf = 2**(k/3)*mn;
        fl = np.sqrt((2.**(k/3)*mn) * 2**((k-1.)/3)*mn)
        fr = np.sqrt((2.**(k/3)*mn) * 2**((k+1.)/3)*mn)
        A = np.zeros((numBands, len(f)) )

        for i in range(len(cf)) :
            b = np.argmin((f-fl[i])**2)
            fl[i] = f[b]
            fl_ii = b

            b = np.argmin((f-fr[i])**2)
            fr[i] = f[b]
            fr_ii = b
            A[i, np.arange(fl_ii,fr_ii)] = 1

        rnk = np.sum(A, axis=1)
        numBands = np.where((rnk[1:] >= rnk[:-1]) & (rnk[1:] != 0))[-1][-1]+1
        A = A[:numBands+1,:];
        cf = cf[:numBands+1];
        return A, cf

    def _remove_silent_frames(self, x, y):
        pass

    def _correlation(self, x, y):
        '''
        Input shape is (batch_size, bands, time dimension)
        '''
        xn = x - torch.mean(x, dim=2, keepdim=True)
        xn /= torch.sqrt(torch.sum(xn**2, dim=2, keepdim=True))
        yn = y - torch.mean(y, dim=2, keepdim=True)
        yn /= torch.sqrt(torch.sum(yn**2, dim=2, keepdim=True))
        r = torch.mean(torch.sum(xn * yn, dim=2))
        return r

class CombinedSIRSAR(nn.Module):
    def __init__(self, weight=1):
        super(CombinedSIRSAR, self).__init__()
        self.weight = weight
        self.sir = SignalInterferenceRatio()
        self.sar = SignalArtifactRatio()

    def forward(self, prediction, target, interference):
        # inter_norm = torch.mean(interference**2, dim = 1, keepdim=True)
        # target_norm = torch.mean(target**2, dim = 1, keepdim=True)
        # ref_correlation = torch.mean(prediction * target, dim = 1, keepdim=True)
        # inter_correlation = torch.mean(prediction * interference, dim = 1, keepdim=True)
        # project = inter_norm * ref_correlation * target + target_norm * inter_correlation * interference
        # sirsar = -torch.mean(project**2) / (torch.mean(prediction**2) +2e-7) * torch.mean(prediction * target)**2
        # return sirsar
        sir = self.sir(prediction, target, interference)
        sar = self.sar(prediction, target, interference)
        return (1 - self.weight) * sir + self.weight * sar

class CombinedSDRSTOI(nn.Module):
    def __init__(self, weight=1):
        super(CombinedSDRSTOI, self).__init__()
        self.weight = weight
        self.sdr = SignalDistortionRatio()
        self.stoi = ShortTimeObjectiveIntelligibility()

    def forward(self, prediction, target, interference):
        sdr = self.sdr(prediction, target, interference)
        stoi = self.stoi(prediction, target, interference)
        return (1 - self.weight) * sdr + self.weight * stoi

def main():
    clean, _ = librosa.core.load('results/clean_example.wav', sr=16000)
    plt.plot(clean)
    stoi = SignalArtifactRatio()
    clean = Variable(torch.FloatTensor(clean))
    noisy = Variable(torch.FloatTensor(noisy))
    noise = noisy - clean
    print('SAR: ', stoi(noisy.unsqueeze(0), clean.unsqueeze(0), noise.unsqueeze(0)))

if __name__ == '__main__':
    main()
