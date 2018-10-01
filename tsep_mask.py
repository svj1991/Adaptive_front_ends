# Prerequisites
from __future__ import print_function

# Neuralnety things
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoaderIter
import visdom

# General utilities
import time
import glob2
import librosa
from tqdm import trange

import numpy.random as random
from numpy import std, vstack, hstack, argsort, argmax, array, hanning, real, imag, floor, eye, savez
from numpy.fft import rfft, fft
from numpy.linalg import pinv

import pdb
#
# Data loader
#

# Make a speech+noise data loader
# SDR, SIR, SAR estimation
def bss_eval( sep, i, sources):
    # Current target
    from numpy import dot, linalg, log10
    min_len = min([len(sep), len(sources[i])])
    sources = sources[:,:min_len]
    sep = sep[:min_len]
    target = sources[i]

    # Target contribution
    s_target = target * dot( target, sep.T) / dot( target, target.T)

    # Interference contribution
    pse = dot( dot( sources, sep.T), \
    linalg.inv( dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif= sep - pse;

    # Interference + artifacts contribution
    e_total = e_interf + e_artif;

    # Computation of the log energy ratios
    sdr = 10*log10( sum( s_target**2) / sum( e_total**2));
    sir = 10*log10( sum( s_target**2) / sum( e_interf**2));
    sar = 10*log10( sum( (s_target + e_interf)**2) / sum( e_artif**2));

    # Done!
    return (sdr, sir, sar)
class Mix_Dataset( Dataset):
    def __init__( self, target_path, masker_path, sr=16000, snr=0, exclude=True, autoenc=False, val=2):

        # Remember these
        self.sr = sr
        self.ae = autoenc
        self.go_on = True

        # Some length constants
        self.sec = 2
        self.tq = 2048
        self.l = int( self.tq * (floor( self.sec*self.sr)/self.tq))

        # Get file lists
        self.target_files = random.permutation( self.get_files( target_path))
        self.masker_files = random.permutation( self.get_files( masker_path))
        print( 'Got', len( self.target_files), 'target files')
        print( 'Got', len( self.masker_files), 'masker files')

        # Do we have a hidden validation set?
        if exclude:
            vt = self.target_files[:val]
            vm = self.masker_files[:val]
            self.val = [vt,vm]
            self.target_files = list( set( self.target_files) - set( vt))
            self.masker_files = list( set( self.masker_files) - set( vm))
        else:
            self.val = [self.target_files,self.masker_files]

    # Return list of audio files in each directory
    def get_files( self, p):
        l = glob2.glob( p + '/**/*.wav') + glob2.glob( p + '/**/*.ogg')

        # Keep the ones longer than minimal number of secs
        return [i for i in l if librosa.core.get_duration( filename=i) > self.sec]

    def __len__( self):
        return 100000000 # arbitrary since I generate the data

    # Return a noisy/clean recording pair, returns ~2sec chunks
    def __getitem__( self, i):
        # Get random files
        ft = random.choice( self.target_files)
        fm = random.choice( self.masker_files)

        # Get random start times
        ot = random.random() * (librosa.core.get_duration( filename=ft) - self.sec)
        om = random.random() * (librosa.core.get_duration( filename=fm) - self.sec)

        # Load the sounds
        t,_ = librosa.core.load( ft, sr=self.sr, duration=self.sec, offset=ot, res_type='kaiser_fast')
        m,_ = librosa.core.load( fm, sr=self.sr, duration=self.sec, offset=om, res_type='kaiser_fast')

        # Trim length to something we can STFT cleanly
        t = t[:self.l]
        m = m[:self.l]

        # Normalize
        t = t / std( t)
        m = m / std( m)

        # Pack 'em
        if self.ae:
            return torch.FloatTensor( t), torch.FloatTensor( t)
        else:
            return torch.FloatTensor( t+m), torch.FloatTensor( t)

    # Return a validation set
    def getvals( self, val=2):
        # Go through all the validation files
        X, Y = [], []
        for i in range( len( self.val[0])):
            # Load them
            t,_ = librosa.core.load( self.val[0][i], sr=self.sr, res_type='kaiser_fast')
            m,_ = librosa.core.load( self.val[1][i], sr=self.sr, res_type='kaiser_fast')

            # Clip to shortest length
            if len( t) > len( m):
                t = t[:len(m)]
            else:
                m = m[:len(t)]

            # Trim length to something we can STFT cleanly
            t = t[:self.tq*(len(t)/self.tq)]
            m = m[:self.tq*(len(t)/self.tq)]

            # Make them unit variance
            t = t / std( t)
            m = m / std( m)

            # Add the mixture and the target/masker pair to the list
            if self.ae:
                X.append( torch.FloatTensor( t))
                Y.append( vstack( (t,m)))
            else:
                X.append( torch.FloatTensor( t+m))
                Y.append( vstack( (t,m)))

        return X,Y


# Define the network's loss functions

# SDR
def sdr_loss(out, targ):
    l = -torch.mean( out * targ)**2 / (torch.mean( out**2) + 2e-7)

def sir_loss(out, targ, interf):
    l = -torch.mean( out * interf)**2 / (torch.mean( out * targ) + 2e-7)
    return l

# To be added
def sar_loss(out, targ, interf):

# To be added
def stoi_loss(out, targ, interf):

# Composite Loss functions
# To be added


def mse_loss(out, targ):
    l = torch.mean( torch.pow( out-targ, 2.))
    return l

def l1_regularizer_loss( tr, wt):
    return wt * torch.mean( torch.abs( tr))

def l2_regularizer_loss( tr, wt):
    return wt * torch.mean( torch.abs( tr)**2)




def myloss( out, targ, tr=None, mse=False):
    if mse:
        # MSE
        l = torch.mean( torch.pow( out-targ, 2.))
    else:
        # SDR
        l = -torch.mean( out * targ)**2 / (torch.mean( out**2) + 2e-7)

    # Add some regularization
    if tr is not None:
        l += .1*torch.mean( torch.abs( tr))
    return l


# Define the network
class fd_snn_t( nn.Module):
    # Initialize
    def __init__( self, ft_size=1024, sep_sizes=[512], hop=256,
        dropout=0., adapt_fe=False, smoother=5, ortho=False, masking=False):
        super( fd_snn_t, self).__init__()

        # Remember these
        self.sz = ft_size
        self.hp = hop
        self.adapt_fe = adapt_fe
        self.masking = masking

        # Transform constants
        wn = hanning( ft_size+1)[:-1]**.5
        f = fft( eye( ft_size))
        if adapt_fe:
            f = vstack( (real( f[:int(ft_size/2),:]),imag(f[:int(ft_size/2),:])))
        else:
            f = vstack( (real( f[:int(ft_size/2+1),:]),imag(f[:int(ft_size/2+1),:])))
        self.ft = torch.nn.Parameter( torch.FloatTensor( 0.001 * f[:,None,:]), requires_grad=adapt_fe)
        if ortho:
            # pdb.set_trace()
            self.it = self.ft # .permute(2,1,0)
        else:
            self.it = torch.nn.Parameter( torch.FloatTensor( wn * pinv( f).T[:,None,:]), requires_grad=adapt_fe)

        # Smoother?
        if adapt_fe:
            self.bn = nn.BatchNorm1d( ft_size)
            self.l = smoother/2
            self.l1 = self.l if (smoother%2 != 0) else (self.l-1)
            self.sm = nn.Conv1d( ft_size, ft_size, smoother, groups=ft_size, padding=smoother-1)

        # Separator
        if adapt_fe:
            n = [ft_size] + sep_sizes + [ft_size]
        else:
            n = [int(ft_size/2)+1] + sep_sizes + [int(ft_size/2)+1]
        self.sep = nn.ModuleList([])
        for i in range( len( n)-1):
            ll = nn.Linear( n[i], n[i+1])
            self.sep.append( ll)

        # To be used throughout
        self.dp = nn.Dropout( dropout)


    # Define the forward pass, all data is (batch x time x dimensions)
    def forward( self, x):
        # Cache size, input is (bt x time)
        bt = x.size(0)
        T = x.size(1)

        # Do some dropout on the input
        x = self.dp( x)

        # Reshape to facilitate transform
        x = x.view( bt, 1, T)

        # Forward transform, gives (bt x sz x time)
        tx = F.conv1d( x, self.ft, stride=self.hp, padding=self.sz)

        # DFT or not?
        if not self.adapt_fe:
            # Get magnitude and phase
            a = torch.sqrt( tx[:,:int(self.sz/2)+1,:]**2 + tx[:,int(self.sz/2)+1:,:]**2)
            p = Variable( torch.atan2( tx[:,int(self.sz/2)+1:,:].data, tx[:,:int(self.sz/2)+1,:].data), requires_grad=False)
        else:
            tx = self.bn( tx)
            # Rectify and smooth
            txs = F.softplus( self.sm( torch.abs( tx))[:,:,int(self.l1):-int(self.l)])

            # Split to modulator and carrier
            a = txs
            p = tx / (a+2e-7)

        # Convert from (bt x dim x time) to (time*bt x dim) for dense layers
        x = a.permute( 0, 2, 1).contiguous().view( -1, a.size(1))

        # Is it not a mask?
        if not self.masking:

            # Run dense layers, softplus & dropout them
            for l in self.sep:
                x = self.dp( F.softplus( l( x)))

        else:
             # Run dense layers, softplus & dropout them
             for l in self.sep[:-1]:
                 x = self.dp( F.softplus( l( x)))

            # Apply final dense layer, sigmoid & dropout
            x =  self.dp( F.sigmoid( self.sep[-1]( x)))


        # Change to Conv1 format again, from (bt*time x dim) to (bt x dim x time)
        x = x.view( bt, -1, x.size(1)).permute( 0, 2, 1).contiguous()

        if not self.masking:
            # Remodulate
            if not self.adapt_fe:
                x = torch.cat( [x*torch.cos( p), x*torch.sin( p)], dim=1)
            else:
                x = x * p

        else:
            # Remodulate
            if not self.adapt_fe:
                x = torch.cat( [x * a*torch.cos( p), x * a*torch.sin( p)], dim=1)
            else:
                x = x * tx

        x = self.dp( x)

        # Resynth (use 2d until pytorch is updated with 1d fix)
        x = F.conv_transpose1d( x, self.it, stride=self.hp, padding=self.sz)

        # Return output and fwd transform magnitudes
        # return x.view( bt), tx
        return x.view( bt, -1), tx

# Evaluate on validation set
def evaluate( net, x, y, fn='tsep'):
    import pdb
    r = array( [])
    e = array( [0.,0.,0.])
    for i in range( len( x)):
        z,_ = net.forward( Variable( x[i].unsqueeze(0)).type_as( next( net.parameters())))
        z = z[0].data.cpu().numpy()
        e += bss_eval( z, 0, y[i])
        r = hstack( (r,x[i].numpy()/max(1.1*abs(x[i].numpy())),z/max(1.1*abs(z))))
    # pdb.set_trace()
    librosa.output.write_wav( 'tsep_mse.wav', r, 16000)
    return e / len( x)


# Visdom plotting routine
def trainplot( e, ve, sxr, vis, win=[None,None], eid=None):
    # Loss plots
    data = [
        # Loss
        dict(
            x=range( len( e)), y=e, name='Training: ' + str( e[-1]),
            hoverinfo='y', line=dict( width=1), mode='lines', type='scatter'),

        # Validation loss
        # dict(
        #     x=range( 0, bt*len(ve), bt), y=ve, name='Validation: '+str( ve[-1]),
        #     hoverinfo='y', line=dict( width=1), mode='lines', type='scatter')
    ]
    layout = dict(
        showlegend=True,
        legend=dict( orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict( r=30, b=40, l=50, t=50),
        font=dict( family='Bell Gothic Std'),
        xaxis=dict( autorange=True, title='Training samples'),
        yaxis=dict( autorange=True, title='Loss'),
        title='Losses',
    )
    vis._send( dict( data=data, layout=layout, win=win[0], eid=eid))

    # BSS_EVAL plots
    data2 = [
        # SDR
        dict(
            x=range( 0, len(e), int(len(e)/len( sxr))), y=[i[0] for i in sxr], name='SDR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SIR
        dict(
            x=range( 0, len(e), int(len(e)/len( sxr))), y=[i[1] for i in sxr], name='SIR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SAR
        dict(
            x=range( 0, len(e), int(len(e)/len( sxr))), y=[i[2] for i in sxr], name='SAR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter')
    ]
    layout2 = dict(
        showlegend=True,
        legend=dict( orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict( r=30, b=40, l=50, t=50),
        font=dict( family='Bell Gothic Std'),
        xaxis=dict( autorange=True, title='Training samples'),
        yaxis=dict( autorange=True, title='dB'),
        title='BSS_EVAL'
    )

    vis._send( dict( data=data2, layout=layout2, win=win[1], eid=eid))


def main():
    import argparse
    parser = argparse.ArgumentParser( description='tsep network')

    # Transform options
    parser.add_argument( '--filters', '-f', type=int, default=1024,
                        help='Number of filters in front end')
    parser.add_argument( '--fourier', '-dft', action='store_true', default=False,
                        help='Use a Fourier transform')
    parser.add_argument( '--mask', '-mask', action='store_true', default=False,
                        help='Apply a mask')
    parser.add_argument( '--orthogonal', '-or', action='store_true', default=False,
                        help='Use an "orthogonal" transform')
    parser.add_argument( '--hop', '-hp', type=int, default=16,
                        help='Transform hop/stride')

    # Network options
    parser.add_argument( '--smoother', '-sm', type=int, default=5,
                        help='Smoothing layer length')
    parser.add_argument( '--denoisersize', '-ds', type=int, default=[1024],
                        nargs='*', help='Denoiser layer sizes')

    # Training options
    parser.add_argument( '--learningrate', '-lr', type=float, default=.001,
                        help='Learning rate')
    parser.add_argument( '--batchsize', '-b', type=int, default=16,
                        help='Batch size')
    parser.add_argument( '--dropout', '-d', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument( '--iterations', '-it', type=int, default=2,
                        help='Number of training samples to learn from')
    parser.add_argument( '--name', '-n', default=None,
                        help='Model name')
    parser.add_argument( '--mse', '-mse', action='store_true', default=False,
                        help='Use MSE instead of SDR')

    # Get the arguments
    args = parser.parse_args()

    # Optionally dump arguments to a log file
    if args.name is not None:
        f = open( args.name + '-args.txt', 'w')
        for keys,values in vars( args).items():
            f.write( '%13s: %s\n' % ((keys), str( values)))
        f.close()

    # Instantiate network

    net = fd_snn_t( ft_size=args.filters, hop=args.hop, smoother=args.smoother, sep_sizes=args.denoisersize, dropout=args.dropout, adapt_fe=not args.fourier, ortho=args.orthogonal, masking=args.mask)
    net = torch.nn.DataParallel( net, device_ids=[0,1])
    net = net.cuda()

    # Select data
    random.seed(25)
    M = Mix_Dataset('/usr/local/timit/timit-wav/train/dr1/f*',
                    '/usr/local/timit/timit-wav/train/dr1/m*',
                   # '/usr/local/snd/Nonspeech/',
                   val=16)
    MI = DataLoaderIter( DataLoader( M, batch_size=args.batchsize, num_workers=0, pin_memory=True))

    # Setup optimizer
    opt = torch.optim.Adam( filter( lambda p: p.requires_grad, net.parameters()), lr=args.learningrate)

    # Initialize these for training
    vis = visdom.Visdom( port=5800)

    # Get validation data
    xv,yv = M.getvals()

    # Clear performance metrics
    e, be = [], []

    # Training loop
    pb = trange( args.iterations, unit='s', unit_scale=True, mininterval=.5, smoothing=.9)
    pli = 30 # Plotting/validation interval in seconds

    # Return bases ordered in mag freq domain
    def frbases( w):
        fw = abs( rfft( w, axis=1))**2
        fw /= (fw.max( axis=1)[:,None] + 2e-7)
        fw[:,0] = 0 # ignore DC
        i = argsort( argmax( fw, axis=1))
        return fw[i,:].T

    lt = time.time()-2*pli
    try:
        it = 0
        while it <= args.iterations:

            # Get data and move to GPU
            x,y = next( MI)
            inps = Variable( x).type_as( next( net.parameters()))
            target = Variable( y).type_as( next( net.parameters()))

            # Get loss
            net.train()
            z,h = net( inps)
            loss =

            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Report
            net.eval()
            e.append( abs( loss.data[0]))

            # Test on validation data
            if time.time() - lt > pli or it == args.iterations:
                be += [list( evaluate( net, xv, yv, args.name))]
                trainplot( e, [], be, vis, win=['loss','bss'], eid=args.name)

                # Save training metrics for posterity
                if args.name is not None:
                    savez( args.name, loss=e, bss=be)
                lt = time.time()

            # Update the progress bar
            pb.set_description( 'L:%.3f P:%.1f/%.1f/%.1f' %
                  (e[-1], be[-1][0], be[-1][1], be[-1][2]))
            pb.update( args.batchsize)
            it += args.batchsize

    except KeyboardInterrupt:
        pass


# Run it!
if __name__ == "__main__":
    main()
