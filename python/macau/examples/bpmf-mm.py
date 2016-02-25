import scipy.io
import macau
import sys

if len(sys.argv) < 3:
    print("Usage:\npython %s train.mm test.mm [num_latents]" % sys.argv[0])
    sys.exit(1)

train = scipy.io.mmread(sys.argv[1])
test  = scipy.io.mmread(sys.argv[2])

num_latent = 100

results = macau.bpmf(train, test, num_latent = num_latent, burnin=20, nsamples=10)

