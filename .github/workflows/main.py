import tqdm
from torch.optim import Adam, SGD
from model import *
from utils import *
import opt

def train(X, y, A, a):
    opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1 = 0, 0, 0, 0
    
    A_sl = a * A + np.eye(A.shape[0])

    if opt.args.is_pass != 0:
        if opt.args.is_pass == 1:
            adjs = get_adjs(A)
            for a in adjs:
                X = a.dot(X)
        else:
            adjs = get_laps(A)
            for a in adjs:
                X = a.dot(X)

    enc_dims = [opt.args.n_input] + opt.args.enc_dims
    dec_dims = opt.args.dec_dims + [opt.args.n_input]
    model = OUR(opt.args.layers, enc_dims, dec_dims).to(opt.args.device)

    X = numpy_to_torch(X).to(opt.args.device)
    A_sl = numpy_to_torch(A_sl).to(opt.args.device)
    
    centers = model_init(model, X, y)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    
    for epoch in range(opt.args.epoch):
        # input & output
        Z1, Z2, Z, Q, X_ = model(X)
        P = target_distribution(Q)

        loss_cv = cross_view_loss(Z1, Z2, A_sl)
        loss_kl = distribution_loss(Q, P)
        loss_rec = reconstruction_loss(X, X_)

        # print(loss_cv, loss_kl)
        loss = loss_cv + 10 * loss_kl + loss_rec
        
        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        acc, nmi, ari, f1, _ = clustering(Z, y)
        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.f1 = f1
            print(epoch, "ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1))
        
    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1

if __name__ == '__main__':
    # initialize
    setup()

    # load data
    X, y, A = load_graph_data(opt.args.name)

    acc, nmi, ari, f1 = train(X, y, A, 1.0)
    print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1))

