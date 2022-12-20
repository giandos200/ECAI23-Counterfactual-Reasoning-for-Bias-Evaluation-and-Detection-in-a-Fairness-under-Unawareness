import pickle
import numpy as np
from tabulate import tabulate
from tqdm import tqdm


def get_result_from_pickle(path):
    with open(path,"rb") as file:
        return pickle.load(file)


def CFmetrics(Results, method, SF):
    corpus = []

    # for k in tqdm(Results.keys()):
    for k in tqdm(['LR','SVM','LGB','XGB','AdvDeb','LFERM']):
        corpus.append([method,k])
        risultati = get_result_from_pickle(Results[k])
        individual = []
        onlyNeg = []
        onlyPositive = []
        negPriv = []
        posPriv = []
        negUnpriv = []
        posUnpriv = []
        R = risultati[method]['sample_CF']
        for u in R:
            sample, y_real, result, y_sens, y_CF_sens, CF = u
            if y_real[SF].item() == y_sens.item():
                Flip = (y_CF_sens!=y_real[SF].item()).astype(int)
                individual.append(Flip)
                if result.item()==0:
                    onlyNeg.append(Flip)
                    if y_real[SF].item()==0:
                        negUnpriv.append(Flip)
                    else:
                        negPriv.append(Flip)
                elif result.item()==1:
                    onlyPositive.append(Flip)
                    if y_real[SF].item()==0:
                        posUnpriv.append(Flip)
                    else:
                        posPriv.append(Flip)

        # head1 = ['method','model','AUC', 'ACC', 'DSP', 'DEO', 'DAO',
        #          '$Flip_p$@10','$Flip_p$@50','$Flip_p$@100',
        #         '$Flip_u$@10','$Flip_u$@50','$Flip_u$@100',
        #          '$\Delta$@10','$\Delta$@50','$\Delta$@100' ]
        # head2 = [ '$nDCG_p$@10', '$nDCG_p$@50','$nDCG_p$@100',
        #           '$nDCG_u$@10', '$nDCG_u$@50','$nDCG_u$@100',
        #           '$\Delta_{nDCG}$@10', '$\Delta_{nDCG}$@50','$\Delta_{nDCG}$@100'
        #           ]
        head1 = ['method','model',
                 'CFlip_p@10','CFlip_p@50','CFlip_p@100',
                'CFlip_u@10','CFlip_u@50','CFlip_u@100',
                 'Delta@10','Delta@50','Delta@100' ]
        head2 = [ 'nDCG_p@10', 'nDCG_p@50','nDCG_p@100',
                  'nDCG_u$@10', 'nDCG_u@50','nDCG_u@100',
                  'Delta_nDCG@10', 'Delta_nDCG@50','Delta_nDCG@100'
                  ]
        Flip_priv = []
        Flip_unp = []
        Delta_Flip = []
        ndcg_priv = []
        ndcg_unp = []
        delta_ndcg = []
        for i in [10,50,100]:
            Flip = "{:.3f}".format(sum([u[:i].mean() for u in negUnpriv])/(negUnpriv.__len__()+1e-10)*100)
            #print(f'Percentage Flip Unprivileged at n°CF={i}, = {Flip}')
            Flip2 = "{:.3f}".format(sum([u[:i].mean() for u in negPriv])/(negPriv.__len__()+1e-10)*100)
            #print(f'Percentage Flip Privileged at n°CF={i}, = {Flip2}')
            #print(f'Difference between Priv and Unpriv at n°CF={i}, = {abs(Flip-Flip2)}')
            #totFlip = sum([u[:i].mean() for u in individual])/individual.__len__()
            #print(f'totalFlip at n°CF={i}, = {totFlip}')
            if float(Flip2)==0 and negPriv.__len__()==0 :
                Flip_priv.append('0^*')
            else:
                Flip_priv.append(Flip2)
            if float(Flip)==0 and negUnpriv.__len__()==0:
                Flip_unp.append('0^*')
            else:
                Flip_unp.append(Flip)
            Delta_Flip.append(abs(float(Flip)-float(Flip2)))

            IDCG = [(2**1-1)/np.log2(u+1) for u in range(1,i+1)]
            if negPriv.__len__()==0:
                nDCG_priv='0^*'
            else:
                nDCG_priv = "{:.4f}".format(np.mean([sum([(2**1-1)/np.log2(u+1) for u in range(1,min([i+1,u1.shape[0]])) if u1[u-1]==0])/sum(IDCG[:min([i,u1.shape[0]])]) for u1 in negPriv]))
            if negUnpriv.__len__()==0:
                nDCG_unpriv='0^*'
            else:
                nDCG_unpriv = "{:.4f}".format(np.mean([sum([(2**1-1)/np.log2(u+1) for u in range(1,min([i+1,u1.shape[0]])) if u1[u-1]==0])/sum(IDCG[:min([i,u1.shape[0]])]) for u1 in negUnpriv]))
            if negUnpriv.__len__()==0:
                if negPriv.__len__()==0:
                    nDCG_delta='0^*'
                else:
                    nDCG_delta=nDCG_priv
            elif negPriv.__len__()==0:
                nDCG_delta=nDCG_unpriv
            else:
                nDCG_delta = "{:.4f}".format(abs(float(nDCG_priv) - float(nDCG_unpriv)))
            ndcg_priv.append(nDCG_priv)
            ndcg_unp.append(nDCG_unpriv)
            delta_ndcg.append(nDCG_delta)
        l=[]
        [l.extend(lu) for lu in [Flip_priv,Flip_unp,Delta_Flip,ndcg_priv,ndcg_unp,delta_ndcg]]
        corpus[-1].extend(l)

    print(tabulate(corpus, headers=head1+head2, ))