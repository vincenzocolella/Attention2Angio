
from cleanfid import fid

#FID

def compute_fid():
    score_FID = fid.compute_fid('/content/Attention2Angio/Results/real_target','/content/Attention2Angio/Results/fake',mode="legacy_tensorflow")
    
    return score_FID

#KID

def compute_kid():
    score_KID = fid.compute_kid('/content/Attention2Angio/Results/real_target','/content/Attention2Angio/Results/fake',mode="legacy_tensorflow")
        
    return score_KID
    
    
if __name__ == '__main__':
    print(compute_fid())
    print(compute_kid())