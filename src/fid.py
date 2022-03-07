
from cleanfid import fid
#FID
score_FID = fid.compute_fid('/content/Attention2Angio/Results/fake', '/content/Attention2Angio/Results/real_target')
   
print(score_FID)

#KID
score_KID = fid.compute_kid('/content/Attention2Angio/Results/fake', '/content/Attention2Angio/Results/real_target')
    
print(score_KID)