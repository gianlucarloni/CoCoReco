import numpy as np
import torch

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

##### Some utility classes ####################################################################
class STEFunction(torch.autograd.Function):
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x

##### Causality blocks ####################################################################

class CausalityMapBlock(nn.Module):
    def __init__(self):

        '''
        Block to compute the causality maps of the deep neural network's latent features
        Inpired and adapted from the following papers:
            1- "Carloni, G., & Colantonio, S. (2024). Exploiting causality signals in medical images: A pilot study with empirical results. Expert Systems with Applications, 123433."
            
            2- "Carloni, G., Pachetti, E., & Colantonio, S. (2023). Causality-Driven One-Shot Learning for Prostate Cancer Grading from MRI. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 2616-2624)."

            3- "Carloni, G., Tsaftaris A., S., & Colantonio, S. (2024) CROCODILE: Causality aids RObustness via COntrastive DIsentangled LEarning" (accpeted at MICCAI 2024 International Workshop UNSURE, 10 Oct 2024)
            
        To lower the computational burden, we leverage only the 'max' option instead of the 'lehmer' one (possible future extension of this work: utilize lehmer mean).
        '''
        super(CausalityMapBlock, self).__init__()
        print("CausalityMapBlock initialized")
        
    def forward(self,x): #(bs,k,n,n)   batch size, number of feature maps, spatial dimensions of the feature maps
        
        if torch.isnan(x).any():
            print(f"...the current feature maps object contains NaN")
            raise ValueError
        maximum_values = torch.max(torch.flatten(x,2), dim=2)[0]  #flatten: (bs,k,n*n), max: (bs,k) 
        MAX_F = torch.max(maximum_values, dim=1)[0]  #MAX: (bs,) 
        x_div_max=x/(MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) +1e-8) #TODO added epsilon; #implement batch-division: each element of each feature map gets divided by the respective MAX_F of that batch
        x = torch.nan_to_num(x_div_max, nan = 0.0)

        ## After having normalized the feature maps, comes the distinction between the method by which computing causality.
        #Note: to prevent ill posed divisions and operations, we sometimes add a small epsilon (e.g., 1e-8) and nan_to_num() command.

        sum_values = torch.sum(torch.flatten(x,2), dim=2)
        if torch.sum(torch.isnan(sum_values))>0:
            sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
        
        maximum_values = torch.max(torch.flatten(x,2), dim=2)[0]  
        mtrx = torch.einsum('bi,bj->bij',maximum_values,maximum_values) #batch-wise outer product, the max value of mtrx object is 1.0
        tmp = mtrx/(sum_values.unsqueeze(1) +1e-8) #TODO added epsilon
        causality_maps = torch.nan_to_num(tmp, nan = 0.0)

        # To avoid collapse, we could add a small penalty term to the causality maps
        # causality_maps = causality_maps + torch.randn_like(causality_maps)*1e-6 #add some noise to avoid zero values

        # now we scale them between 0 and 1 so that they can be used without bringing to explosion/collapse of the values
        max_cmaps = torch.max(causality_maps, dim=1, keepdim=True)[0]
        min_cmaps = torch.min(causality_maps, dim=1, keepdim=True)[0]
        causality_maps = (causality_maps - min_cmaps) / (max_cmaps - min_cmaps + 1e-8)
        print(f"causality_maps: min {torch.min(causality_maps)}, max {torch.max(causality_maps)}, mean {torch.mean(causality_maps)}")
        
        return causality_maps # shape: (bs, k, k)
    
class CausalityFactorsExtractor(nn.Module):
    def __init__(self):

        '''
        Given a causality map, this block computes the causality factors for enhancing the feature maps according to their causal influence in the scene in an attention-like fashion.
        
        Inpired and adapted from the following papers:
            1- "Carloni, G., & Colantonio, S. (2024). Exploiting causality signals in medical images: A pilot study with empirical results. Expert Systems with Applications, 123433."
            
            2- "Carloni, G., Pachetti, E., & Colantonio, S. (2023). Causality-Driven One-Shot Learning for Prostate Cancer Grading from MRI. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 2616-2624)."            
        
        In this implementation, we consider a causality direction "causes" and propose a new setting we name "muladd", which first multiplies the feature maps by the causality factors and then adds the original feature maps to the product.
        This way, some feature maps are not enhanced (weight of zero, they pass as they are), while others are enhanced by a factor that depends on their causal influence in the scene. It is a sort of attention module.
        '''
        
        super(CausalityFactorsExtractor, self).__init__()
        self.STE = StraightThroughEstimator() #Bengio et al 2013
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        print("CausalityFactorsExtractor initialized")

    def forward(self, x, causality_maps):
        '''
        x [bs, k, h, w]: the feature maps from the convolutional network.;
        causality_maps [bs, k, k]: the output of a CausalityMapsBlock().

        By leveraging algaebric transformations and torch functions, we efficiently extracts the causality factors with few lines of code.

        In this implementation, we scale the values of the causality factors between 0 and 1.
        '''
        triu = torch.triu(causality_maps, 1) #upper triangular matrx (excluding the principal diagonal)
        tril = torch.tril(causality_maps, -1).permute((0,2,1)).contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0,2,1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f #sum of booleans is the OR logic

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        multiplicative_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
        
        multiplicative_factors = self.relu(multiplicative_factors) #we do not want negative weights
        # now we scale the factors between 0 and 1 so that they can be used as weights withour bringing to explosion of the values
        max_factors = torch.max(multiplicative_factors, dim=1, keepdim=True)[0]
        min_factors = torch.min(multiplicative_factors, dim=1, keepdim=True)[0]
        multiplicative_factors = (multiplicative_factors - min_factors) / (max_factors - min_factors + 1e-8)
        print(f"multiplicative_factors: min {torch.min(multiplicative_factors)}, max {torch.max(multiplicative_factors)}, mean {torch.mean(multiplicative_factors)}")

        ## Directly return the "attended" ("causally"-weighted) version of x. Later in the code, we will add this to the original feature maps, x.
        # We efficiently utilize einsum operation to multiply each (rectified) factor for the corresponding 2D feature map, for every minibatch, in a single pass
        return torch.einsum('bkmn,bk->bkmn', x, multiplicative_factors)


###### CoCoReco ####################################################################

class WeightedFeatureProjection(nn.Module):
    # This is our (learnable) skip connection to emulate extensive afferent end efferent connections of human visual areas. The modulation weight is biologically motivated by recent connectome studies.
    def __init__(self, n_features_desired, n_features_incoming, direction, output_size, weight = 0.10):
        super(WeightedFeatureProjection, self).__init__()
        if n_features_incoming == n_features_desired:
            self.adjustChannels = nn.Identity()
        else:
            self.adjustChannels = nn.Conv2d(in_channels=n_features_incoming, out_channels=n_features_desired, kernel_size=1, padding=0, bias=False)
        
        self.direction = direction # forward (early layer informs deeper) or backward (deeper layer informs earlier)
        if self.direction == 'forward':
            self.adjustSize = nn.AdaptiveAvgPool2d(output_size=(output_size, output_size)) # average pooling to reduce the size of the features
        elif self.direction == 'backward':
            self.adjustSize = nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=True) # bilinear upsampling to increase the size of the features
        self.weight = weight

    def forward(self, x):
        x = self.adjustChannels(x) # project the incoming features to the desired number of features
        x = self.weight*self.adjustSize(x) # adjust the size of the features and weight them
        return x
    
class CoCoReco(nn.Module):
    def __init__(self, num_classes=10, initial_HW_size=160, use_causality=True):
        super(CoCoReco, self).__init__()
        self.num_classes = num_classes
        self.initial_HW_size = initial_HW_size
        self.use_causality = use_causality

        # depending on the depth of the network, the size of the feature map at the fully connected layer will vary.
        # In our setting, we have 5 pooling layers, so the size of the feature map at the fully connected layer will be 1/2^5 of the initial size
        self.size_at_fc = int(np.floor(self.initial_HW_size/128)) 

        ## Utility layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Retina-LGN-V1 pathway
        self.retina = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.lgn = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.v1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Ventral pathway
        self.v2_thin = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.v4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.v8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.it = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # Dorsal pathway
        self.v2_thick = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.v3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # (ventro-dorsal stream)
        self.v5_mt = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.mst = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.ipl = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)
        # (dorso-dorsal stream)
        self.v3a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.v6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spl= nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)

        # Pre-frontal cortex
        self.pfc = nn.Conv2d(in_channels=64, out_channels=1024, kernel_size=3, stride=1, padding=1)

        ## superior colliculus and pulvinar
        self.sc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) # differently from the LGN layer, this layer has a larger kernel size and higher padding to simulate the retinotopic map and fast processing (M-cell path) of the superior colliculus
        self.pulvinar = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # TODO the same for the pulvinar
        
        ## Decision making at the inferior temporal cortex
        self.fc = nn.Linear(1024 * self.size_at_fc * self.size_at_fc, num_classes)

        ## Weighted feature projection layers
        self.sc2pulvinar = WeightedFeatureProjection(n_features_desired=16, n_features_incoming=32, direction='backward', output_size=round(initial_HW_size/2), weight=1.0)
        self.sc2lgn = WeightedFeatureProjection(n_features_desired=16, n_features_incoming=32, direction='backward', output_size=round(initial_HW_size/2), weight=0.10)
        self.lgn2v2_thin = WeightedFeatureProjection(n_features_desired=64, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/8), weight=1.0)
        self.lgn2v4 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/16), weight=1.0)
        
        self.pulvinar2v2thick = WeightedFeatureProjection(n_features_desired=64, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/8), weight=1.0) # this is actually identical to self.lgn2v2_thin
        self.pulvinar2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/32), weight=1.0)

        self.v12v4 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/16), weight=0.05)
        self.v12v8 = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/32), weight=0.035)
        self.v2thin2v8 = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/32), weight=0.065)
        self.v2thin2it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/64), weight=0.025)
        self.v42it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/64), weight=0.07)

        self.lgn2v2thick = WeightedFeatureProjection(n_features_desired=64, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/8), weight=1.0) # this is actually identical to self.lgn2v2_thin
        self.lgn2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/32), weight=1.0)
        self.v12v3 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/16), weight=0.12)
        self.v12v3a = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/32), weight=0.05)
        self.v12v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/64), weight=0.05)
        self.v2thick2v3a = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/32), weight=0.085)
        self.v2thick2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/32), weight=0.015)
        self.v2thick2v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/64), weight=0.07)

        self.v32v4 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=256, direction='backward', output_size=round(initial_HW_size/16), weight=0.12)
        self.v32it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/64), weight=0.05)
        self.v32mst = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/64), weight=0.02)
        self.v32v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/64), weight=0.025)

        self.v3a2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=512, direction='backward', output_size=round(initial_HW_size/32), weight=0.02)

        self.v62v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=1024, direction='backward', output_size=round(initial_HW_size/32), weight=0.025)
        self.v42mst = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/64), weight=0.025)
    
        self.pfc2it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=1024, direction='backward', output_size=round(initial_HW_size/64), weight=1.0)
        self.pfc2pc = WeightedFeatureProjection(n_features_desired=1024, n_features_incoming=1024, direction='forward', output_size=round(initial_HW_size/128), weight=1.0)
        self.pc2it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=2048, direction='backward', output_size=round(initial_HW_size/64), weight=1.0)

        # causality blocks
        if self.use_causality:
            self.causality_map_block = CausalityMapBlock()
            self.causality_factors_extractor = CausalityFactorsExtractor()

    def forward(self, x): # At each feature layer, we alternate conv layers, relu, and pooling layers sequentially
        ###
        v1_enhanced=None
        pfc_enhanced=None
        it_enhanced=None
        ###
        x = self.pool(self.relu(self.retina(x)))

        ## The first firing is sent to the superior colliculus and pulvinar (M-cell pathway, 10% of retinal axons) and to the LGN (P-cell pathway, 90% of retinal axons)

        # 10% of the information is sent to the superior colliculus and pulvinar (tectopulvinar pathway)
        sc = self.pool(self.relu((self.sc(0.10*x))))
        # the pulvinar receives from the retina and the superior colliculus itself
        pulvinar = self.pool(self.relu(self.pulvinar(0.10*x + self.sc2pulvinar(sc))))
        # 90% of the axons send retina information to the geniculostriate pathway
        lgn = self.pool(self.relu(self.lgn(0.90*x + self.sc2lgn(sc))))

        ## At this point, the signal is sent to the primary visual cortex via a joint M-and-P-cell pathway
        v1 = self.pool(self.relu(self.v1(lgn)))
        if self.use_causality:
            # The signal at V1 (represented as feature maps of shape (batch_size, num_fmaps, h_fmaps, w_fmaps)) is used to study the causality maps and factors
            v1_causality_maps = self.causality_map_block(v1) # the output of the causality map block is a tensor of shape (batch_size, num_fmaps, num_fmaps)
            v1_enhanced = self.causality_factors_extractor(v1, v1_causality_maps) # the output of the causality factors extractor is a tensor of shape (batch_size, num_fmaps, h_fmaps, w_fmaps)
            # We conceive the addition of the original feature maps to the causally-enhanced feature maps as a sort of residual connection and attention mechanism
            v1 = v1 + v1_enhanced
        else:
            v1_causality_maps=None

        ## Now, a fast M pathway sends information directly to the high level areas, prefrontal cortex. It contains low spatial frequency information
        pfc = self.pool(self.relu(self.pfc(v1)))
        if self.use_causality:
            # The signal at PFC is used to study the causality maps and factors
            pfc_causality_maps = self.causality_map_block(pfc) # 
            pfc_enhanced = self.causality_factors_extractor(pfc, pfc_causality_maps) # 
            pfc = pfc + pfc_enhanced
        else:
            pfc_causality_maps=None

        # Contextually, from the primary visual cortex, the information is sent to the ventral and dorsal pathways
        v2_thin = self.pool(self.relu(self.v2_thin(v1 + self.lgn2v2_thin(lgn))))
        v2_thick = self.pool(self.relu(self.v2_thick(v1 + self.lgn2v2thick(lgn) + self.pulvinar2v2thick(pulvinar))))

        # On the dorsal stream, the signal is sent from the V2_thick to the V3 area (then to the V3a, V5mt, MST, V6, and finally to the parietal cortex)
        v3 = self.pool(self.relu(self.v3(v2_thick + self.v12v3(v1))))
        # On the ventral stream, the signal is sent from the V2_thin to the V4 area (then to the V8 area, and finally to the inferior temporal cortex)
        v4 = self.pool(self.relu(self.v4(v2_thin + self.lgn2v4(lgn) + self.v12v4(v1) + self.v32v4(v3))))
        
        v8 = self.pool(self.relu(self.v8(v4 + self.v12v8(v1) + self.v2thin2v8(v2_thin) + 0.075*v3)))
        v3a = self.pool(self.relu(self.v3a(0.075*v3 + self.v12v3a(v1) + self.v2thick2v3a(v2_thick) + 0.03*v4)))    
        
        v6 = self.pool(self.relu(self.v6(v3a + self.v12v6(v1) + self.v2thick2v6(v2_thick) + self.v32v6(v3))))

        v5_mt = self.pool(self.relu(self.v5_mt(v3 + self.lgn2v5mt(lgn) + self.v2thick2v5mt(v2_thick) + self.pulvinar2v5mt(pulvinar) + self.v3a2v5mt(v3a) + self.v62v5mt(v6))))
        
        mst = self.pool(self.relu(self.mst(v5_mt + self.v32mst(v3) + 0.02*v3a + self.v42mst(v4))))
        

        # Parietal cortex receives from the bottomup dorsal stream and from the topdown prefrontal cortex
        ipl = self.pool(self.relu(self.ipl(mst + self.pfc2pc(pfc))))      
        spl = self.pool(self.relu(self.spl(v6 + self.pfc2pc (pfc))))

        # All the signals converge to the inferior temporal cortex: bottom up processes from the ventral and dorsal streams, and top down processes from the prefrontal and parietal cortices       
        it = self.pool(self.relu(self.it(v8 + self.v2thin2it(v2_thin) + self.v42it(v4) + self.v32it(v3) + self.pfc2it(pfc) + self.pc2it(ipl) + self.pc2it(spl))))
        
        if self.use_causality:
            # The signal at IT (represented as feature maps of shape (batch_size, num_fmaps, h_fmaps, w_fmaps)) is used to study the causality maps and factors
            it_causality_maps = self.causality_map_block(it) # the output of the causality map block is a tensor of shape (batch_size, num_fmaps, num_fmaps)
            it_enhanced = self.causality_factors_extractor(it, it_causality_maps) # the output of the causality factors extractor is a tensor of shape (batch_size, num_fmaps, h_fmaps, w_fmaps)
            # We conceive the addition of the original feature maps to the causally-enhanced feature maps as a sort of residual connection and attention mechanism
            it = it + it_enhanced
        else:
            it_causality_maps=None

        # and the actual decision is made at its fully connected layer
        x = self.fc(it.flatten(start_dim=1))
        
        # return x, it_causality_maps, v1_causality_maps, pfc_causality_maps

        return x, it_causality_maps, v1_causality_maps, pfc_causality_maps, v1-v1_enhanced, v1_enhanced, pfc-pfc_enhanced, pfc_enhanced, it-it_enhanced, it_enhanced


#%% To compare our CoCoReco with a baseline model, we can use a from-scratch CNN of the same depth: 7 convolutional layers
class ScratchCNN(nn.Module):
    def __init__(self, num_classes=10, initial_HW_size=160):
        super(ScratchCNN, self).__init__()
        self.num_classes = num_classes
        self.initial_HW_size = initial_HW_size
        self.size_at_fc = int(np.floor(self.initial_HW_size/128)) 
        ## Utility layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # let us build a CNN with the 7 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(1024 * self.size_at_fc * self.size_at_fc, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))
        x = self.pool(self.relu(self.conv7(x)))
        x = self.fc(x.flatten(start_dim=1))
        return x, None, None, None
