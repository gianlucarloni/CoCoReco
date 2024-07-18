import torch
import torch

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.resnet = models.resnet18(pretrained=False)
#         self.resnet.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.resnet(x)
#         return x

class WeightedFeatureProjection(nn.Module):
    def __init__(self, n_features_desired, n_features_incoming, direction, output_size, weight = 0.10):
        super(WeightedFeatureProjection, self).__init__()
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
    def __init__(self, num_classes=10, initial_HW_size=160):
        super(CoCoReco, self).__init__()
        self.num_classes = num_classes
        self.initial_HW_size = initial_HW_size

        ## Utiity layers
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
        self.sc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pulvinar = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.sc2pulvinar = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        ## Decision making at the inferior temporal cortex
        self.fc = nn.Linear(1024 * 3 * 3, num_classes)

        ## Weighted feature projection layers
        finire di sistemare le dimensioni X delle feature maps al denominatore: 

        self.sc2pulvinar = WeightedFeatureProjection(n_features_desired=16, n_features_incoming=32, direction='backward', output_size=round(initial_HW_size/X), weight=1.0)
        self.sc2lgn = WeightedFeatureProjection(n_features_desired=16, n_features_incoming=32, direction='backward', output_size=round(initial_HW_size/X), weight=0.10)
        self.lgn2v2_thin = WeightedFeatureProjection(n_features_desired=64, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/X), weight=1.0)
        self.lgn2v4 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/X), weight=1.0)
        self.v12v4 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/X), weight=0.05)
        self.v12v8 = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/X), weight=0.035)
        self.v2thin2v8 = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/X), weight=0.065)
        self.v2thin2it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/X), weight=0.025)
        self.v42it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/X), weight=0.07)

        self.lgn2v2thick = WeightedFeatureProjection(n_features_desired=64, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/X), weight=1.0)
        self.lgn2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=32, direction='forward', output_size=round(initial_HW_size/X), weight=1.0)
        self.v12v3 = WeightedFeatureProjection(n_features_desired=128, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/X), weight=0.12)
        self.v12v3a = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/X), weight=0.05)
        self.v12v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=64, direction='forward', output_size=round(initial_HW_size/X), weight=0.05)
        self.v2thick2v3a = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/X), weight=0.085)
        self.v2thick2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/X), weight=0.015)
        self.v2thick2v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=128, direction='forward', output_size=round(initial_HW_size/X), weight=0.07)

        self.v32v4 = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=128, direction='backward', output_size=round(initial_HW_size/X), weight=0.12)
        self.v32it = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/X), weight=0.05)
        self.v32mst = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=256, direction='forward', output_size=round(initial_HW_size/X), weight=0.02)
        self.v32v6 = WeightedFeatureProjection(n_features_desired=512, n_features_incoming=512, direction='forward', output_size=round(initial_HW_size/X), weight=0.025)

        self.v3a2v5mt = WeightedFeatureProjection(n_features_desired=256, n_features_incoming=512, direction='backward', output_size=round(initial_HW_size/X), weight=0.02)

    def forward(self, x): #alternate conv layers, relu, and pooling layers sequentially
        https://app.diagrams.net/#G1TIXjnv0Fq1S7Pk1PgXHxq67-Ix8KorPY#%7B%22pageId%22%3A%22R-Gjc8jRLmUtNvAPahg4%22%7D

        e sistemare l'ordine delle righe in base alla creazione delle feature maps per evitare chiamate alle variabili prima della creazione
        x = self.pool(self.relu(self.retina(x))) 

        # 10% of the information is sent to the superior colliculus and pulvinar (tectopulvinar pathway)
        sc = self.pool(self.relu(self.sc(x)))

        # the pulvinar receives from the retina and the superior colliculus
        pulvinar = self.pool(self.relu(self.pulvinar(x + self.sc2pulvinar(sc))))

        # 90% of the axons send retina information to the geniculostriate pathway
        lgn = self.pool(self.relu(self.lgn(x + self.sc2lgn(sc))))

        v1 = self.pool(self.relu(self.v1(lgn)))

        # Now, the information is sent to the ventral and dorsal pathways
        #   Ventral pathway
        v2_thin = self.pool(self.relu(self.v2_thin(v1 + self.lgn2v2_thin(lgn))))
        v4 = self.pool(self.relu(self.v4(v2_thin + self.lgn2v4(lgn) + self.v12v4(v1) + self.v32v4(v3))))
        v8 = self.pool(self.relu(self.v8(v4 + self.v12v8(v1) + self.v2thin2v8(v2_thin) + 0.075*v3)))
        it = self.pool(self.relu(self.it(v8 + self.v2thin2it(v2_thin) + self.v42it(v4) + self.v32it(v3))))
        #   Dorsal pathway
        v2_thick = self.pool(self.relu(self.v2_thick(v1 + self.lgn2v2thick(lgn))))
        v3 = self.pool(self.relu(self.v3(v2_thick + self.v12v3(v1))))
        #       Ventro-dorsal stream
        v5_mt = self.pool(self.relu(self.v5_mt(v3 + self.lgn2v5mt(lgn) + self.v2thick2v5mt(v2_thick))))
        mst = self.pool(self.relu(self.mst(v5_mt + self.v32mst(v3))))
        ipl = self.pool(self.relu(self.ipl(mst)))
        #       Dorso-dorsal stream
        v3a = self.pool(self.relu(self.v3a(v3 + self.v12v3a(v1) + self.v2thick2v3a(v2_thick) + 0.075*v3)))
        v6 = self.pool(self.relu(self.v6(v3a + self.v12v6(v1) + self.v2thick2v6(v2_thick) + self.v32v6(v3))))
        spl = self.pool(self.relu(self.spl(v6)))
        # Pre-frontal cortex
        pfc = self.pool(self.relu(self.pfc(v1)))

        
        

        x = self.fc(it.flatten(start_dim=1))
        return x
