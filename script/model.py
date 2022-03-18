# https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp
from torchsummary import summary
UNet_ResNet = smp.Unet(
    encoder_name='resnet152',
    encoder_weights='imagenet', # pre_training on ImageNet
    in_channels=1, 
    classes=1
)
# print(UNet_ResNet)

# UNet_ResNet.cuda()
# summary(UNet_ResNet, (1,224,224))

UNet_ResNet1 = smp.UnetPlusPlus(
    encoder_name='resnet152',
    encoder_weights='imagenet', # pre_training on ImageNet
    in_channels=1, 
    classes=1
)