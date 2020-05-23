import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock

model_urls = {
    "resnet18":
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34":
    "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50":
    "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101":
    "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152":
    "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d":
    "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d":
    "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2":
    "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2":
    "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def init_weights(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_uniform_(m.weight)


class NonFcResNet(models.ResNet):
	def __init__(self, input_shape, output_length, block, layers, num_classes,
	             **kwargs):
		super(NonFcResNet, self).__init__(block, layers, num_classes, **kwargs)
		del self.fc
		del self.avgpool
		mid_shape = self.infer_fwp_shape(input_shape)  # mid-shape =[c,h,w]
		self.wReLu = torch.nn.LeakyReLU(inplace=True)
		self.height_reduce = torch.nn.Linear(mid_shape[1], 1)
		self.width_reduce = torch.nn.Sequential(
		    torch.nn.Linear(mid_shape[-1], output_length),
		    torch.nn.LeakyReLU(inplace=True),
		    torch.nn.Linear(output_length, output_length))

		self.ch2cls = torch.nn.Conv2d(in_channels=512 * block.expansion,
		                              out_channels=num_classes,
		                              kernel_size=1)

		self.apply(init_weights)

	def infer_fwp_shape(self, input_shape):
		"""
            return shape = [batch,channel_,width,height]
        """
		fake_data = torch.ones(1, *input_shape)
		return self.forward_part(fake_data).shape[1:]

	def forward_part(self, x):
		"""
        input_shape = [batch,channel,heigth,width]
        return shape = [batch,channel_,height,width]
        """
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x

	def forward(self, x):
		x = self.forward_part(x)
		# x.shape = [batch,channel`,height, width]

		x = self.wReLu(self.width_reduce(x))  #[batch,channel,height,width]
		x = self.wReLu(self.height_reduce(x.transpose(
		    -1, -2)))  #[batch,channel,width,1 ]
		x = self.wReLu(self.ch2cls(x))

		# shape = [batch,channel,width]
		x = x.squeeze(-1)
		x = torch.nn.functional.log_softmax(x, 1)
		return x


def make_model(arch,
               layers,
               num_classes,
               input_shape,
               label_length,
               pretrained,
               progress=True,
               **kwargs) -> torch.nn.Module:
	model = NonFcResNet(input_shape, label_length, BasicBlock, layers,
	                    num_classes, **kwargs)
	if pretrained:
		state_dict = models.load_state_dict_from_url(model_urls[arch],
		                                             progress=progress)
		model.load_state_dict(state_dict)
	return model


if __name__ == "__main__":
	import DataSet

	model = make_model("resnet50", [3, 4, 6, 3], DataSet.num_classes, False)
