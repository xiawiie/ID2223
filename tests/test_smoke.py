import torch
from src.models import CIFAR10AlexNet, CIFAR10ResNet, parameter_count

def test_model_shapes_and_part_a9_architecture():
    x = torch.randn(2, 3, 32, 32)
    alex = CIFAR10AlexNet()
    resnet = CIFAR10ResNet()
    assert alex(x).shape == (2, 10)
    assert resnet(x).shape == (2, 10)
    assert [m.out_channels for m in alex.features if isinstance(m, torch.nn.Conv2d)] == [64, 192, 384, 256, 256]
    assert [len(stage) for stage in (resnet.s1, resnet.s2, resnet.s3, resnet.s4)] == [2, 2, 2, 2]
    assert resnet.fc.in_features == 512
    assert parameter_count(alex) == 6976842
    assert parameter_count(resnet) == 11173962
