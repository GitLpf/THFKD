# THFKD
CIFAR-100 Classification
Please refer to the following content for details on the training of the pre-trained teacher model and student baseline.
python train.py --model wrn-16-2 --suffix baseline1
python train.py --model wrn-40-1 --suffix baseline1
python train.py --model wrn-40-2 --suffix baseline1
python train.py --model resnet8x4 --suffix baseline1
python train.py --model resnet32x4 --suffix baseline1
python train.py --model resnet20 --suffix baseline1
python train.py --model resnet32 --suffix baseline1
python train.py --model resnet56 --suffix baseline1
python train.py --model resnet110 --suffix baseline1

Please refer to the following instructions for training the THFKD model.
  python train.py --model wrn-16-2 --teacher wrn-40-2 --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt --suffix THFKD1
python train.py --model wrn-40-1 --teacher wrn-40-2 --teacher-weight checkpoints/cifar100_wrn-40-2__baseline1_best.pt --suffix THFKD1
python train.py --model resnet8x4 --teacher resnet32x4 --teacher-weight checkpoints/cifar100_resnet32x4__baseline1_best.pt  --suffix THFKD1
python train.py --model shufflev2 --teacher resnet32x4 --teacher-weight checkpoints/cifar100_resnet32x4__baseline1_best.pt --suffix THFKD1
python train.py --model resnet20 --teacher resnet56 --teacher-weight checkpoints/cifar100_resnet56__baseline1_best.pt  --suffix THFKD1
python train.py --model resnet32 --teacher resnet110 --teacher-weight checkpoints/cifar100_resnet110__baseline1_best.pt --suffix THFKD1
