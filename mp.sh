# model parallel
mkdir -p log/model_parallel
mkdir checkpoint
python mp_main.py --gpu "0,1" --dataset cifar10 --net_type resnet18 --batch_size 64 --epochs 20 > log/model_parallel/cifar10_resnet18.log 2>&1
python mp_main.py --gpu "0,1" --dataset mnist --net_type resnet18 --batch_size 64 --epochs 20 > log/model_parallel/mnist_resnet18.log 2>&1

python mp_main.py --gpu "0,1" --dataset cifar10 --net_type resnet32 --batch_size 64 --epochs 20 > log/model_parallel/cifar10_resnet32.log 2>&1
python mp_main.py --gpu "0,1" --dataset cifar10 --net_type convnet --batch_size 64 --epochs 20 > log/model_parallel/cifar10_convnet.log 2>&1
python mp_main.py --gpu "0,1" --dataset cifar10 --net_type resnet-ap --batch_size 64 --epochs 20 > log/model_parallel/cifar10_resnet-ap.log 2>&1
