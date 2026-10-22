import torch

def main():
    a = torch.tensor([1,2])
    print(a)
    print('CUDA is available : ', torch.cuda.is_available())


if __name__ == "__main__":
    main()