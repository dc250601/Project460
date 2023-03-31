from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import PIL

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class GaussianNoise(object):
    def __init__(self, p=0.5, amplitude = 0.5):
        self.p = p
        self.a = amplitude
    def __call__(self, img):
        sigma = 0
        img = np.array(img)
        if np.random.uniform(0,1)<self.p:
            sigma = np.random.uniform(-self.a,self.a)
        noise = np.random.randn(img.shape[0],img.shape[1],img.shape[2])
        noise = (noise  - noise.min())/(noise.max()-noise.min())
        img = img+sigma*noise
        img = 255*(img - img.min())/(img.max() - img.min())
        img = img.astype(np.uint8)
        return PIL.Image.fromarray(img)




class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    107, interpolation=InterpolationMode.BICUBIC,
                    scale=(0.4, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90,90)),
                GaussianNoise(p=1,amplitude=0.5),
                GaussianBlur(p=1.0),
                transforms.ToTensor(),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    107, interpolation=InterpolationMode.BICUBIC,
                    scale=(0.4, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90,90)),
                GaussianNoise(p=1, amplitude=0.5),
                GaussianBlur(p=1.0),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
