import torch
import matplotlib.pyplot as plt
from einops import rearrange

class Image:
    def __init__(self, data, channels_first=True):
        assert len(data) == 3, 'Data has to be 3D to make an image'
        self.data = data
        self.channels_first = channels_first
        self.height, self.width = self._get_width_height()


    def _get_width_height(self):
        if self.channels_first:
            height = self.data.shape[1]
            width = self.data.shape[2]
        else:
            height = self.data.shape[0]
            width = self.data.shape[1]
        return height, width

    @classmethod
    def from_flat(cls, data, width, height):
        data = rearrange(data, '(h w) c -> c h w', w=width, h=height)
        image = cls(data, channels_first=True)
        return image

    def show(self, flip=False):
        data_show = self.flipud() if flip else self.data

        if self.channels_first:
            data_show = rearrange(data_show, 'c h w -> h w c')

        plt.imshow(data_show.cpu())

    def save(self, output_path, flip=False):
        data_save = self.flipud() if flip else self.data

        if self.channels_first:
            data_save = rearrange(data_save, 'c h w -> h w c')

        plt.imsave(output_path, data_save.cpu().numpy())

    def flipud(self):
        if self.channels_first:
            data = torch.flip(self.data, [1])
        else:
            data = torch.flip(self.data, [0])
        return data

    def fliplr(self):
        if self.channels_first:
            data = torch.flip(self.data, [2])
        else:
            data = torch.flip(self.data, [1])
        return data

