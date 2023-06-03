import torch

x=torch.randn(2,3,16,16)
H,W=16,16

mask = torch.zeros((x.shape[0], H * W), device=x.device)
mask[:, :H * W // 2] = 1
mask = mask.reshape(x.shape[0], 1, H, W).repeat(1, 1, 1, 1).flatten(2).transpose(1, 2)

mask=mask.reshape(x.shape[0], 1, H, W)
print(mask.shape)
print(mask)
# x = x.flatten(2).transpose(1, 2)
# x = self.norm1(x)
# x *= mask
# x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # ([1, 64, 56, 56])