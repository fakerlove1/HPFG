import numpy as np


class CutMixMaskGenerator(object):
    def __init__(self,
                 prop_range=(0.25, 0.5),
                 n_boxes=4,
                 random_aspect_ratio=True,
                 prop_by_area=True,
                 within_bounds=True,
                 invert=True):
        """
        prop_range: 
        n_boxes: number of boxes per image

        """
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks=4, mask_shape=(224, 224), rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold 选择应高于阈值的每个掩码的比例
            # shape=[n_masks,n_boxes] 随机生成 min=prop_range[0] max=prop_range[1] 之间大小的数字
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0  # shape=[n_masks,n_boxes]
            if self.random_aspect_ratio:  # 选择随机比例，从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))  # shape=[n_masks,n_boxes]
                x_props = mask_props / y_props  # shape=[n_masks,n_boxes]
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)  # 0.5
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])  # shape=[n_masks,n_boxes,2]

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))  # shape=[n_masks,n_boxes,2]
            rectangles = np.append(positions, positions + sizes, axis=2)  # shape=[n_masks,n_boxes,4]
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)  # shape=[n_masks,1,H,W]
        else:
            masks = np.ones((n_masks, 1) + mask_shape)  # shape=[n_masks,1,H,W]

        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks


if __name__ == "__main__":
    class config:
        cutmix_mask_prop_range = (0.25, 0.5)
        cutmix_boxmask_n_boxes = 4
        cutmix_boxmask_fixed_aspect_ratio = False
        cutmix_boxmask_by_size = False
        cutmix_boxmask_outside_bounds = False
        cutmix_boxmask_no_invert = False

    mask_generator = CutMixMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                         n_boxes=config.cutmix_boxmask_n_boxes,
                                         random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                         prop_by_area=not config.cutmix_boxmask_by_size,
                                         within_bounds=not config.cutmix_boxmask_outside_bounds,
                                         invert=not config.cutmix_boxmask_no_invert)

    print(mask_generator.generate_params(4, (224, 224)))
