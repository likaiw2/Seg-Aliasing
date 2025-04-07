from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class EarthVQADataset(CustomDataset):
    CLASSES = ('Background', 'Building', 'Road', 'Water', 'Barren',
               'Forest', 'Agricultural', 'Playground', 'Pond')

    PALETTE = [
        (255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255),
        (159, 129, 183), (0, 255, 0), (255, 195, 128),
        (165, 0, 165), (0, 185, 246)
    ]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)