from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .wit_datamodule import WITDataModule
from .dagw_datamodule import DAGWDataModule
from .danhomes_datamodule import DanHomesDataModule
from .amhomes_datamodule import AmericanHomesDataModule
from .small_datamodule import SmallHomesDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "wit": WITDataModule,
    "dagw": DAGWDataModule,
    "danh": DanHomesDataModule,
    "amh": AmericanHomesDataModule,
    "smallh": SmallHomesDataModule
}
