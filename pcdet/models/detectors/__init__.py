from .detector3d_template import Detector3DTemplate
from .pg_rcnn import PGRCNN

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PGRCNN': PGRCNN,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
