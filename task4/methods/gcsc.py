from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    """
    Good Closed-Set Classifier (Vaze et al., 2022).

    Deliberately inherits compute_loss unchanged: GCSC's objective IS vanilla
    ten-way cross-entropy. The only difference in the entire pipeline is
    RandAugment(2, 9) in the train transform, applied in img_prep/cifar_data.py.

    The emptiness of this class is the point -- it is the structural evidence
    that no training signal was added, so any change in rejection comes from a
    better-fit representation rather than from a new objective.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.get('randaugment') is True, \
            'GCSC requires randaugment=True; otherwise this is just Vanilla again'