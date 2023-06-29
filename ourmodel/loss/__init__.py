from ourmodel.loss.adv_loss import adv
from ourmodel.loss.coral import CORAL
from ourmodel.loss.cos import cosine
from ourmodel.loss.kl_js import kl_div, js
from ourmodel.loss.mmd import MMD_loss
from ourmodel.loss.mutual_info import Mine
from ourmodel.loss.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]