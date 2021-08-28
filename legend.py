from tf_version import SSD300v2 as tf_ssd
from torch_version import SSD300v2 as torch_ssd

a = tf_ssd((300, 300, 3))
b = torch_ssd((300, 300, 3))