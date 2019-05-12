from ssd.ssd_base import *
from ssd.ssd_extras import *
from ssd.ssd_prediction import *

# vgg16 network
base_network = vgg(base, i=3)
extra_layers = add_extras(extras, i=1024)

base_, extras_, head_ = multi_box(base_network, extra_layers, mbox, num_classes)