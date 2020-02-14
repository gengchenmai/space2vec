
"""
Configuration parameters
"""

SECRET_KEY = 'PUT_A_SECRET_KEY_HERE'
JPEG_QUALITY = 70
DEFAULT_CLASS = 'Hylocichla mustelina'
NUM_TIME_STEPS = 12
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MASK_PATH = 'data/masks/50000/mask_50000.npy'
# To speed up inference can use less time steps and lower spatial resolution
#NUM_TIME_STEPS = 6
#MONTHS = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov']
#MASK_PATH = 'data/masks/60000/mask_60000.npy'

# NOTE If using a different model than one trained on iNat2018 CLASS_META_FILE will need to be changed
CLASS_META_FILE = 'data/categories2018_detailed.json'
MODEL_PATH = '../models/model_inat_2018_full_final.pth.tar'

MAX_NUM_QUERIES_RETURNED = 20  # for autocomplete only return this many partial matches
