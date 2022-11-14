'''
Developed by Gengchen Mai

gengchen.mai@gmail.com
05/08/2019
'''
from argparse import ArgumentParser

from spacegraph_codebase.utils import *
from spacegraph_codebase.Place2Vec.cur_data_utils import load_pointset
from spacegraph_codebase.data_utils import load_ng
from spacegraph_codebase.model import NeighGraphEncoderDecoder
from spacegraph_codebase.train_helper import run_train, run_eval, run_joint_train
from spacegraph_codebase.trainer import *

from torch import optim
import numpy as np

# define arguments
parser = make_args_parser()
args = parser.parse_args()

# load dataset
print("Loading NeighGraph data..")

print("Loading training NeighGraph data..")
train_ng_list = load_ng(args.data_dir + "/neighborgraphs_training.pkl")
print("Loading validation NeighGraph  data..")
val_ng_list = load_ng(args.data_dir + "/neighborgraphs_validation.pkl")
print("Loading testing NeighGraph data..")
test_ng_list = load_ng(args.data_dir + "/neighborgraphs_test.pkl")


print("Loading PointSet data..")
pointset, feature_embedding = load_pointset(data_dir=args.data_dir, 
                                                embed_dim=args.embed_dim,
                                                do_feature_sampling = False)
if args.cuda:
    pointset.feature_embed_lookup = cudify(feature_embedding)

# make model directory
os.makedirs(args.model_dir, exist_ok=True)

# build NN model
trainer = Trainer(args, pointset, train_ng_list, val_ng_list, test_ng_list, feature_embedding, console = True)

trainer.logger.info("All arguments:")
for arg in vars(args):
    trainer.logger.info("{}: {}".format(arg, getattr(args, arg)))

# load model
if args.load_model:
    trainer.logger.info("LOADING MODEL")
    trainer.load_model()

# Save parameters
config = vars(args)
with open(os.path.join(args.model_dir, "config.json"), "w") as outfile:
    json.dump(config, outfile)

# train NN model
trainer.train()
# trainer.eval_model()