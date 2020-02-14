from argparse import ArgumentParser
from spacegraph_codebase.utils import *
from torch import optim
from spacegraph_codebase.train_helper import run_train, run_eval, run_eval_per_type, run_joint_train


def make_args_parser():
    parser = ArgumentParser()
    # dir
    parser.add_argument("--data_dir", type=str, default="./Place2Vec/")
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--num_context_sample", type=int, default=10,
        help='The number of context points we can sample, maximum is 10')

    # model
    parser.add_argument("--embed_dim", type=int, default=64,
        help='Point feature embedding dim')
    parser.add_argument("--dropout", type=float, default=0.5,
        help='The dropout rate used in all fully connected layer')

    # encoder
    parser.add_argument("--enc_agg", type=str, default="mean",
        help='the type of aggragation function for feature encoder')

    # model type
    parser.add_argument("--model_type", type=str, default="relative",
        help='''the type pf model we use, 
        relative: only relatve position; 
        global: only global position; 
        join: relative and global position
        together: use global position of center point in context prediction''')

    parser.add_argument("--num_rbf_anchor_pts", type=int, default=100,
        help='The number of RBF anchor points used in the "rbf" space encoder')
    parser.add_argument("--rbf_kernal_size", type=float, default=10e2,
        help='The RBF kernal size in the "rbf" space encoder')
    parser.add_argument("--rbf_kernal_size_ratio", type=float, default=0,
        help='The RBF kernal size ratio in the relative "rbf" space encoder')

    # space encoder
    parser.add_argument("--spa_enc", type=str, default="gridcell",
        help='the type of spatial relation encoder, none/naive/gridcell/hexagridcell/theory/theorydiag')
    parser.add_argument("--spa_embed_dim", type=int, default=64,
        help='Point Spatial relation embedding dim')
    parser.add_argument("--freq", type=int, default=16,
        help='The number of frequency used in the space encoder')
    parser.add_argument("--max_radius", type=float, default=10e4,
        help='The maximum spatial context radius in the space encoder')
    parser.add_argument("--min_radius", type=float, default=1.0,
        help='The minimum spatial context radius in the space encoder')
    parser.add_argument("--spa_f_act", type=str, default='sigmoid',
        help='The final activation function used by spatial relation encoder')
    parser.add_argument("--freq_init", type=str, default='geometric',
        help='The frequency list initialization method')
    parser.add_argument("--spa_enc_use_layn", type=str, default='F',
        help='whether to use layer normalzation in spa_enc')
    parser.add_argument("--spa_enc_use_postmat", type=str, default='T',
        help='whether to use post matrix in spa_enc')

            
    # global space/position encoder
    parser.add_argument("--g_spa_enc", type=str, default="gridcell",
        help='the type of spatial relation encoder, naive/gridcell/hexagridcell/theory/theorydiag/rbf')
    parser.add_argument("--g_spa_embed_dim", type=int, default=64,
        help='Point Spatial relation embedding dim')
    parser.add_argument("--g_freq", type=int, default=16,
        help='The number of frequency used in the space encoder')
    parser.add_argument("--g_max_radius", type=float, default=10e4,
        help='The maximum spatial context radius in the space encoder')
    parser.add_argument("--g_min_radius", type=float, default=1.0,
        help='The minimum spatial context radius in the space encoder')
    parser.add_argument("--g_spa_f_act", type=str, default='sigmoid',
        help='The final activation function used by spatial relation encoder')
    parser.add_argument("--g_freq_init", type=str, default='geometric',
        help='The frequency list initialization method')
    parser.add_argument("--g_spa_enc_use_layn", type=str, default='F',
        help='whether to use layer normalzation in g_spa_enc')
    parser.add_argument("--g_spa_enc_use_postmat", type=str, default='T',
        help='whether to use post matrix in g_spa_enc')

    # ffn
    parser.add_argument("--num_hidden_layer", type=int, default=3,
        help='The number of hidden layer in feedforward NN in the (global) space encoder')
    parser.add_argument("--hidden_dim", type=int, default=128,
        help='The hidden dimention in feedforward NN in the (global) space encoder')
    parser.add_argument("--use_layn", type=str, default="F",
        help='use layer normalization or not in feedforward NN in the (global) space encoder')
    parser.add_argument("--skip_connection", type=str, default="F",
        help='skip connection or not in feedforward NN in the (global) space encoder')


    parser.add_argument("--use_dec", type=str, default='T',
        help='whether to use another decoder following the initial decoder')


    # initial decoder, without query embedding
    parser.add_argument("--init_decoder_atten_type", type=str, default='concat',
        help='''the type of the intersection operator attention in initial decoder
        concat: the relative model
        g_pos_concat: the together model''')
    parser.add_argument("--init_decoder_atten_act", type=str, default='leakyrelu',
        help='the activation function of the intersection operator attention, see GAT paper Equ 3 in initial decoder')
    parser.add_argument("--init_decoder_atten_f_act", type=str, default='sigmoid',
        help='the final activation function of the intersection operator attention, see GAT paper Equ 6 in initial decoder')
    parser.add_argument("--init_decoder_atten_num", type=int, default=1,
        help='the number of the intersection operator attention in initial decoder')
    parser.add_argument("--init_decoder_use_layn", type=str, default='T',
        help='whether to use layer normalzation in initial decoder')
    parser.add_argument("--init_decoder_use_postmat", type=str, default='T',
        help='whether to use post matrix in initial decoder')


    # decoder 
    parser.add_argument("--decoder_atten_type", type=str, default='concat',
        help='''the type of the intersection operator attention
        concat: the relative model
        g_pos_concat: the together model''')
    parser.add_argument("--decoder_atten_act", type=str, default='leakyrelu',
        help='the activation function of the intersection operator attention, see GAT paper Equ 3')
    parser.add_argument("--decoder_atten_f_act", type=str, default='sigmoid',
        help='the final activation function of the intersection operator attention, see GAT paper Equ 6')
    parser.add_argument("--decoder_atten_num", type=int, default=0,
        help='the number of the intersection operator attention')
    parser.add_argument("--decoder_use_layn", type=str, default='T',
        help='whether to use layer normalzation')
    parser.add_argument("--decoder_use_postmat", type=str, default='T',
        help='whether to use post matrix')

    # encoder decoder
    parser.add_argument("--join_dec_type", type=str, default='max',
        help='the type of join_dec, min/max/mean/cat')
    parser.add_argument("--act", type=str, default='sigmoid',
        help='the activation function for the encoder decoder')

    # train
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.01,
        help='learning rate')
    parser.add_argument("--max_iter", type=int, default=50000000,
        help='the maximum iterator for model converge')
    parser.add_argument("--max_burn_in", type=int, default=5000,
        help='the maximum iterator for relative/global model converge')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tol", type=float, default=0.000001)


    # eval
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5000)


    # load old model
    parser.add_argument("--load_model", action='store_true')

    # cuda
    parser.add_argument("--cuda", action='store_true')

    return parser


def make_args_combine(args):
    args_spa_enc = "-{spa_enc:s}-{spa_embed_dim:d}-{freq:d}-{max_radius:.1f}-{min_radius:.1f}-{spa_f_act:s}-{freq_init:s}-{spa_enc_use_layn:s}-{spa_enc_use_postmat:s}".format(
            spa_enc=args.spa_enc,
            spa_embed_dim=args.spa_embed_dim,
            freq=args.freq,
            max_radius=args.max_radius,
            min_radius=args.min_radius,
            spa_f_act=args.spa_f_act,
            freq_init=args.freq_init,
            spa_enc_use_layn=args.spa_enc_use_layn,
            spa_enc_use_postmat=args.spa_enc_use_postmat)
    args_g_spa_enc = "-{g_spa_enc:s}-{g_spa_embed_dim:d}-{g_freq:d}-{g_max_radius:.1f}-{g_min_radius:.1f}-{g_spa_f_act:s}-{g_freq_init:s}-{g_spa_enc_use_layn:s}-{g_spa_enc_use_postmat:s}".format(
            g_spa_enc=args.g_spa_enc,
            g_spa_embed_dim=args.g_spa_embed_dim,
            g_freq=args.g_freq,
            g_max_radius=args.g_max_radius,
            g_min_radius=args.g_min_radius,
            g_spa_f_act=args.g_spa_f_act,
            g_freq_init=args.g_freq_init,
            g_spa_enc_use_layn=args.g_spa_enc_use_layn,
            g_spa_enc_use_postmat=args.g_spa_enc_use_postmat)
    if args.model_type == "relative":
        args_spa_enc_combined = args_spa_enc
    elif args.model_type == "global":
        args_spa_enc_combined = args_g_spa_enc
    else:
        args_spa_enc_combined = args_spa_enc + args_g_spa_enc

    args_ffn = "-{num_hidden_layer:d}-{hidden_dim:d}-{use_layn:s}-{skip_connection:s}".format(
            num_hidden_layer = args.num_hidden_layer, 
            hidden_dim = args.hidden_dim, 
            use_layn = args.use_layn, 
            skip_connection = args.skip_connection)

    args_init_decoder = "-{init_decoder_atten_type:s}-{init_decoder_atten_act:s}-{init_decoder_atten_f_act:s}-{init_decoder_atten_num:d}-{init_decoder_use_layn:s}-{init_decoder_use_postmat:s}".format(
            init_decoder_atten_type=args.init_decoder_atten_type,
            init_decoder_atten_act=args.init_decoder_atten_act,
            init_decoder_atten_f_act=args.init_decoder_atten_f_act,
            init_decoder_atten_num=args.init_decoder_atten_num,
            init_decoder_use_layn=args.init_decoder_use_layn,
            init_decoder_use_postmat=args.init_decoder_use_postmat)
    args_decoder = "-{decoder_atten_type:s}-{decoder_atten_act:s}-{decoder_atten_f_act:s}-{decoder_atten_num:d}-{decoder_use_layn:s}-{decoder_use_postmat:s}".format(
            decoder_atten_type=args.decoder_atten_type,
            decoder_atten_act=args.decoder_atten_act,
            decoder_atten_f_act=args.decoder_atten_f_act,
            decoder_atten_num=args.decoder_atten_num,
            decoder_use_layn=args.decoder_use_layn,
            decoder_use_postmat=args.decoder_use_postmat)

    args_combine = "/{data:s}-{num_context_sample:d}-{embed_dim:d}-{dropout:.1f}-{enc_agg:s}-{model_type:s}-{num_rbf_anchor_pts:d}-{rbf_kernal_size:.1f}-{rbf_kernal_size_ratio:.2f}{args_spa_enc_combined:s}-{args_ffn:s}-{use_dec:s}{args_init_decoder:s}{args_decoder:s}-{join_dec_type:s}-{act:s}-{opt:s}-{lr:.6f}-{batch_size:d}".format(
            data=args.data_dir.strip().split("/")[-2],
            num_context_sample=args.num_context_sample,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            enc_agg=args.enc_agg,
            model_type=args.model_type,

            num_rbf_anchor_pts = args.num_rbf_anchor_pts,
            rbf_kernal_size=args.rbf_kernal_size,
            rbf_kernal_size_ratio=args.rbf_kernal_size_ratio,

            args_spa_enc_combined=args_spa_enc_combined,

            args_ffn=args_ffn,

            use_dec=args.use_dec,

            args_init_decoder=args_init_decoder,

            args_decoder=args_decoder,

            join_dec_type = args.join_dec_type,
            act = args.act,
            opt=args.opt,
            lr=args.lr,
            batch_size=args.batch_size
            )

    return args_combine



class Trainer():
    """
    Trainer
    """
    def __init__(self, args, pointset, train_ng_list, val_ng_list, test_ng_list, feature_embedding, console = True):
        self.args = args
        self.pointset = pointset
        self.feature_embedding = feature_embedding
        
        self.train_ng_list = train_ng_list
        self.val_ng_list = val_ng_list
        self.test_ng_list = test_ng_list
 
        self.args_combine = make_args_combine(args) #+ ".L2"

        self.log_file = args.log_dir + self.args_combine + ".log"
        self.model_file = args.model_dir + self.args_combine + ".pth"

        self.logger = setup_logging(self.log_file, console = console, filemode='a')

        if args.model_type == "relative" and args.spa_enc == "random":
            self.enc_dec = None
            self.optimizer = None
            return
        elif args.model_type == "global" and args.g_spa_enc == "random":
            self.enc_dec = None
            self.optimizer = None
            return

        # make feature encoder
        self.enc = get_encoder(pointset.feature_embed_lookup, feature_embedding, pointset, args.enc_agg)

        if args.model_type == "relative" or args.model_type == "join" or args.model_type == "together":
            # make relative space encoder
            self.spa_enc = get_spa_encoder(
                args,
                model_type = args.model_type,
                spa_enc_type=args.spa_enc, 
                pointset = pointset,
                spa_embed_dim=args.spa_embed_dim, 
                coord_dim = 2, 
                num_rbf_anchor_pts = args.num_rbf_anchor_pts,
                rbf_kernal_size = args.rbf_kernal_size,
                frequency_num = args.freq, 
                max_radius = args.max_radius,
                min_radius = args.min_radius,
                f_act = args.spa_f_act,
                freq_init = args.freq_init,
                use_postmat=args.spa_enc_use_postmat)
#                                 dropout = args.dropout,
#                                 num_hidden_layer = args.num_hidden_layer, 
#                                 hidden_dim = args.hidden_dim, 
#                                 use_layn = args.use_layn, 
#                                 skip_connection = args.skip_connection)
        else:
            self.spa_enc = None

        if args.model_type == "global" or args.model_type == "join" or args.model_type == "together":
            # make global space encoder
            self.g_spa_enc = get_spa_encoder(
                args,
                model_type = args.model_type,
                spa_enc_type=args.g_spa_enc, 
                pointset = pointset,
                spa_embed_dim=args.g_spa_embed_dim, 
                coord_dim = 2, 
                num_rbf_anchor_pts = args.num_rbf_anchor_pts,
                rbf_kernal_size = args.rbf_kernal_size,
                frequency_num = args.g_freq, 
                max_radius = args.g_max_radius,
                min_radius = args.g_min_radius,
                f_act = args.g_spa_f_act,
                freq_init = args.g_freq_init,
                use_postmat=args.g_spa_enc_use_postmat)
#                                 dropout = args.dropout,
#                                 num_hidden_layer = args.num_hidden_layer, 
#                                 hidden_dim = args.hidden_dim, 
#                                 use_layn = args.use_layn, 
#                                 skip_connection = args.skip_connection)
        else:
            self.g_spa_enc = None

        # make decoder
        if args.model_type == "relative" or args.model_type == "join" or args.model_type == "together":

            # make query embedding initial decoder
            self.init_dec = get_context_decoder(dec_type=args.init_decoder_atten_type, 
                                query_dim=args.embed_dim, 
                                key_dim=args.embed_dim, 
                                spa_embed_dim=args.spa_embed_dim, 
                                g_spa_embed_dim=args.g_spa_embed_dim,
                                have_query_embed = False, 
                                num_attn = args.init_decoder_atten_num, 
                                activation = args.init_decoder_atten_act, 
                                f_activation = args.init_decoder_atten_f_act, 
                                layn = args.init_decoder_use_layn, 
                                use_postmat = args.init_decoder_use_postmat,
                                dropout = args.dropout)

            if args.use_dec == "T":
                # make decoder
                self.dec = get_context_decoder(dec_type=args.decoder_atten_type, 
                                    query_dim=args.embed_dim, 
                                    key_dim=args.embed_dim, 
                                    spa_embed_dim=args.spa_embed_dim, 
                                    g_spa_embed_dim=args.g_spa_embed_dim,
                                    have_query_embed = True, 
                                    num_attn = args.decoder_atten_num, 
                                    activation = args.decoder_atten_act, 
                                    f_activation = args.decoder_atten_f_act, 
                                    layn = args.decoder_use_layn, 
                                    use_postmat = args.decoder_use_postmat,
                                    dropout = args.dropout)
            else:
                self.dec = None

            if args.model_type == "join":
                self.joint_dec = JointRelativeGlobalDecoder(feature_embed_dim = args.embed_dim, 
                                f_act = args.act, 
                                dropout = args.dropout,
                                join_type = args.join_dec_type)
            else:
                self.joint_dec = None

        else:
            self.init_dec = None
            self.dec = None
            self.joint_dec = None

        if args.model_type == "global" or args.model_type == "join":
            # make global space decoder
            self.g_spa_dec = DirectPositionEmbeddingDecoder(g_spa_embed_dim=args.g_spa_embed_dim, 
                                feature_embed_dim=args.embed_dim, 
                                f_act = args.act, 
                                dropout = args.dropout)
        else:
            self.g_spa_dec = None


        # if args.model_type == "global" or args.model_type == "relative":
        # make encoder encoder
        self.enc_dec = get_enc_dec(model_type=args.model_type, 
                            pointset=pointset, 
                            enc = self.enc, 
                            spa_enc = self.spa_enc, 
                            g_spa_enc = self.g_spa_enc, 
                            g_spa_dec = self.g_spa_dec, 
                            init_dec=self.init_dec, 
                            dec=self.dec, 
                            joint_dec=self.joint_dec, 
                            activation = args.act, 
                            num_context_sample = args.num_context_sample, 
                            num_neg_resample = 10)

        if args.cuda:
            self.enc_dec.cuda()

        if args.opt == "sgd":
            self.optimizer = optim.SGD(filter(lambda p : p.requires_grad, self.enc_dec.parameters()), lr=args.lr, momentum=0)
        elif args.opt == "adam":
            self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.enc_dec.parameters()), lr=args.lr)

        print("create model from {}".format(self.args_combine + ".pth"))
        self.logger.info("Save file at {}".format(self.args_combine + ".pth"))

        
    def load_model(self):
        self.logger.info("Load model from {}".format(self.model_file))
        self.enc_dec.load_state_dict(torch.load(self.model_file))

    def eval_model(self, flag="TEST"):
        if flag == "TEST":
            mrr_, hit1_, hit5_, hit10_ = run_eval(self.enc_dec, self.test_ng_list, 0, self.logger, do_full_eval = False)
            self.logger.info("Test MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))

            mrr_, hit1_, hit5_, hit10_ = run_eval(self.enc_dec, self.test_ng_list, 0, self.logger, do_full_eval = True)
            self.logger.info("Test MRR: {:f}, 100 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))
        elif flag == "VALID":
            mrr_, hit1_, hit5_, hit10_ = run_eval(self.enc_dec, self.val_ng_list, 0, self.logger, do_full_eval = False)
            self.logger.info("Valid MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))

            mrr_, hit1_, hit5_, hit10_ = run_eval(self.enc_dec, self.val_ng_list, 0, self.logger, do_full_eval = True)
            self.logger.info("Valid MRR: {:f}, 100 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))

    def eval_model_per_type(self, typeid2root = None, flag="TEST"):
        if flag == "TEST":
            type2mrr, type2hit1, type2hit5, type2hit10 = run_eval_per_type(self.enc_dec, self.pointset, self.test_ng_list, 0, self.logger, typeid2root = typeid2root, do_full_eval = True)
        elif flag == "VALID":
            type2mrr, type2hit1, type2hit5, type2hit10 = run_eval_per_type(self.enc_dec, self.pointset, self.val_ng_list, 0, self.logger, typeid2root = typeid2root, do_full_eval = True)

        return type2mrr, type2hit1, type2hit5, type2hit10

    def train(self):
        run_train(model=self.enc_dec, 
                    optimizer=self.optimizer, 
                    train_ng_list=self.train_ng_list, 
                    val_ng_list=self.val_ng_list, 
                    test_ng_list=self.test_ng_list, 
                    logger=self.logger,
                    max_iter=self.args.max_iter, 
                    batch_size=self.args.batch_size, 
                    log_every=self.args.log_every, 
                    val_every=self.args.val_every, 
                    tol=self.args.tol,
                    model_file=self.model_file)
        if self.enc_dec is not None:
            torch.save(self.enc_dec.state_dict(), self.model_file)