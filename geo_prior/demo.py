"""
Demo that either 1) takes a location as input and returns a prediction indicating
the likelihood that each category is present there, or 2) takes a category ID as
input and generates a prediction for each location on the globe.
"""
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import os
from six.moves import urllib

from geo_prior import models
from geo_prior import utils
from geo_prior import grid_predictor as grid


def download_model(model_url, model_path):

    # Download pre-trained model if it is not currently available.
    if not os.path.isfile(model_path):
        try:
            print('Downloading model from: ' + model_url)
            urllib.request.urlretrieve(model_url, model_path)
        except:
            print('Failed to download model from: ' + model_url)


def main(args):

    download_model(args.model_url, args.model_path)
    print('Loading model: ' + args.model_path)
    net_params = torch.load(args.model_path, map_location='cpu')
    params = net_params['params']
    model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                         num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])
    model.load_state_dict(net_params['state_dict'])
    model.eval()

    # load class names
    with open(args.class_names_path) as da:
        class_data = json.load(da)

    if args.demo_type == 'location':
        # convert coords to torch
        coords = np.array([args.longitude, args.latitude])[np.newaxis, ...]
        obs_coords = utils.convert_loc_to_tensor(coords, params['device'])
        obs_time = torch.ones(coords.shape[0], device=params['device'])*args.time_of_year*2 - 1.0
        loc_time_feats = utils.encode_loc_time(obs_coords, obs_time, concat_dim=1, params=params)

        print('Making prediction ...')
        with torch.no_grad():
            pred = model(loc_time_feats)[0, :]
        pred = pred.cpu().numpy()

        num_categories = 25
        print('\nTop {} likely categories for location {:.4f}, {:.4f}:'.format(num_categories, coords[0,0], coords[0,1]))
        most_likely = np.argsort(pred)[::-1]
        for ii, cls_id in enumerate(most_likely[:num_categories]):
            print('{}\t{}\t{:.3f}'.format(ii, cls_id, np.round(pred[cls_id], 3)) + \
                '\t' + class_data[cls_id]['our_name'] + ' - ' + class_data[cls_id]['preferred_common_name'])

    elif args.demo_type == 'map':
        # grid predictor - for making dense predictions for each lon/lat location
        gp = grid.GridPredictor(np.load('data/ocean_mask.npy'), params, mask_only_pred=True)

        if args.class_of_interest == -1:
            args.class_of_interest = np.random.randint(len(class_data))
        print('Selected category: ' + class_data[args.class_of_interest]['our_name'] +\
            ' - ' + class_data[args.class_of_interest]['preferred_common_name'])

        print('Making prediction ...')
        grid_pred = gp.dense_prediction(model, args.class_of_interest, time_step=args.time_of_year)

        op_file_name = class_data[args.class_of_interest]['our_name'].lower().replace(' ', '_') + '.png'
        print('Saving prediction to: ' + op_file_name)
        plt.imsave(op_file_name, 1.0-grid_pred, cmap='afmhot', vmin=0, vmax=1)


if __name__ == "__main__":

    info_str = '\nPresence-Only Geographical Priors for Fine-Grained Image Classification.\n\n' + \
               'This demo can be run in one of two ways:\n' + \
               '1) Give a location and get a list of most likely classes there e.g\n' + \
               '   python demo.py location --longitude -118.1445155 --latitude 34.1477849 --time_of_year 0.5\n' + \
               'Input coordinates should be in decimal degrees i.e. ' + \
               'Longitude: [-180, 180], Latitude: [-90, 90], and Time of year [0, 1].\n\n' + \
               '2) Give a category ID as input and get a prediction for each location on the globe for that category e.g.\n' + \
               '   python demo.py map --class_of_interest 3731\n' + \
               'If class_of_interest is not specified a random one will be selected.\n\n'

    model_path = 'models/model_inat_2018_full_final.pth.tar'
    model_url  = 'http://www.vision.caltech.edu/~macaodha/projects/geopriors/model_inat_2018_full_final.pth.tar'
    class_names_path = 'web_app/data/categories2018_detailed.json'

    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('demo_type', type=str, help='Can be either "map" or "location".')
    parser.add_argument('--model_path', type=str, default=model_path,
        help='Path to trained model.')
    parser.add_argument('--model_url', type=str, default=model_url,
        help='Path to remote trained model.')
    parser.add_argument('--class_names_path', type=str, default=class_names_path,
        help='Path to class names.')
    parser.add_argument('--longitude', type=float, default=-118.1445155,
        help='Longitude of interest.')
    parser.add_argument('--latitude', type=float, default=34.1477849,
        help='Latitude of interest.')
    parser.add_argument('--time_of_year', type=float, default=0.5,
        help='Time of year [0, 1].')
    parser.add_argument('--class_of_interest', type=int, default=-1,
        help='Class of interest [0, num_classes-1].')
    args = parser.parse_args()

    main(args)

