from __future__ import print_function
from flask import Flask, render_template, request, make_response, jsonify
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import json
import torch
import config

import sys
sys.path.append('../')
import geo_prior.grid_predictor as grid
import geo_prior.models as models


application = Flask(__name__)
application.secret_key = config.SECRET_KEY


# load class names
with open(config.CLASS_META_FILE) as da:
    class_meta_data = json.load(da)
class_names = [cc['our_name'] for cc in class_meta_data]
class_names_joint = [cc['our_name'] + ' - ' + cc['preferred_common_name'] for cc in class_meta_data]
default_class_index = class_names.index(config.DEFAULT_CLASS)

# load background mask
mask = np.load(config.MASK_PATH).astype(np.int)
mask_lines = (np.gradient(mask)[0]**2 + np.gradient(mask)[1]**2)
mask_lines[mask_lines > 0.0] = 1.0
mask_lines = 1.0 - mask_lines
mask = mask.astype(np.uint8)

# create placeholder image that will be displayed
blank_im = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for cc in range(3):
    blank_im[:,:,cc] = (255*mask_lines).astype(np.uint8)

# load model
net_params = torch.load(config.MODEL_PATH, map_location='cpu')
params = net_params['params']
params['device'] = 'cpu'
model = models.FCNet(params['num_feats'], params['num_classes'], params['num_filts'], params['num_users']).to(params['device'])
model.load_state_dict(net_params['state_dict'])
model.eval()

# grid predictor - for making dense predictions for each lon/lat location
gp = grid.GridPredictor(mask, params, mask_only_pred=True)

# generate features
print('generating location features')
feats = []
for time_step in np.linspace(0,1,config.NUM_TIME_STEPS+1)[:-1]:
    feats.append(gp.dense_prediction_masked_feats(model, time_step))
print('location features generated')

def create_images(index_of_interest):
    images = []
    for tt, time_step in enumerate(np.linspace(0,1,config.NUM_TIME_STEPS+1)[:-1]):

        with torch.no_grad():
            pred = model.eval_single_class(feats[tt], index_of_interest)
            pred = torch.sigmoid(pred).data.cpu().numpy()

        # copy the prediction into an image for display
        im = blank_im.copy()
        im[:,:,0] = (255*(np.clip(mask_lines-gp.create_full_output(pred), 0, 1))).astype(np.uint8)

        # encode the image
        im_output = BytesIO()
        Image.fromarray(im).save(im_output, 'PNG')
        #Image.fromarray(im).save(im_output, "JPEG", quality=config.JPEG_QUALITY)
        im_data = base64.b64encode(im_output.getvalue()).decode('utf-8')
        im_output.close()
        images.append(im_data)

    return images


@application.route('/')
@application.route('/index')
@application.route('/index', methods=['POST'])
def index():

    if request.method == 'POST':
        if request.form['submit_btn'] == 'random':
            index_of_interest = np.random.randint(len(class_names))
        else:
            index_of_interest = int(request.form['class_of_interest'])
    else:
        index_of_interest = default_class_index

    class_data = class_meta_data[index_of_interest]

    print(class_data['our_id'], class_data['our_name'])

    # generate distribution maps for the species of interest
    images = create_images(index_of_interest)
    print('images created\n')

    return render_template('index.html', images=images, time_steps_txt=config.MONTHS,
                           class_data=class_data, im_height=mask.shape[0])


@application.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query').lower()
    print(query)
    results = [(cc, class_names_joint[cc]) for cc in range(len(class_names_joint)) if query in class_names_joint[cc].lower()]
    num_return = min(len(results), config.MAX_NUM_QUERIES_RETURNED)

    matching_classes = []
    for cc in range(num_return):
        matching_classes.append({'label':results[cc][1], 'idx':results[cc][0]})

    return jsonify(matching_classes=matching_classes)


if __name__ == '__main__':
    application.run(host='127.0.0.1', port=8000)
    #application.run()

