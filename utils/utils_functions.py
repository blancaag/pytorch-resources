import libraries
from libraries import *

## General utils

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def silence():
    back = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return back
def speak(back): sys.stdout = back

def resize(im_arr, new_size=(150, 150)):
    from PIL import Image
    im = Image.fromarray(np.uint8(im_arr))
    im = im.resize(new_size, Image.NEAREST)
    return np.array(im)

def rescale_normalize(im, mean=0.5, std=0.5):
    im = im / 255. # rescale values to [0, 1]
    im = (im - mean) / std
    return im

## Modelling utils

def train_and_validate(net, loss_func, optimizer, metrics, epochs, loaders, cuda=False):

    train_loader, val_loader = loaders[0], loaders[-1]

    log = []

    for i in range(epochs): # in each epoch:

        print('Epoch %d..' % i)
        epoch_log = OrderedDict()

        ## training
        prefix = 'train_'
        batch_losses = []
        running_loss = 0.00
        for _, data in enumerate(train_loader): # in each batch:

            X, y = Variable(data[0]), Variable(data[1])
            if cuda: X, y = X.cuda(), y.cuda()

            # forward + backward + optimize
            optimizer.zero_grad()
            pred = net(X)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            # logging
            batch_losses.append(loss.data[0])
            running_loss += loss.data[0]
            epoch_log[prefix + 'loss'] = running_loss / (_ + 1)

        if metrics: # compute metrics per batch/alternatively: per poepoch: how if I dont have all the data in memory?
            for metric in metrics: # e.g.: 'accuracy' ONLY FOR LAST BATCH?
                epoch_log[prefix + metric.__name__] = metric(pred, y)

        epoch_log[prefix + 'loss_hist'] = batch_losses

        ## validation
        prefix = 'val_'
        batch_losses = []
        running_loss = 0.00
        for _, data in enumerate(val_loader): # in each batch:

            X, y = Variable(data[0]), Variable(data[1])
            if cuda: X, y = X.cuda(), y.cuda()

            # forward
            pred = net(X)
            loss = loss_func(pred, y)

            # logging
            batch_losses.append(loss.data[0])
            running_loss += loss.data[0]
            epoch_log[prefix + 'loss'] = running_loss / (_ + 1)

        if metrics: # compute metrics per batch/alternatively: per poepoch: how if I dont have all the data in memory?
            for metric in metrics: # e.g.: 'accuracy' ONLY FOR LAST BATCH?
                epoch_log[prefix + metric.__name__] = metric(pred, y)

        epoch_log[prefix + 'loss_hist'] = batch_losses

    print("Train loss: ", log['train_loss'], "Val loss: ", log['val_loss'])

    log.append(epoch_log)


def plot_log(log, metric):
    """
    Plots the metric history recorded during training and validation -function training_and_validation()

    Options:
    'loss': log of per epoch losses
    'loss_hist': log of per batch losses
    any metric specifified in metrics argument in training_and_validation() -e.g.: 'accuracy'
    E.g.:  plot_log(log, 'loss')
    """

    fig = plt.figure(figsize=(10, 3))

    [log[i]['train_loss'] for i in range(len(log))]

    plt.plot([log[i]['train_' + metric] for i in range(len(log))])
    plt.plot([log[i]['val_' + metric] for i in range(len(log))])

    plt.ylabel(metric)
    plt.xlabel('epoch/batch')
    plt.legend(['train_' + metric, 'val_' + metric], loc='upper right')
#     plt.title('log', size=12)
    plt.tight_layout()


def evaluate_model(model, dataloader, metrics=[categorical_accuracy]):
    """
    Evaluate the mdel over a set of provided metris.
    E.g.: evaluate_model(net, valloader)
    """
    log = {}

    running_sum = {}
    for i in metrics: running_sum[i.__name__] = 0.00

    for _, data in enumerate(dataloader):

        X, y = Variable(data[0]), Variable(data[1])
        pred = model(X)

        for i in metrics:
            running_sum[i.__name__] += i(pred, y)
            log[i.__name__] = running_sum[i.__name__] / (_ + 1)

    print('Network\'s evaluation metrics on %d test images: ' % _)
    [print(' - %s: %f%%' % (i.__name__, log[i.__name__])) for i in metrics]


def take_sample(path, target_size=(512, 512), pil=False):
    """
    Take and plot image form input folder
    Expects path to images..
    """
    ims_gen = (join(path, i) for i in listdir(path) if isfile(join(path, i)))
    for _, i in enumerate(ims_gen):
        if _ == 0:
            if pil: im = img_to_array(load_img(i, target_size=target_size))
            else:
                im = cv2.imread(i).astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, target_size)
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(im.astype(np.uint8))
                return im

def plot(im):
    """
    Expects numpy array
    """
    if im.ndim == 4: im = im[0]
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(im.astype(np.uint8))

def plot_ims(data, ncols=3, beautiful=False):
    """
    Expects numpy array
    """
    if data.shape[0] % ncols != 0:
        raise ValueError('Expecting data to contain a number of images multiple of 3 but got an array with shape:', data.shape[0])

    ncols = ncols
    nrows = data.shape[0] // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(12, nrows*5)
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data[i].astype(np.uint8))
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

def plot_images(data): # plot_images(12)
    """
    Expects numpy array
    """
    nrows = 2
    ncols = data.shape[0] // nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(12, 5)
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data[i].astype(np.uint8))
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

## Data loading functions:

def load_data(output_folder, split=0.8, path='../data/sushi_or_sandwich/', target_size=(256, 256), pil=False):

    cd_output = join(output_folder, 'compressed_data')
    if not os.path.exists(cd_output): os.mkdir(cd_output)

    cd_folder = join(cd_output, str(target_size[0]) + '_' + str(target_size[1]))

    if os.path.exists(cd_folder):
        print('Loading compressed data..')

        x_train = load_array(join(cd_folder, 'x_train'))
        y_train = load_array(join(cd_folder, 'y_train'))
        x_test = load_array(join(cd_folder, 'x_test'))
        y_test = load_array(join(cd_folder, 'y_test'))

        return x_train, y_train, x_test, y_test, None, None

    print('Reading original data..')

    flabs = (join(path, i) for i in listdir(path) if not isfile(join(path, i)))

    d_lab = {'sandwich': 0, 'sushi': 1}

    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in flabs:
        ims_gen = (join(i, j) for j in listdir(i) if isfile(join(i, j)))

        for j in ims_gen:
            if pil: im = img_to_array(load_img(j, target_size=target_size))
            else:
                im = cv2.imread(j).astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # b,g,r = cv2.split(im); im = cv2.merge([r,g,b])
                im = cv2.resize(im, target_size)

            ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
            labs = np.append(labs, (d_lab[i.split(path)[1]], ), axis=0)

    return preprocess_data(ims, labs, cd_folder, split)

def load_data_extended(output_folder, split=0.8, path='../data/sushi_or_sandwich/', target_size=(256, 256), pil=False):

    cd_output = join(output_folder, 'compressed_data_extended')
    if not os.path.exists(cd_output): os.mkdir(cd_output)

    cd_folder = join(cd_output, str(target_size[0]) + '_' + str(target_size[1]))

    if os.path.exists(cd_folder):
        print('Loading compressed data..')

        x_train = load_array(join(cd_folder, 'x_train'))
        y_train = load_array(join(cd_folder, 'y_train'))
        x_test = load_array(join(cd_folder, 'x_test'))
        y_test = load_array(join(cd_folder, 'y_test'))

        return x_train, y_train, x_test, y_test, None, None

    print('Reading original data..')

    fsplit = [join(path, i) for i in listdir(path) if not isfile(join(path, i))]

    d_lab = {'sandwich': 0, 'sushi': 1}
    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in fsplit:
        flabs = [join(i, j) for j in listdir(i) if not isfile(join(i, j))]

        for j in flabs:
            ims_gen = (join(j, k) for k in listdir(j) if isfile(join(j, k)))

            for k in ims_gen:
                if pil: im = img_to_array(load_img(k, target_size=target_size))
                else:
                    im = cv2.imread(k).astype(np.float32)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, target_size)

                ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
                labs = np.append(labs, (d_lab[j.split(i)[1].split('/')[1]], ), axis=0)

    return preprocess_data(ims, labs, cd_folder, split)

def preprocess_data(ims, labs, cd_folder, split=0.8):

    # 1. concatenation and shuffling

    x = ims
    y = labs

    shuffle_ix = np.arange(x.shape[0])
    rgen = np.random.RandomState()
    rgen.seed(0)
    rgen.shuffle(shuffle_ix)

    x = x[shuffle_ix,:,:,:]
    y = y[shuffle_ix]

    mean_set = x.mean().astype(np.float32)
    std_set = x.std().astype(np.float32)

    # 2. setting training and validation sets

    split_ix = x.shape[0] * int(split*10) // 10

    x_train = x[:split_ix,:,:,:]
    y_train = y[:split_ix]

    x_test = x[split_ix:,:,:,:]
    y_test = y[split_ix:]

    if not os.path.exists(cd_folder):
        os.mkdir(cd_folder)

        save_array(join(cd_folder, 'x_train'), x_train)
        save_array(join(cd_folder, 'y_train'), y_train)
        save_array(join(cd_folder, 'x_test'), x_test)
        save_array(join(cd_folder, 'y_test'), y_test)

    return x_train, y_train, x_test, y_test, mean_set, std_set

def load_test_data(split=0.8, path='../source/test_data/', target_size=(256, 256), pil=False):

    cd_output = join('../source/', 'compressed_data')
    if not os.path.exists(cd_output): os.mkdir(cd_output)

    cd_folder = join(cd_output, str(target_size[0]) + '_' + str(target_size[1]))

    if os.path.exists(cd_folder):
        print('Loading compressed data..')

        x_test = load_array(join(cd_folder, 'x_test'))
        y_test = load_array(join(cd_folder, 'y_test'))

        return x_test, y_test

    print('Reading original images..')

    flabs = (join(path, i) for i in listdir(path) if not isfile(join(path, i)))

    d_lab = {'sandwich': 0, 'sushi': 1}

    ims = np.empty(((0,) + target_size + (3,)))
    labs = np.empty((0,))

    for i in flabs:
        ims_gen = (join(i, j) for j in listdir(i) if isfile(join(i, j)))

        for j in ims_gen:
            if pil: im = img_to_array(load_img(j, target_size=target_size))
            else:
                im = cv2.imread(j).astype(np.float32)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # b,g,r = cv2.split(im); im = cv2.merge([r,g,b])
                im = cv2.resize(im, target_size)

            ims = np.append(ims, im.reshape((1,) + im.shape), axis=0)
            labs = np.append(labs, (d_lab[i.split(path)[1]], ), axis=0)

    if not os.path.exists(cd_folder):
        os.mkdir(cd_folder)

        save_array(join(cd_folder, 'x_test'), ims)
        save_array(join(cd_folder, 'y_test'), labs)

    return ims, labs



## Modelling utils

def evaluate(model, x_test, y_test):
    metrics = model.evaluate(x_test, y_test, verbose=0)
    for i,j in zip(model.metrics_names, range(len(metrics))):
            print("Test %s: %.2f%%" % (i, metrics[j] * 100))

## Data augmentation utils:

# cropping functions

def centered_crop(x, crop_size = (128, 128), **kwargs):
    center_w, center_h = x.shape[0] // 2, x.shape[1] // 2
    half_w, half_h = crop_size[0] // 2, crop_size[1] // 2
    x = x[center_w - half_w:center_w + half_w, center_h - half_h:center_h + half_h, :]
    x = cv2.resize(x, (256, 256))
    return x

def random_crop(x, crop_size = (128, 128), seed=None, **kwargs):
    np.random.seed(seed)
    w, h = x.shape[0], x.shape[1]
    range_w = (w - crop_size[0]) #// 2
    range_h = (h - crop_size[1]) #// 2
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    x = x[offset_w:offset_w + crop_size[0], offset_h:offset_h + crop_size[1], :]
    x = cv2.resize(x, (256, 256))
    return x

# custom preprocessing function

def set_preprocess_input_function(model, cropping=None):

    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        model: pretrained model to be used
        crop: ['centered', 'random', None].
    # Returns
        Preprocessed tensor.
    """

    if model in ['inception_v3', 'xception', 'mobilenet']:
        def preprocess_input(x):
            x /= 255.
            x -= 0.5
            x *= 2.

            if cropping=='centered': centered_crop(x)
            if cropping=='random': random_crop(x)
            return x

    elif model in ['densenet_121', 'densenet_161', 'densenet_169']:
        def preprocess_input(x):
            # Subtract mean pixel and multiple by scaling constant
            # Reference: https://github.com/shicai/DenseNet-Caffe
            x[:,:,0] = (x[:,:,0] - 103.94) * 0.017
            x[:,:,1] = (x[:,:,1] - 116.78) * 0.017
            x[:,:,2] = (x[:,:,2] - 123.68) * 0.017

            if cropping=='centered': centered_crop(x)
            if cropping=='random': random_crop(x)
            return x

    elif model in ['resnet_50', 'vgg_16', 'vgg_19']:
        def preprocess_input(x):
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

            if cropping=='centered': x = centered_crop(x)
            if cropping=='random': x = random_crop(x)
            return x

    return preprocess_input

def set_reverse_preprocess_input_function(model):

    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        model: pretrained model to be used
        crop: ['centered', 'random', None].
    # Returns
        Preprocessed tensor.
    """

    if model in ['mobilenet']:
        def reverse_preprocess_input(x):
            x /= 2.
            x += 0.5
            x *= 255.
            return x

    elif model in ['inception_v3', 'xception']:
        def reverse_preprocess_input(x):
            x /= 2.
            x += 0.5
            x *= 255.
            return x

    elif model in ['densenet_121', 'densenet_161', 'densenet_169']:
        def reverse_preprocess_input(x):
            # subtract mean pixel and multiple by scaling constant
            # reference: https://github.com/shicai/DenseNet-Caffe
            x[:,:,0] = (x[:,:,0] + 103.94) / 0.017
            x[:,:,1] = (x[:,:,1] + 116.78) / 0.017
            x[:,:,2] = (x[:,:,2] + 123.68) / 0.017
            return x

    elif model in ['resnet_50', 'vgg_16', 'vgg_19']:
        def reverse_preprocess_input(x):
            # zero-center by Imagenet channel mean
            x[..., 0] += 103.939
            x[..., 1] += 116.779
            x[..., 2] += 123.68
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            return x

    return reverse_preprocess_input

# defining the augmented data generators

def aug_data_generators(model, n_gen=None, cropping=None):

    preprocess_input = set_preprocess_input_function(model=model, cropping=cropping)

    fill_mode = ['constant', 'nearest', 'reflect', 'wrap']

    ad_gens = []
    count = 0
    for i in range(0, 180, 60):
        for j in range(0, 8, 3):
            for k in range(0, 10, 3):
                for l in range(0, 1, 1): #50, 10):
                    for m in [True, False]:
                        for n in fill_mode:
                            count += 1
                            exec("""ad_gen_{} = ImageDataGenerator(
                                    rotation_range={},              # i
                                    width_shift_range={}/10,        # j
                                    height_shift_range={}/10,       # j
                                    shear_range={}/10,              # j
                                    zoom_range={}/10,               # k
                                    channel_shift_range={},         # l
                                    horizontal_flip={},             # m
                                    vertical_flip=not {},           # m
                                    fill_mode={},                   # n
                                    cval=0,
                                    preprocessing_function=preprocess_input)
                            """.format(count, i, j, j, j, k, l, m, m, 'n'))
                            exec('ad_gens.append(ad_gen_{})'.format(count))
    print("Total number of available aug. data generators: ", len(ad_gens))

    if n_gen: shuffle_ix = list(np.random.randint(0, len(ad_gens), n_gen))
    else: shuffle_ix = list(np.random.randint(0, len(ad_gens), len(ad_gens)))

    ad_gens_g = [ad_gens[i] for i in shuffle_ix]
    print("Selected number of aug. data generators: ", len(shuffle_ix))

    return ad_gens_g

# generates and saves a sample of the augmented data generators

def save_aug_data_sample(ad_gens, model, target_size, data_path, output, n_plot=30, set_return=False):
    stdout = silence() # filters the stdout from 'flow_from_directory'

    sample_folder = join(output, 'aug_data_sample')

    if os.path.exists(sample_folder): shutil.rmtree(sample_folder)
    os.mkdir(sample_folder)

    reverse_preprocess_input = set_reverse_preprocess_input_function(model)

    sample = np.empty(((0,) + target_size + (3,)))
    for _, gen in enumerate(ad_gens):

        if _ < n_plot:
            for i, j in gen.flow_from_directory(data_path, target_size=target_size,
                                                batch_size=1, seed=0,
                                                save_to_dir=sample_folder):
                i = reverse_preprocess_input(i)
                sample = np.append(sample, i, axis=0);
                sample_lab = j

                break
        else: break
    speak(stdout)
    s_plot_ix = list(np.random.randint(0, sample.shape[0], n_plot))
    plot_ims(sample[s_plot_ix])

    if set_return: return sample, sample_lab

# pre-calculates without-top model output ('features')

def generate_aug_data_features(base_model, ad_gens, data, output_folder, batch_size=16, iters=1, verbose=False):

    cd_folder = join(output_folder, 'compressed_features')

    # if data has previous loaded and saved:

    if os.path.exists(cd_folder):
        print('Loading compressed features..')

        x_adf_train = load_array(join(cd_folder, 'x_ad_feat_train'))
        y_adf_train = load_array(join(cd_folder, 'y_ad_feat_train'))
        x_adf_test = load_array(join(cd_folder, 'x_ad_feat_test'))
        y_adf_test = load_array(join(cd_folder, 'y_ad_feat_test'))

        return x_adf_train, y_adf_train, x_adf_test, y_adf_test

    # if not, generate features and save them:

    print('Processing features..')

    x_adf_train = np.empty(((0,) + base_model.output_shape[1:]))
    y_adf_train = np.empty((0,))

    x_adf_test = np.empty(((0,) + base_model.output_shape[1:]))
    y_adf_test = np.empty((0,))

    for _, i in enumerate(ad_gens):

        #i.fit(data['x_train'])

        adg_train = i.flow(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=False)
        adg_test = i.flow(data['x_test'], data['y_test'], batch_size=batch_size, shuffle=False)

        x_ad_feat_train = base_model.predict_generator(adg_train, iters*adg_train.n//adg_train.batch_size, verbose=0)
        y_ad_feat_train = np.concatenate([data['y_train']]*iters)
        x_ad_feat_test = base_model.predict_generator(adg_test, iters*adg_test.n//adg_test.batch_size, verbose=0)
        y_ad_feat_test = np.concatenate([data['y_test']]*iters)

        x_adf_train = np.append(x_adf_train, x_ad_feat_train, axis=0)
        y_adf_train = np.append(y_adf_train, y_ad_feat_train, axis=0)
        x_adf_test = np.append(x_adf_test, x_ad_feat_test, axis=0)
        y_adf_test = np.append(y_adf_test, y_ad_feat_test, axis=0)

        if verbose and _%10: print("Generator ", _)

    if not os.path.exists(cd_folder):

        os.mkdir(cd_folder)

        # saving
        save_array(join(cd_folder, 'x_ad_feat_train'), x_adf_train)
        save_array(join(cd_folder, 'y_ad_feat_train'), y_adf_train)
        save_array(join(cd_folder, 'x_ad_feat_test'), x_adf_test)
        save_array(join(cd_folder, 'y_ad_feat_test'), y_adf_test)

    return x_adf_train, y_adf_train, x_adf_test, y_adf_test

# rm train_2225.jpg
# rm train_2276.jpg
