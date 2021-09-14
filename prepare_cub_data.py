import os
import re
import string

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from utils import sample_batch_img, save_dict, load_dict, save_pickle, load_pickle, make_dir
from config import cfg


def prepare_dirs():
    make_dir(cfg.DIRS.PROC_DATA_DIR)
    make_dir(cfg.DIRS.PROC_TXT_DIR)
    make_dir(cfg.DIRS.RESULTS_DIR)
    make_dir(cfg.DIRS.CHECKPOINTS_DIR)
    make_dir(cfg.DIRS.TRAIN_LOGS_DIR)

    for dir in cfg.DIRS.PROC_IMG_DIRS:
        make_dir(dir)

    for dir in cfg.DIRS.PROC_STYLE_DIR:
        make_dir(dir)

    for dir in cfg.DIRS.SAMPLE_BATCH_DIR:
        make_dir(dir)


def generate_word2idx():
    """
    Generates and saves word2idx dict,
    where every word from txt caption is a key
    and word's index is it's value:
    {word_1: idx_1, word_2: idx_2, ... word_n: idx_n}
    """
    word2idx = {}
    word_idx = 1

    for root, subFolders, files in os.walk(cfg.DIRS.RAW_TXT_DIR):
        if files:

            for filename in files:
                with open(f"{root}/{filename}", "r") as file:

                    for line in file:
                        line = line.translate(str.maketrans('', '', string.punctuation))
                        word_list = re.split('�|,| |\n|_|-|!|\+', line)

                        for word in word_list:
                            if word not in word2idx.keys() and word != "":
                                word2idx[word] = word_idx
                                word_idx += 1

    save_dict(f"{cfg.DIRS.PROC_DATA_DIR}/word2idx.json", word2idx)


def generate_idx2word():
    """
    Generates and saves reversed word2idx dict
    for digit to string transition:
    {idx_1: word_1, idx_2: word_2, ... idx_n: word_n}
    """
    word2idx = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/word2idx.json")
    idx2word = {v: k for k, v in word2idx.items()}
    save_dict(f"{cfg.DIRS.PROC_DATA_DIR}/idx2word.json", idx2word)


def generate_file2idx():
    file2idx = {}
    with open(f"{cfg.DIRS.RAW_DATA_DIR}/images.txt") as file:
        for line in file:
            values = line.split()
            idx = int(values[0])
            fname = values[1:]
            file2idx[fname[0]] = idx

    save_dict(f"{cfg.DIRS.PROC_DATA_DIR}/file2idx.json", file2idx)


def generate_idx2bbox():
    idx2bbox = {}
    with open(f"{cfg.DIRS.RAW_DATA_DIR}/bounding_boxes.txt") as file:
        for line in file:
            values = line.split()
            idx = int(values[0])
            bbox_coords = np.asarray(values[1:], dtype='float32').astype(np.int32)
            idx2bbox[idx] = bbox_coords

    save_pickle(f"{cfg.DIRS.PROC_DATA_DIR}/idx2bbox.pickle", idx2bbox)


def generate_pretrained_word2vec():
    """
    Loads pretrained glove vectors, generates and saves
    word2vec dict
    """
    print('Loading word vectors...')
    word2vec = {}
    with open(f"{cfg.DIRS.RAW_DATA_DIR}/glove.6B.{cfg.ENC.SEQ_DIM}d.txt") as embed_file:
        # word vec[0] vec[1] vec[2] ...
        for line in embed_file:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    save_pickle(f"{cfg.DIRS.PROC_DATA_DIR}/word2vec_glove.pickle", word2vec)


def generate_pretrained_embeds():
    """
    Generates and saves an embedding matrix of pretrained word vectors
    as a numpy.ndarray.
    """
    word2vec = load_pickle(f"{cfg.DIRS.PROC_DATA_DIR}/word2vec_glove.pickle")
    word2idx = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/word2idx.json")

    # prepare embedding matrix
    num_words = min(cfg.ENC.VOCAB_SIZE, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, cfg.ENC.SEQ_DIM), dtype=np.float32)
    for word, i in word2idx.items():
        if i < cfg.ENC.VOCAB_SIZE:

            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    np.save(f"{cfg.DIRS.PROC_DATA_DIR}/embed_mtrx_glove.npy", embedding_matrix)


def generate_file2caption_digit():
    """
    Generates and saves a dictionary, where
    keys are filenames, values are lists of
    all text captions, digitized.
    """
    file2caption = {}
    word2idx = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/word2idx.json")

    for root, subFolders, files in os.walk(cfg.DIRS.RAW_TXT_DIR):
        if files:

            for filename in files:
                file_captions = []

                with open(f"{root}/{filename}", "r") as file:
                    for line in file:
                        line = line.translate(str.maketrans('', '', string.punctuation))
                        word_list = re.split('�|,| |\n|_|-|!|\+', line)
                        word_idx_list = [word2idx[word] for word in word_list if word in word2idx]

                        if len(word_idx_list) > cfg.ENC.MAX_SEQ_LEN:
                            word_idx_list = word_idx_list[:cfg.ENC.MAX_SEQ_LEN]

                        if len(word_idx_list) > 1:
                            word_idx_list += [0] * (cfg.TRAIN.ENC.MAX_SEQ_LEN - len(word_idx_list))
                            file_captions.append(word_idx_list)

                file2caption[filename[:-4]] = file_captions

    save_dict(f"{cfg.DIRS.PROC_DATA_DIR}/file2caption_digit.json", file2caption)


def process_images():
    """
    Processes and saves all raw images, as resized numpy.ndarrays,
    along with their digitized captions saved as numpy.ndarrays
    and stored im PROC_TXT_DIR
    """
    file2caption = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/file2caption_digit.json")
    file2idx = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/file2idx.json")
    idx2bbox = load_pickle(f"{cfg.DIRS.PROC_DATA_DIR}/idx2bbox.pickle")

    img_idx = 0
    for root, subFolders, files in os.walk(cfg.DIRS.RAW_IMG_DIR):
        if files:

            for filename in files:
                fpath = f"{root}/{filename}"
                img = cv2.imread(fpath)
                img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                bbox = idx2bbox[file2idx[fpath[20:]]]

                # correct bbox
                y_max, x_max, _ = img.shape
                img_c_x = bbox[0] + int(0.5 * bbox[2])
                img_c_y = bbox[1] + int(0.5 * bbox[3])
                box_res = np.maximum(bbox[2], bbox[3])
                x0 = np.clip(img_c_x - int(0.5*box_res*1.2), a_min=0, a_max=x_max)
                x1 = np.clip(img_c_x + int(0.5*box_res*1.2), a_min=0, a_max=x_max)
                y0 = np.clip(img_c_y - int(0.5*box_res*1.2), a_min=0, a_max=y_max)
                y1 = np.clip(img_c_y + int(0.5*box_res*1.2), a_min=0, a_max=y_max)

                # correct channels
                if img.shape[-1] != 3:
                    img = np.repeat(
                        np.expand_dims(img, axis=-1), repeats=3, axis=-1
                    )

                captions = np.array(file2caption[filename[:-4]])
                img = img[y0:y1, x0:x1]

                for res, res_dir in zip(cfg.TRAIN.RES_LIST, cfg.DIRS.PROC_IMG_DIRS):
                    img_res = np.array(cv2.resize(img, (int(res), int(res)), cv2.INTER_AREA))
                    img_res = ((img_res / 255) * 2) - 1

                    np.save(
                        f"{res_dir}/{img_idx}.npy", img_res
                    )

                np.save(
                    f"{cfg.DIRS.PROC_TXT_DIR}/{img_idx}.npy", captions
                )

                img_idx += 1


def process_style_images():
    """
    Processes and saves all raw style images, as resized numpy.ndarrays,
    """

    img_idx = 0
    for root, subFolders, files in os.walk(cfg.DIRS.RAW_IMG_DIR):
        if files:

            for filename in files:
                fpath = f"{root}/{filename}"
                img = cv2.imread(fpath)
                img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # correct channels
                if img.shape[-1] != 3:
                    img = np.repeat(
                        np.expand_dims(img, axis=-1), repeats=3, axis=-1
                    )

                for res, res_dir in zip(cfg.TRAIN.RES_LIST, cfg.DIRS.PROC_STYLE_DIRS):
                    img_res = np.array(cv2.resize(img, (int(res), int(res)), cv2.INTER_AREA))
                    img_res = ((img_res / 255) * 2) - 1

                    np.save(
                        f"{res_dir}/{img_idx}.npy", img_res
                    )

                img_idx += 1


def check_data(batch_size: int):
    """
    Randomly samples and saves dataset images in all resolutions
    to .jpg files, with their text captions as titles.
    """

    idx2word = load_dict(f"{cfg.DIRS.PROC_DATA_DIR}/idx2word.json")

    plt.figure()
    for res, res_dir in zip(cfg.TRAIN.RES_LIST, cfg.DIRS.SAMPLE_BATCH_DIR):
        img_batch, cap_batch = sample_batch_img(res=int(res), batch_size=batch_size)
        img_batch = img_batch * 0.5 + 0.5
        i = 0

        for img, cap in zip(img_batch, cap_batch):
            txt_cap = ' '.join([idx2word[str(int(w_idx))] for w_idx in cap if w_idx != 0])
            plt.clf()
            plt.title(txt_cap)
            plt.imshow(img)
            plt.tight_layout()
            plt.savefig(f"{res_dir}/{i}.jpg")
            plt.close()
            i += 1


if __name__ == '__main__':
    # # Create directories
    prepare_dirs()

    # #
    # generate_file2idx()
    # generate_idx2bbox()
    # generate_word2idx()
    # generate_idx2word()
    # generate_file2caption_digit()
    process_images()
    check_data(batch_size=128)
    # generate_pretrained_word2vec()
    # generate_pretrained_embeds()
    pass