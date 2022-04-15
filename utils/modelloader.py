import os
import logging
import torch


def save(filepath, filename, model, optimizer, epoch, best_metric):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    model_path = '%s/%s' % (filepath, filename)
    logging.info("saving model: %s" % model_path)
    torch.save({'model': model, 'optimizer': optimizer, 'epoch': epoch, 'best_metric': best_metric}, model_path)


def load(filepath):
    logging.info('loading model: %s' % filepath)
    ckpt = torch.load(filepath)
    return ckpt['model'], ckpt['optimizer'], ckpt['epoch'], ckpt['best_metric']
