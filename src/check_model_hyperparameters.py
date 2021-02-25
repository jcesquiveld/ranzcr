from ranzcr.model import RanzcrClassifier

if __name__ == '__main__':
    MODEL_PATH = '../models/apprentice/resnet200d-epoch_06-fold_3-val_score_0.935.ckpt'
    model = RanzcrClassifier.load_from_checkpoint(MODEL_PATH)
    print(model.hparams)