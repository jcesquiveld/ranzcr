from ranzcr.model import RanzcrClassifier

if __name__ == '__main__':
    MODEL_PATH = '../models/basecamp/resnet200d-epoch_11-fold_0-val_score_0.947.ckpt'
    model = RanzcrClassifier.load_from_checkpoint(MODEL_PATH)
    print(model.hparams)