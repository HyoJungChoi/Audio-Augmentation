import random 
from sklearn.metrics import roc_curve

def eer():
    
    rd=[random.choice(fnms_eval) for i in range(len(fnms_eval))]

    partition = {'train': fnms_tr, 
             'dev': fnms_dev,
             'eval': rd}
    
    evaluation_generator_cqt= DataGenerator_ag(partition['eval'],  input_shape = input_shape_cqt, **params_cqt_eval)
    scores_eval = model.predict_generator(generator=evaluation_generator_cqt, verbose=1)

    label_eval=[]
    for fnm in partition['eval'] :
        label_eval.append(fnm['label'])
    
    fpr, tpr, threshold = roc_curve(label_eval, scores_eval, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return(EER)


#### 10회 복원추출 후 EER

erdyn=[]

while True:
    erdyn.append(eer())
    
    if len(erdyn)==10:
        break