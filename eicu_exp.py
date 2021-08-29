from layer.output_layer import *
from layer.Decoder import *
from layer.Encoder import *
from utils import *
import tensorflow as tf
import copy
previous_visit = 0
predicted_visit = 1
batch_size = 128
epochs = 110

MAX_CINDEX = 0
hidden_size1_final = 0
hidden_size2_final = 0
hidden_size3_final = 0
hidden_size4_final = 0
a1_final = 0
a2_final = 0
a3_final = 0
learning_rate_final = 0
l2_regularization_final = 0
MASK_RATE = 1
SHUFFLE_RATE = 1


def f_get_risk_predictions2(o_list, pred, last_meas_, pre_time):
    _, num_Category = np.shape(pred)
    pred_s = np.zeros(np.shape(pred))
    pred_a = np.zeros(np.shape(pred))
    for i in range(pred_s.shape[0]):
        l = int(last_meas_[i][0])
        # print(l)
        pred_s[i, l:(l+pre_time+1)] = pred[i, l:(l+pre_time+1)]
        # pred_a[i,  l:] = pred[i,  l:]
        for o in range(o_list.shape[1]):
            pred_a[i,  l:] = pred_a[i,  l:] + o_list[i, o,  l:]
    risk = np.sum(pred_s, axis=1)  # risk score until eval_time
    risk = risk / (np.sum(pred_a, axis=1))

    return risk
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i]+1)] = 1 # last measurement time

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i].max()!= 0:  #not censored
            mask[i,:,int(time[i])] = label[i]
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i]) # last measurement time
            t2 = int(time[i]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask


def train_model_all_cs(train_set, test_set, feature_dims, hidden_size, num_category,
                    num_event, learning_rate, l2_regularization, MASK_RATE, SHUFFLE_RATE, ith_fold):
    feature = train_set.x
    # shuffle_index = list(range(5))
    # np.random.shuffle(shuffle_index)
    # shuffle_index = [4, 3, 2, 1, 0]
    # print(shuffle_index)
    # 超参数
    # hidden_size = 2 ** (int(hidden_size))
    # learning_rate = 10 ** learning_rate
    # l2_regularization = 10 ** l2_regularization

    print('feature_size----{}'.format(feature_dims))
    print('hidden_size{}-----learning_rate{}----l2_regularization{}----'.format(hidden_size, learning_rate,
                                                                                l2_regularization))

    encoder = Encoder(hidden_size=128, model_type='LSTM')
    fc_net = []
    for i in range(num_event):
        fc_net.append(FC_SAP(hidden_size=hidden_size, num_category=num_category))
    decoder = Decoder(hidden_size=32, feature_dims=feature_dims, model_type='TimeLSTM2')

    mlp = MLP2(hidden_size=num_category * num_event)
    logged = set()
    result = []
    result_index = []
    shuffle_index = [4, 3, 2, 1, 0]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    while train_set.epoch_completed < epochs:
        # 输入input
        input_x_train_, input_t_train_, input_y_train_, input_day_train_, input_mask1_train_, input_mask2_train_, input_mask3_train_ = train_set.next_batch(batch_size)
        visit_len = input_x_train_.shape[1]
        # 梯度下降更新
        with tf.GradientTape() as tape:
            # 做mask
            mask_index = int(np.random.random() * (visit_len - 1))
            if MASK_RATE == 0:
                mask_input_x_train = copy.deepcopy(input_x_train_)
            else:
                mask_input_x_train = copy.deepcopy(input_x_train_)
                random_select_list = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(random_select_list)
                random_select_list_sort = pd.Series(random_select_list).sort_values()
                random_select_list_sort_index = random_select_list_sort.index[
                                                :int(mask_input_x_train.shape[0] * MASK_RATE)]
                mask_input_x_train[random_select_list_sort_index, mask_index, :] = 0
            # 生成预测的序列
            trajectory_encode_last_h, trajectory_encode_h_list = encoder(mask_input_x_train, batch=batch_size)
            predicted_trajectory_x_train, predicted_trajectory_decode_h = decoder(
                (trajectory_encode_last_h, input_day_train_[:,:,0]),
                predicted_visit=visit_len,
                batch=batch_size)
            gen_mse_loss = tf.reduce_mean(
                tf.keras.losses.mse(input_x_train_[:, mask_index, :], predicted_trajectory_x_train[:, mask_index, :]))
            mask_input_x_train_add = copy.deepcopy(mask_input_x_train)
            mask_input_x_train_add[:, mask_index, :] = predicted_trajectory_x_train[:, mask_index, :]
            mask_input_x_train_trajectory_generation_decode_h, mask_input_x_train_trajectory_generation_h_list = encoder(
                mask_input_x_train_add,
                batch=batch_size)
            real_decode_h, real_trajectory_encode_h_list = encoder(input_x_train_, batch=batch_size)
            clf_loss = 0
            neg_likelihood_loss = 0
            predicted_output = []

            for i in range(num_event):

                predicted_output_ = fc_net[i](real_decode_h)
                predicted_output.append(predicted_output_)

            out = tf.stack(predicted_output, axis=1)  # stack referenced on subject
            out = tf.reshape(out, [-1, num_event * hidden_size])

            out = mlp(out)
            out = tf.reshape(out, [-1, num_event, num_category])
            for i in range(num_event):
                predicted_output_ = out[:, i, :]
                I_2 = input_y_train_[:,-1,i].reshape((-1, 1)).astype('float32')
                ### LOSS-FUNCTION 1 -- Log-likelihood loss
                denom = 1 - tf.reduce_sum(input_mask1_train_[:, i, :] * predicted_output_,
                                          axis=1)  # make subject specific denom.
                denom = tf.clip_by_value(denom, tf.cast(1e-08, dtype=tf.float32),
                                         tf.cast(1. - 1e-08, dtype=tf.float32))

                # for uncenosred: log P(T=t,K=k|x,Y,t>t_M)
                tmp1 = tf.reduce_sum(input_mask2_train_[:, i, :] * predicted_output_, axis=1)
                tmp1 = I_2 * log(div(tmp1, denom))

                # for censored: log \sum P(T>t|x,Y,t>t_M)
                tmp2 = tf.reduce_sum(input_mask2_train_[:, i, :] * predicted_output_, axis=1),

                tmp2 = (1. - I_2) * log(div(tmp2, denom))

                neg_likelihood_loss += - tf.reduce_mean(tmp1 + tmp2)

            if SHUFFLE_RATE == 0:
                shuffled_input_x_train = input_x_train_
            else:
                random_select_list2 = np.array(range(mask_input_x_train.shape[0]))
                np.random.shuffle(shuffle_index)
                random_select_list_sort2 = pd.Series(random_select_list2).sort_values()
                random_select_list_sort_index2 = random_select_list_sort2.index[
                                                 :int(mask_input_x_train.shape[0] * SHUFFLE_RATE)]

                shuffled_input_x_train_mask1 = np.ones_like(input_x_train_)
                shuffled_input_x_train_mask1[random_select_list_sort_index2, :, :] = 0
                shuffled_input_x_train_mask0 = np.zeros_like(input_x_train_)
                shuffled_input_x_train_mask0[random_select_list_sort_index2, :, :] = 1
                shuffled_input_x_train = (input_x_train_ * shuffled_input_x_train_mask0)[:, shuffle_index,
                                         :] + input_x_train_ * shuffled_input_x_train_mask1

            # shuffled_input_x_train = input_x_train_[:, shuffle_index, :]
            shuffled_generated_decode_h, shuffled_generated_decode_h_list = encoder((shuffled_input_x_train),
                                                                                    batch=batch_size)



            # 对比学习
            contrast_loss_matrix = tf.matmul(shuffled_generated_decode_h, tf.transpose(real_decode_h))
            contrast_loss_numerator = tf.linalg.diag_part(contrast_loss_matrix)
            contrast_loss_denominator = tf.reduce_sum(tf.math.exp(contrast_loss_matrix),
                                                      axis=1)
            contrast_loss_cs = -tf.reduce_mean(contrast_loss_numerator - tf.math.log(contrast_loss_denominator))

            # 对比学习2
            contrast_loss_trajectory_generation = tf.matmul(mask_input_x_train_trajectory_generation_decode_h,
                                                            tf.transpose(real_decode_h))
            contrast_loss_trajectory_generation_numerator = tf.linalg.diag_part(contrast_loss_trajectory_generation)
            contrast_loss_trajectory_generation_denominator = tf.reduce_sum(
                tf.math.exp(contrast_loss_trajectory_generation),
                axis=1)
            contrast_loss_cg = -tf.reduce_mean(
                contrast_loss_trajectory_generation_numerator - tf.math.log(
                    contrast_loss_trajectory_generation_denominator))

            # 对比学习3
            contrast_loss_risk = 0
            for i in range(num_event):
                label = input_y_train_[:, -1, i].reshape((-1, 1)).astype('float32')
                h_e = tf.gather(real_decode_h, tf.where(label == 1)[:, 0])
                h_0 = tf.gather(real_decode_h, tf.where(label != 1)[:, 0])
                contrast_loss_risk_numerator = tf.matmul(h_e, tf.transpose(h_e))
                contrast_loss_risk_denominator = tf.math.exp(tf.matmul(h_e, tf.transpose(h_0)))
                contrast_loss_risk += -tf.reduce_sum(
                    contrast_loss_risk_numerator - tf.math.log(tf.reduce_sum(contrast_loss_risk_denominator)))

            whole_loss = gen_mse_loss * 0.1 + clf_loss * 1 + neg_likelihood_loss * 1 + contrast_loss_cs * 0.01 + contrast_loss_cg * 0.01 + contrast_loss_risk*0.1
            fc_net_variables = []
            for i in range(3):
                fc_net_variables.extend([var for var in fc_net[i].trainable_variables])

            encoder_variables = [var for var in encoder.trainable_variables]
            for weight in encoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            decoder_variables = [var for var in decoder.trainable_variables]
            for weight in decoder.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)
            sap_variables = [var for var in mlp.trainable_variables]
            for weight in mlp.trainable_variables:
                whole_loss += tf.keras.regularizers.l2(l2_regularization)(weight)

            variables = sap_variables + encoder_variables + fc_net_variables + decoder_variables
            gradient = tape.gradient(whole_loss, variables)
            optimizer.apply_gradients(zip(gradient, variables))

            if train_set.epoch_completed % 1 == 0 and train_set.epoch_completed not in logged:

                logged.add(train_set.epoch_completed)

                input_x_test = test_set.x
                input_y_test = test_set.y
                input_t_test = test_set.t
                input_day_test = test_set.day
                batch_test = input_x_test.shape[0]
                context_state_test, real_trajectory_encode_h_list_test = encoder(input_x_test, batch=batch_test)

                c_index_output = []
                label_test = input_y_test[:, -1].reshape((-1, 1)).astype('float32')
                ett_test = input_t_test[:, -1].reshape((-1, 1)).astype('float32')
                day_test = input_day_test[:, -1].reshape((-1, 1)).astype('float32')
                last_meas_ = ett_test + day_test
                predicted_o_list = []
                for i in range(num_event):
                    predicted_o_list.append(fc_net[i](context_state_test))
                out = np.stack(predicted_o_list, axis=1)  # stack referenced on subject
                out = np.reshape(out, [-1, num_event * hidden_size])
                out = mlp(out)
                out = np.reshape(out, [-1, num_event, num_category])

                for i in range(num_event):
                    predicted_output_test = out[:, i, :]
                    # predicted_risk_test = f_get_risk_predictions(predicted_output_test, last_meas_, 8)
                    predicted_risk_test = f_get_risk_predictions2(out, predicted_output_test, last_meas_, 8)
                    predicted_output.append(predicted_risk_test)
                    I_2 = input_y_test[:, -1, i].reshape((-1, )).astype('float32')
                    # T_2 = ett_test * I_2 + np.ones_like(ett_test) * (1 - I_2) * 8
                    #print(np.shape(ett_test),np.shape(predicted_risk_test),np.shape(I_2))
                    c_index = concordance_index(ett_test[:,0], -predicted_risk_test, I_2)
                    c_index_output.append(c_index)

                c_index_output.append(np.mean(c_index_output))
                result.append(c_index_output)
                result_index.append(np.sum(c_index_output))

                if (train_set.epoch_completed + 1) % 2 == 0:
                    print(
                        '----epoch:{}, whole_loss:{}, contrast_loss:{},contrast_loss2:{},clf_loss:{},neg_likelihood_loss:{},gen_loss:{}, c_index:{}'.format(train_set.epoch_completed, whole_loss, contrast_loss_cs, contrast_loss_cg, clf_loss ,neg_likelihood_loss,
                                           gen_mse_loss, c_index_output))

        tf.compat.v1.reset_default_graph()
    result = np.array(result)
    result_index = np.array(result_index)
    max_i = np.where(result_index == result_index.max())
    print(np.squeeze(result[max_i, :]))
    return np.squeeze(result[max_i, :])


def data_spilt(features, labels, event_num):
    split = []
    for i in range(event_num):
        features_i = features[i].reshape(features[i].shape[0], -1)
        labels_i = labels[i][:,:,-1].reshape(labels[i].shape[0], -1)
        split_i = StratifiedShuffleSplit(5, 0.2, 0.8, 1).split(features_i,labels_i)
        split.append(split_i)
    return split


def data_divided(data, visit_len, feature_dims):
    labels = data[:, 42:47].astype('float32').reshape(-1, visit_len, 5)
    features = data[:, 0:39].astype('float32').reshape(-1, visit_len, 39)
    days = data[:, 40].astype('float32').reshape(-1, visit_len, 1)
    ett = data[:, 41].astype('float32').reshape(-1, visit_len, 1)
    return features, labels, days, ett


def data_concat(split,features,labels, days, ett, mask1, mask2, mask3, event_num):
    train_index_0, test_index_0 = next(split[0])
    train_features = features[0][train_index_0]
    train_labels = labels[0][train_index_0]
    train_days = days[0][train_index_0]
    train_ett = ett[0][train_index_0]
    train_mask1 = mask1[0][train_index_0]
    train_mask2 = mask2[0][train_index_0]
    train_mask3 = mask3[0][train_index_0]

    test_features = features[0][test_index_0]
    test_labels = labels[0][test_index_0]
    test_days = days[0][test_index_0]
    test_ett = ett[0][test_index_0]
    test_mask1 = mask1[0][test_index_0]
    test_mask2 = mask2[0][test_index_0]
    test_mask3 = mask3[0][test_index_0]

    for i in range(event_num-1):
        train_index, test_index = next(split[i + 1])
        train_features = np.concatenate([train_features, features[i + 1][train_index]], axis=0)
        train_labels = np.concatenate([train_labels, labels[i + 1][train_index]], axis=0)
        train_days = np.concatenate([train_days, days[i + 1][train_index]], axis=0)
        train_ett = np.concatenate([train_ett, ett[i + 1][train_index]], axis=0)
        train_mask1 = np.concatenate([train_mask1, mask1[i + 1][train_index]], axis=0)
        train_mask2 = np.concatenate([train_mask2, mask2[i + 1][train_index]], axis=0)
        train_mask3 = np.concatenate([train_mask3, mask3[i + 1][train_index]], axis=0)

        test_features = np.concatenate([test_features, features[i + 1][test_index]], axis=0)
        test_labels = np.concatenate([test_labels, labels[i + 1][test_index]], axis=0)
        test_days = np.concatenate([test_days, days[i + 1][test_index]], axis=0)
        test_ett = np.concatenate([test_ett, ett[i + 1][test_index]], axis=0)
        test_mask1 = np.concatenate([test_mask1, mask1[i + 1][test_index]], axis=0)
        test_mask2 = np.concatenate([test_mask2, mask2[i + 1][test_index]], axis=0)
        test_mask3 = np.concatenate([test_mask3, mask3[i + 1][test_index]], axis=0)

    return DataSetWithMask2(train_features, train_ett, train_labels, train_days, train_mask1, train_mask2,
                            train_mask3), DataSetWithMask2(test_features, test_ett, test_labels, test_days, test_mask1,
                                                           test_mask2, test_mask3)


def mask_divide(label,mask1,mask2,mask3):



    mask1 = [mask1[np.where(label == 0)],mask1[np.where(label == 1)],mask1[np.where(label == 2)],mask1[np.where(label == 3)],mask1[np.where(label == 4)],mask1[np.where(label == 5)],mask1[np.where(label == 6)],mask1[np.where(label ==7)]]
    mask2 = [mask2[np.where(label == 0)], mask2[np.where(label == 1)], mask2[np.where(label == 2)],
             mask2[np.where(label == 3)], mask2[np.where(label == 4)], mask2[np.where(label == 5)], mask2[np.where(label == 6)], mask2[np.where(label == 7)]]
    mask3 = [mask3[np.where(label == 0)], mask3[np.where(label == 1)], mask3[np.where(label == 2)],
             mask3[np.where(label == 3)], mask3[np.where(label == 4)], mask3[np.where(label == 5)], mask3[np.where(label == 6)], mask3[np.where(label == 7)]]
    return mask1,mask2,mask3


def experiment(MASK_RATE,SHUFFLE_RATE):
    # MASK_RATE = 1
    # SHUFFLE_RATE = 1
    train_repeat = 5
    data = import_eicu_data()
    data['death_reason'] = data['AcuteRespiratoryFailure'] + data['AcuteRenalFailure'] * 2 + data['Pneumonia'] * 4
    data['max_time'] = data['ett'] + data['time']
    data_copy = data.values
    data_copy = np.reshape(data_copy, [-1, 5, data.shape[1]])
    data_final_copy = data_copy[:, -1, :]
    data_final_copy[:, -1].max()
    last_meas = data_final_copy[:, -8]  # pat_info[:, 3] contains age at the last measurement
    label = data_final_copy[:, -2]  # two competing risks
    time = data_final_copy[:, -1]  # age when event occurred
    num_category = int(np.max(time) * 1.1)  # or specifically define larger than the max tte
    num_event = 3
    data_final_copy = pd.DataFrame(data_final_copy, columns=data.columns)
    mask1 = f_get_fc_mask1(last_meas, num_event, num_category)
    mask2 = f_get_fc_mask2(time, data_final_copy.values[:,43:46], num_event, num_category)
    mask3 = f_get_fc_mask3(time, -1, num_category)
    mask1, mask2, mask3 = mask_divide(label, mask1, mask2, mask3)


    event_group_type = 2**num_event
    data_dict = dict()
    for i in range(event_group_type):
        data_dict[i] = np.asarray(data[data['death_reason'] == i])
    index_list = []
    feature_dims = 39
    hidden_size = 30
    visit_len = 5
    features = []
    labels = []
    days = []
    ett = []
    for i in range(event_group_type):
        features_0, labels_0, days_0, ett_0 = data_divided(data_dict[i], visit_len, feature_dims)
        features.append(features_0)
        labels.append(labels_0)
        days.append(days_0)
        ett.append(ett_0)
    model_type = 'train_model_eicu'
    l2_regularization = 0.00001
    learning_rate = 0.0002
    k_folds = 5

    print(model_type)
    for i in range(train_repeat):
        print("iteration number: %d" % i)
        k_folds = 5
        test_size = 0.2
        split = data_spilt(features, labels, event_group_type)


        for ith_fold in range(k_folds):
            print('{} th fold of {} folds'.format(ith_fold, k_folds))

            train_set, test_set = data_concat(split, features, labels, days, ett, mask1, mask2, mask3, event_group_type)
            if model_type == 'train_model_eicu':
                # RNN
                c_index1, c_index2, c_index3, c_index4 = train_model_all_cs(
                    train_set=train_set,
                    test_set=test_set,
                    feature_dims=feature_dims,
                    hidden_size=hidden_size,
                    num_category=num_category,
                    num_event=num_event,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    MASK_RATE=MASK_RATE,
                    SHUFFLE_RATE=SHUFFLE_RATE,
                    ith_fold=ith_fold
                )
           
            else:
                c_index1, c_index2, c_index3, c_index4, c_index5 = train_model_all_cs(
                    train_set=train_set,
                    test_set=test_set,
                    feature_dims=feature_dims,
                    hidden_size=hidden_size,
                    num_category=num_category,
                    num_event=num_event,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    MASK_RATE=MASK_RATE,
                    SHUFFLE_RATE=SHUFFLE_RATE,
                    ith_fold=ith_fold
                )
           
            index_list.append([c_index1, c_index2, c_index3, c_index4])
        print('epoch  {}-----all_ave  {}'.format(i, np.mean(index_list, axis=0)))
        auc_list = pd.DataFrame(index_list, columns=['c_index1', 'c_index2', 'c_index3', 'c_index4'])
        auc_list.to_excel(
            'result//eicu_result_{}_lr={}_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE,
                                                                                SHUFFLE_RATE),
            index=False)
    auc_list = pd.DataFrame(index_list, columns=['c_index1', 'c_index2', 'c_index3', 'c_index4'])
    auc_list.to_excel(
        'result//eicu_result_{}_lr={}_mask_rate={}_srate={}.xlsx'.format(model_type, learning_rate, MASK_RATE,
                                                                            SHUFFLE_RATE),
        index=False)



if __name__ == '__main__':
    experiment(1,1)
