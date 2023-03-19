# -*- coding: UTF-8 -*-
'''
@Time    : 2022/9/17 11:20
@Author  : 魏林栋
@Site    : 
@File    : Tools.py
@Software: PyCharm
'''
import os
import cv2
import yaml
import pydicom
import pandas as pd
import numpy as np
import nibabel as nib
from radiomics import featureextractor
from PySide2 import QtUiTools, QtCore
from concurrent.futures import ThreadPoolExecutor
# 画图
import matplotlib.pyplot as plt
import seaborn as sns
# 机器学习相关
from sklearn.linear_model import LassoCV, Lasso
from scipy.stats import ttest_ind
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from lib.metrics import analysis_pred_binary, draw_roc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# pd.set_option('display.width', 30)  # 设置字符显示宽度
# pd.set_option('display.max_rows', 10)  # 设置显示最大行
# pd.set_option('display.max_columns', 10)  # 设置显示最大列，None为显示所有列

class UiLoader(QtUiTools.QUiLoader):
    _baseinstance = None

    def createWidget(self, classname, parent=None, name=''):
        if parent is None and self._baseinstance is not None:
            widget = self._baseinstance
        else:
            widget = super(UiLoader, self).createWidget(classname, parent, name)
            if self._baseinstance is not None:
                setattr(self._baseinstance, name, widget)
        return widget

    def loadUi(self, uifile, baseinstance=None):
        self._baseinstance = baseinstance
        widget = self.load(uifile)
        QtCore.QMetaObject.connectSlotsByName(widget)
        return widget

class Tools:
    dirs = {}
    def __init__(self):
        self.images = []
        self.extractor = None
        self.modelName = 'RandomForest'
        self.model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    def importPath(self, dir):
        dir = os.path.normpath(dir)
        opath, _ = os.path.split(dir)
        self.dirs[_] = {}
        count = len(dir.split(os.sep))
        for i, j, k in os.walk(dir):
            files = i.split(os.sep)
            self.findAddress(0, files[count - 1:], self.dirs[_], k)
        return self.dirs

    def findAddress(self, index, files, dirs, k):
        if index < len(files) - 1:
            key = files[index + 1]
            if key not in dirs.keys():
                dirs[key] = {}
            self.findAddress(index + 1, files, dirs[key], k)
        if index == len(files) - 1:
            dirs['`'] = k
        return

    def normaliza(self, x):
        '''
        :param x: 输入ndarray矩阵
        :return: 归一化矩阵数据
        '''
        np.seterr(divide='ignore', invalid='ignore')
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def dcmToPng(self, address, save_path):
        data = pydicom.read_file(address)
        image = self.normaliza(data.pixel_array)
        cv2.imwrite(save_path, image * 255)

    def deleteTemp(self, path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                self.deleteTemp(f_path)
            else:
                os.remove(f_path)

    def niigzToPng(self, filePath, targetPath):
        self.images = []
        image_arr = nib.load(filePath).get_fdata()
        with ThreadPoolExecutor(max_workers=5) as pool:
            for index in range(image_arr.shape[2]):
                pool.submit(lambda cxp: self.saveImage(*cxp), (image_arr[:, :, index], index, targetPath.replace('\\', '/')))
        return self.images

    def saveImage(self, image, index, targetPath):
        address = f'{targetPath}/{index}.png'
        image = self.normaliza(image)
        image = cv2.resize(image, (512,512))
        cv2.imwrite(address, image*255.0)
        self.images.append(address)

    def createExtractor(self, settings=None,yamlPath='./exampleCT.yaml', params=None):
        if self.extractor is None:
            if settings is None:
                with open(yamlPath, 'r') as f:
                    settings = yaml.load(f.read(), Loader=yaml.FullLoader)
            if params is None:
                params = {'correctMask': True}
            # 创建extractor类
            self.extractor = featureextractor.RadiomicsFeatureExtractor(settings, **params)

    def featurExtract(self, df, index, ifcol=False):
        columns = ['ID', 'label']
        line = []
        data = df.iloc[index]['dataPath']
        mask = df.iloc[index]['maskPath']
        ID = df.iloc[index]['ID']
        label = df.iloc[index]['label']
        line.append(ID)
        line.append(label)
        featureVector = self.extractor.execute(data, mask, label=1)
        for featureName in featureVector:
            imageType, featureClass, feature = featureName.split('_')
            if imageType == 'diagnostics':
                continue
            if ifcol:
                if featureName not in columns:
                    columns.append(featureName)
            line.append(featureVector[featureName])
        if ifcol:
            return line, columns
        return line

    def normalize_df(self, data, not_norm):
        columns = [c for c in data.columns if c not in not_norm]
        for column in columns:
            max_ = np.max(data[column])
            min_ = np.min(data[column])
            data[column] = (data[column] - min_) / (max_ - min_)
        return data

    def clinic_statistic(self, data, stats_columns):
        ulabels = data['label'].unique()  # 查看label里分几类，通常情况下是分 0 和 1 （即二分类）
        if len(ulabels) != 2:
            '''此处写一个错误检测，一定要是二分类问题，如果出错了把错误的部分显示在错误提示里'''
            raise ValueError(f'此接口只能用于2元结果类型的显著性检测，现在{len(ulabels)}元，他们是:{ulabels}，如果是多元显著性检测可以'
                             f'使用`clinic_stats_chi_square`接口')
        stats = {}
        for c in stats_columns:  # c = 特征名
            # Compute p_value
            p_value = ttest_ind(data[data['label'] == 0][c],
                                data[data['label'] == 1][c]).pvalue
            if p_value == 0:
                p_value = ''
            elif p_value < 1e-3:
                p_value = '<0.001'
                # p_value = 1e-6
            s = {'mean': np.mean(data[c]), 'std': np.std(data[c]), '__pvalue__': p_value}
            for label in [0, 1]:
                l_data = data[data['label'] == label][c]
                s.update({f"mean | label={label}": np.mean(l_data), f'std | label={label}': np.std(l_data)})
            # stats的结构是{'特征名'：{'mean': 0.1121, 'std': 0.121, '__pvalue__': 0.11, 'mean | label=0':0.11, 'std | label=0':0.11, 'mean | label=1':0.11, 'std | label=1':0.11}}
            stats[c] = s
        return self.pretty_stats(stats)

    def pretty_stats(self, stats):
        title = ['feature_name']
        title.extend(f"label={label}" for label in ['ALL', '0', '1'])
        title.append('pvalue')  # title = ['feature_name', '-label=All', '-label=0', '-label=1', 'pvalue']
        c_stats = []
        for k, v in stats.items():
            # k = 特征名 v = {'mean': 0.1121, 'std': 0.121, '__pvalue__': 0.11, 'mean | label=0':0.11, 'std | label=0':0.11, 'mean | label=1':0.11, 'std | label=1':0.11}
            group_lines = [k]
            mean_keys = ['mean'] + [f"mean | label={label}" for label in [0, 1]]
            std_keys = ['std'] + [f"std | label={label}" for label in [0, 1]]
            for mk, sk in zip(mean_keys, std_keys):
                if mk in v and sk in v:
                    group_lines.append(f"{v[mk]:.4f}±{v[sk]:.4f}")
                else:
                    group_lines.append('null')
            group_lines.append(v['__pvalue__'])
            c_stats.append(group_lines)
        return pd.DataFrame(c_stats, columns=title)

    def select_feature(self, corr, threshold=0.9, keep=1, verbose=False, topn=1):
        feature_names = corr.columns
        drop_feature_names = [x for x, y in np.array(corr.isna().all().reset_index()) if y]
        has_corr_features = True
        while has_corr_features:
            has_corr_features = False
            corr_fname = {}
            feature2drop = [fname for fname in feature_names if fname not in drop_feature_names]
            for i, fi in enumerate(feature2drop):
                corr_num = 0
                for j in range(i + 1, len(feature2drop)):
                    if abs(corr[fi][feature2drop[j]]) > threshold:
                        corr_num += 1
                corr_fname[fi] = corr_num
            corr_fname = sorted(corr_fname.items(), key=lambda x: x[1], reverse=True)
            for fname, corr_num in corr_fname[:topn]:
                if corr_num >= keep:
                    has_corr_features = True
                    drop_feature_names.append(fname)
                    if verbose:
                        print(f'len {len(feature2drop)}, {fname} has {corr_num} features')

        return [fname for fname in feature_names if fname not in drop_feature_names]

    def lasso_cv_coefs(self, X_data, y_data, alpha_logmin=-3, points=50, cv=10, **kwargs):
        # 每个特征值随lambda的变化
        # 构造以10为基底的等比数列，如果添加一个函数 np.log10（alphas） = [-3.         -2.93877551 -2.87755102 ... -0.06122449  0.        ]
        alphas = np.logspace(alpha_logmin, 0, points)
        # alphas: 用于计算模型的alpha列表。如果为None，自动设置Alpha。  cv: 定义用的是几折交叉验证  n_jobs: 交叉验证期间要使用的CPU核心数量 -1 表示用所有的处理器
        lasso_cv = LassoCV(alphas=alphas, cv=cv, n_jobs=-1).fit(X_data, y_data)
        # coefs: 每一个特征随着alphas（也就是拉姆达）的变化值，这个值是逐渐趋近于零的。shape是（1，145，50）
        _, coefs, _ = lasso_cv.path(X_data, y_data, alphas=alphas)
        coefs = np.squeeze(coefs).T  # 现在的shape是（50，145），146个特征，50个alphas值

        MSEs = lasso_cv.mse_path_  # 这里是从（50，145）的数据里提取出了（50，10）的数据，我判断应该是从145个特征里选择了10个最好的特征
        MSEs_mean = np.mean(MSEs, axis=1)
        MSEs_std = np.std(MSEs, axis=1)

        # 开始绘图
        plt.figure()
        plt.semilogx(lasso_cv.alphas_, coefs, '-', **kwargs)  # 画出特征回归的图像，此函数用于把x轴转换为对数格式的方式显示数据

        lambda_info = ''
        if lasso_cv.alpha_ != 1.0:  # lasso_cv.alpha_：交叉验证选择的惩罚量
            plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
            lambda_info = f"(λ={lasso_cv.alpha_:.6f})"
        plt.xlabel(f'Lambda{lambda_info}')
        plt.ylabel('Coefficients')
        plt.savefig(f'./result/lasso_coefs.png', bbox_inches='tight')
        # 开始画第二幅图
        default_params = {'fmt': "o", 'ms': 3, 'mfc': 'r', 'mec': 'r', 'ecolor': 'b', 'elinewidth': 2, 'capsize': 2,
                          'capthick': 1}
        default_params.update(kwargs)
        plt.figure()
        plt.errorbar(lasso_cv.alphas_, MSEs_mean, yerr=MSEs_std, **default_params)  # 把回归后的均值和方差显示出来
        plt.semilogx()  # 此函数用于以x轴转换为对数格式的方式显示数据

        lambda_info = ''
        if lasso_cv.alpha_ != 1.0:  # 如果惩罚量不是1那么就在 x = 惩罚量 的地方画黑色竖线，达到区分的目的
            plt.axvline(lasso_cv.alpha_, color='black', ls="--", **kwargs)
            lambda_info = f"(λ={lasso_cv.alpha_:.6f})"

        plt.xlabel(f'Lambda{lambda_info}')
        plt.ylabel('MSE')
        ax = plt.gca()
        y_major_locator = MultipleLocator(0.1)  # 设置横坐标间隔是0.1
        ax.yaxis.set_major_locator(y_major_locator)  # 和 MultipleLocator 类结合使用的函数
        plt.savefig('./result/lasso_efficiency.png')
        return lasso_cv.alpha_, ['./result/lasso_coefs.png', './result/lasso_efficiency.png']

    def lassoSiftFeature(self, data, alpha):
        y_data = data[['label']]
        X_data = data.drop(['label'], axis=1)
        column_names = X_data.columns

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0, test_size=0.2)
        models = []
        for label in ['label']:
            clf = Lasso(alpha=alpha)
            clf.fit(X_train, y_train[label])
            models.append(clf)

        COEF_THRESHOLD = 1e-6  # 筛选的特征阈值
        scores = []
        selected_features = []
        for label, model in zip(['label'], models):
            feat_coef = [(feat_name, coef) for feat_name, coef in zip(column_names, model.coef_)
                         # column_names = （145） model.coef_ = （145）
                         if COEF_THRESHOLD is None or abs(coef) > COEF_THRESHOLD]  # feat_coef = [(特征名， 惩罚量)]
            result = [feat for feat, _ in feat_coef]
            result.append('label')
            selected_features.append(result)
            formula = ' '.join([f"{coef:+.6f} * {feat_name}" for feat_name, coef in feat_coef])
            score = f"{label} = {model.intercept_} {'+' if formula[0] != '-' else ''} {formula}"
            scores.append(score)

        plt.figure()
        feat_coef = sorted(feat_coef, key=lambda x: x[1])  # 根据惩罚量大小对特征进行排序
        feat_coef_df = pd.DataFrame(feat_coef, columns=['feature_name', 'Coefficients'])
        feat_coef_df.plot(x='feature_name', y='Coefficients', kind='barh')
        plt.savefig(f'./result/feature_weights.png', bbox_inches='tight')

        return data[selected_features[0]], './result/feature_weights.png'

    def predict(self, X_data, y_data):
        # 交叉验证
        models = self.create_clf_model(['RandomForest'])
        results = self.get_bst_split(X_data, y_data, models, test_size=0.2, metric_fn=roc_auc_score, n_trails=5,
                                cv=True, random_state=0)

        _, (X_train_sel, X_test_sel, y_train_sel, y_test_sel) = results['results'][results['max_idx']]
        trails, _ = zip(*results['results'])
        cv_results = pd.DataFrame(trails, columns=['RandomForest'])
        # 可视化每个模型在不同的数据划分中的效果。
        plt.figure()
        sns.boxplot(data=cv_results)
        plt.ylabel('AUC %')
        plt.xlabel('Model Nmae')
        plt.savefig(f'./result/model_cv.png', bbox_inches='tight')
        cv_results.to_csv('./cv_result.csv', index=False)

        self.model.fit(X_train_sel, y_train_sel['label'])

        predictions = [[(self.model.predict(X_train_sel), self.model.predict(X_test_sel))]]
        pred_scores = [[(self.model.predict_proba(X_train_sel), self.model.predict_proba(X_test_sel))]]

        metric = []
        pred_sel_idx = []
        result = {'train': {}, 'test': {}}
        for label, prediction, scores in zip(['label'], predictions, pred_scores):
            pred_sel_idx_label = []
            for mname, (train_pred, test_pred), (train_score, test_score) in zip('RandomForest', prediction, scores):
                # 计算训练集指数
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_train_sel[label],
                    train_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres, f"{label}-train"))
                result['train']['model_name'] = mname
                result['train']['Accuracy'] = acc
                result['train']['AUC'] = auc
                result['train']['95% CI'] = ci
                result['train']['Sensitivity'] = tpr
                result['train']['Specificity'] = tnr
                result['train']['PPV'] = ppv
                result['train']['NPV'] = npv
                result['train']['Precision'] = precision
                result['train']['Recall'] = recall
                result['train']['F1'] = f1
                result['train']['Threshold'] = thres
                # 计算验证集指标
                acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_test_sel[label],
                                                                                                      test_score[:, 1])
                ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
                metric.append((mname, acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres, f"{label}-test"))
                result['test']['model_name'] = mname
                result['test']['Accuracy'] = acc
                result['test']['AUC'] = auc
                result['test']['95% CI'] = ci
                result['test']['Sensitivity'] = tpr
                result['test']['Specificity'] = tnr
                result['test']['PPV'] = ppv
                result['test']['NPV'] = npv
                result['test']['Precision'] = precision
                result['test']['Recall'] = recall
                result['test']['F1'] = f1
                result['test']['Threshold'] = thres
                # 计算thres对应的sel idx
                pred_sel_idx_label.append(np.logical_or(test_score[:, 0] >= thres, test_score[:, 1] >= thres))

            pred_sel_idx.append(pred_sel_idx_label)
        metric = pd.DataFrame(metric, index=None, columns=['model_name', 'Accuracy', 'AUC', '95% CI',
                                                           'Sensitivity', 'Specificity',
                                                           'PPV', 'NPV', 'Precision', 'Recall', 'F1',
                                                           'Threshold', 'Task'])
        # 绘图
        plt.figure()
        plt.subplot(211)
        sns.barplot(x='model_name', y='Accuracy', data=metric, hue='Task')
        plt.subplot(212)
        sns.lineplot(x='model_name', y='Accuracy', data=metric, hue='Task')
        plt.savefig(f'./result/model_acc.png', bbox_inches='tight')


        # Plot all ROC curves

        image = draw_roc([np.array(y_train_sel['label']), np.array(y_test_sel['label'])],
                 list(pred_scores[0][0]),
                 labels=['Train', 'Test'], title=f"ROC")

        return ['./result/model_cv.png', './result/model_acc.png', image], result


    def check_pos_label_consistency(self, pos_label, y_true):
        # ensure binary classification if pos_label is not specified
        # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
        # triggering a FutureWarning by calling np.array_equal(a, b)
        # when elements in the two arrays are not comparable.
        classes = np.unique(y_true)
        if (pos_label is None and (
                classes.dtype.kind in 'OUS' or
                not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1])))):
            classes_repr = ", ".join(repr(c) for c in classes)
            raise ValueError(
                f"y_true takes value in {{{classes_repr}}} and pos_label is not "
                f"specified: either make y_true take value in {{0, 1}} or "
                f"{{-1, 1}} or pass pos_label explicitly."
            )
        elif pos_label is None:
            pos_label = 1.0

        return pos_label


    def create_clf_model(self, model_names):
        models = {}
        # 判断是纯字符串，使用默认参数进行配置
        if isinstance(model_names, (list, tuple)):
            if 'svm' in model_names or 'SVM' in model_names:
                models['SVM'] = SVC(probability=True, random_state=0)
            # KNN
            if 'knn' in model_names or 'KNN' in model_names:
                models['KNN'] = KNeighborsClassifier(algorithm='kd_tree')
            # DecisionTree
            if 'dt' in model_names or 'DecisionTree' in model_names:
                models['DecisionTree'] = DecisionTreeClassifier(max_depth=None,
                                                                min_samples_split=2, random_state=0)
            # RandomForest
            if 'rf' in model_names or 'RandomForest' in model_names:
                models['RandomForest'] = RandomForestClassifier(n_estimators=10, max_depth=None,
                                                                min_samples_split=2, random_state=0)
            # ExtraTree
            if 'et' in model_names or 'ExtraTrees' in model_names:
                models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                                            min_samples_split=2, random_state=0)
            # XGBoost
            if 'xgb' in model_names or 'XGBoost' in model_names:
                models['XGBoost'] = XGBClassifier(n_estimators=10, objective='binary:logistic',
                                                  use_label_encoder=False, eval_metric='error')
            # LightGBM
            if 'lgb' in model_names or 'LightGBM' in model_names:
                models['LightGBM'] = LGBMClassifier(n_estimators=10, max_depth=-1, objective='binary')

            # Multi layer perception
            if 'mlp' in model_names or 'MLP' in model_names:
                models['MLP'] = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, solver='sgd',
                                              random_state=0)

            if 'lr' in model_names or 'LR' in model_names:
                models['LR'] = LogisticRegression(random_state=0)
        return models


    def get_bst_split(self, X_data, y_data, models, test_size=0.2, metric_fn=accuracy_score, n_trails=10,
                      cv=False, shuffle=False, metric_cut_off: float = None, random_state=None,
                      use_smote=False, **kwargs):

        results = []
        max_model = None
        max_model_name = None
        max_idx = 0
        max_metric = None
        metrics = {}
        dataset = []
        if not isinstance(X_data, pd.DataFrame) or not isinstance(y_data, pd.DataFrame):
            X_data = pd.DataFrame(X_data)
            y_data = pd.DataFrame(y_data)
            print('你的数据不是DataFrame类型，可能遇到未知错误！')
        if cv:  # 这里是使用交叉验证的方法
            # 分层K折交叉验证器， 提供训练集或测试集索引以将数据切分为训练集或测试集。
            skf = StratifiedKFold(n_splits=n_trails, shuffle=shuffle or random_state is not None,
                                  random_state=random_state)
            for train_index, test_index in skf.split(X_data, y_data):
                X_train, X_test = X_data.loc[train_index], X_data.loc[test_index]  # 划分数据集
                y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]  # 划分数据集
                dataset.append([X_train, X_test, y_train, y_test])
        for idx in range(n_trails):
            trail = []
            if cv:
                X_train, X_test, y_train, y_test = dataset[idx]
            else:
                rs = None if random_state is None else (idx + random_state)
                X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size,
                                                                    random_state=rs)
            X_train_smote, y_train_smote = X_train, y_train
            for model_name, model in models.items():
                # model.fit(X_train, y_train)
                # sample_weight = [1 if i == 0 else 0.5 for i in list(np.array(y_train))]
                # 训练模型
                if kwargs:
                    try:
                        model.fit(X_train_smote, y_train_smote, **kwargs)
                        print(f'正在训练{model_name}, 使用{kwargs}。')
                    except Exception as e:
                        model.fit(X_train_smote, y_train_smote)
                        print(f'因为：{e}，训练{model_name}使用{kwargs}失败。')
                else:
                    model.fit(X_train_smote, y_train_smote)
                y_pred = model.predict(X_test)
                if metric_fn == roc_auc_score:
                    y_proba = model.predict_proba(X_test)[:, 1]  # 预测X的类概率
                    metric = metric_fn(y_test, y_proba)
                else:
                    metric = metric_fn(y_test, y_pred)
                if model_name not in metrics:
                    metrics[model_name] = []
                metrics[model_name].append((idx, metric))
                if max_metric is None or metric > max_metric:
                    max_metric = metric
                    max_idx = idx
                    max_model = model
                    max_model_name = model_name
                trail.append(metric)
            results.append((trail, (X_train, X_test, y_train, y_test)))
            # 当满足用户需求的时候，可以停止。
            if metric_cut_off is not None and max_metric is not None and max_metric > metric_cut_off:
                print(f'Get best split cut off on {idx + 1} trails!')
                break
        return {'max_idx': max_idx, "max_model": max_model, "max_metric": max_metric, 'max_model_name': max_model_name,
                "results": results, 'metrics': metrics}


if __name__ == '__main__':
    obj = Tools()
    df = pd.read_csv(r'C:\Users\Administrator\Desktop\DLApp\normalize.csv')
    columns = [column for column in df.columns.tolist() if column not in ['ID', 'label']]
    df2 = obj.clinic_statistic(df, columns)
    df2.to_csv('./666.csv')